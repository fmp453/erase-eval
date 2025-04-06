# Bootstrapped from:
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/cli_lora_pti.py


import os
import re
import math
import random
import itertools
from typing import Optional, Literal, Any, Union
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.utils import set_seed
from safetensors.torch import safe_open
from tqdm import trange

from train_methods.lora_save import save_all

OBJECT_TEMPLATE = ["a photo of {}"]
STYLE_TEMPLATE = ["a photo in the style of {}"]
NAKED_TEMPLATE = ["a photo of naked"]
EMBED_FLAG = "<embed>"


class PivotalTuningDatasetCapation(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        token_map: Optional[dict] = None,
        use_template: Optional[str] = None,
        class_data_root=None,
        class_prompt=None,
        size=512,
        h_flip=True,
        color_jitter=False,
        resize=True,
        blur_amount: int = 70,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.resize = resize

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.token_map = token_map

        self.use_template = use_template

        if use_template == "naked":
            self.templates = NAKED_TEMPLATE
        elif use_template == "style":
            self.templates = STYLE_TEMPLATE
        else:
            self.templates = OBJECT_TEMPLATE

        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
        self.h_flip = h_flip
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(0.1, 0.1)
                if color_jitter
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.blur_amount = blur_amount

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.use_template:
            assert self.token_map is not None
            input_tok = list(self.token_map.values())[0]

            text = random.choice(self.templates).format(input_tok)
        else:
            text = self.instance_images_path[index % self.num_instance_images].stem
            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text = text.replace(token, value)
        
        if self.h_flip and random.random() > 0.5:
            hflip = transforms.RandomHorizontalFlip(p=1)

            example["instance_images"] = hflip(example["instance_images"])
            
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        
        example["uncond_prompt_ids"] = self.tokenizer(
            "",
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids        

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

def get_models(pretrained_model_name_or_path: str, placeholder_tokens: list[str], initializer_tokens: list[str], device="cuda:0"):

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")

    placeholder_token_ids = []

    for token, init_tok in zip(placeholder_tokens, initializer_tokens):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids.append(placeholder_token_id)

        # Load models and create wrapper for stable diffusion
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):
            # <rand-"sigma">, e.g. <rand-0.5>
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[0]) * sigma_val
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        elif init_tok.isnumeric():
            token_embeds[placeholder_token_id] = token_embeds[int(init_tok)]
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")        
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_ids,
    )

def text2img_dataloader(train_dataset, train_batch_size: int, tokenizer: CLIPTokenizer):
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        uncond_ids = [example["uncond_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if examples[0].get("class_prompt_ids", None) is not None:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        uncond_ids = tokenizer.pad(
            {"input_ids": uncond_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "uncond_ids":uncond_ids,
            "pixel_values": pixel_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_dataloader


def loss_step(batch, unet: UNet2DConditionModel, vae: AutoencoderKL, text_encoder: CLIPTextModel, scheduler: DDPMScheduler, t_mutliplier=1.0):
    weight_dtype = torch.float32

    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype).to(unet.device)).latent_dist.sample()
    latents = latents * 0.18215

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    timesteps = torch.randint(0, int(scheduler.config.num_train_timesteps * t_mutliplier), (bsz,), device=latents.device)
    timesteps = timesteps.long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device))[0]
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

    if batch.get("mask", None) is not None:
        mask = batch["mask"].to(model_pred.device).reshape(model_pred.shape[0], 1, batch["mask"].shape[2], batch["mask"].shape[3])
    
        # resize to match model_pred
        mask = F.interpolate(mask.float(), size=model_pred.shape[-2:], mode="nearest") + 0.05
        
        mask = mask / mask.mean()

        model_pred = model_pred * mask

        target = target * mask

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss


def train_inversion(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    dataloader,
    num_steps: int,
    scheduler,
    index_no_updates,
    optimizer: optim.Optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path: str,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    accum_iter: int = 1,
    clip_ti_decay: bool = True
):

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    # Original Emb for TI
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()
    
    index_updates = ~index_no_updates
    loss_sum = 0.0

    for epoch in range(math.ceil(num_steps / len(dataloader))):
        unet.eval()
        text_encoder.train()
        for batch in dataloader:

            lr_scheduler.step()

            with torch.set_grad_enabled(True):
                loss = loss_step(batch, unet, vae, text_encoder, scheduler) / accum_iter
                loss.backward()
                loss_sum += loss.detach().item()

                if global_step % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    with torch.no_grad():

                        # normalize embeddings
                        if clip_ti_decay:
                            pre_norm = text_encoder.get_input_embeddings().weight[index_updates, :].norm(dim=-1, keepdim=True)
                        
                            lambda_ = min(1.0, 100 * lr_scheduler.get_last_lr()[0])
                            text_encoder.get_input_embeddings().weight[index_updates] = F.normalize(text_encoder.get_input_embeddings().weight[index_updates, :], dim=-1) * (pre_norm + lambda_ * (0.4 - pre_norm))
                            # print(pre_norm)

                        current_norm = text_encoder.get_input_embeddings().weight[index_updates, :].norm(dim=-1)
                        
                        text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

                        # print(f"Current Norm : {current_norm}")

                global_step += 1
                progress_bar.update(1)

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if global_step % save_steps == 0:
                save_all(
                    unet=unet,
                    text_encoder=text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(save_path, f"step_inv_{global_step}.safetensors"),
                    save_lora=False,
                )

            if global_step >= num_steps:
                return

def ti_component(
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    use_template: Literal[None, "object", "style", "naked"] = None,
    placeholder_tokens: str = "<s>",
    placeholder_token_at_data: Optional[str] = None,
    initializer_tokens: Optional[str] = None,
    class_prompt: Optional[str] = None,
    seed: int = 42,
    resolution: int = 512,
    color_jitter: bool = True,
    train_batch_size: int = 1,
    max_train_steps_ti: int = 1000,
    save_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    clip_ti_decay: bool = True,
    learning_rate_ti: float = 5e-4,
    scale_lr: bool = False,
    lr_scheduler: str = "linear",
    lr_warmup_steps: int = 0,
    weight_decay_ti: float = 0.00,
    device="cuda:0",
    extra_args: Optional[dict] = None,
):
    torch.manual_seed(seed)
    
    if output_dir is not None:
        output_dir = output_dir.replace(" ", "-")
        os.makedirs(output_dir, exist_ok=True)
    
    placeholder_tokens = placeholder_tokens.split("|")
    if initializer_tokens is None:
        print("PTI : Initializer Token not give, random inits")
        initializer_tokens = ["<rand-0.017>"] * len(placeholder_tokens)
    else:
        initializer_tokens = initializer_tokens.split("|")

    assert len(initializer_tokens) == len(placeholder_tokens), "Unequal Initializer token for Placeholder tokens."

    if placeholder_token_at_data is not None:
        tok, pat = placeholder_token_at_data.split("|")
        token_map = {tok: pat}

    else:
        token_map = {"DUMMY": "".join(placeholder_tokens)}

    print("Placeholder Tokens", placeholder_tokens)
    print("Initializer Tokens", initializer_tokens)

    # get the models
    text_encoder, vae, unet, tokenizer, placeholder_token_ids = get_models(
        pretrained_model_name_or_path,
        placeholder_tokens,
        initializer_tokens,
        device=device,
    )

    noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

    ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size if scale_lr else learning_rate_ti
    
    train_dataset = PivotalTuningDatasetCapation(
        instance_data_root=instance_data_dir,
        token_map=token_map,
        use_template=use_template,
        class_data_root=None,
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=resolution,
        color_jitter=color_jitter
    )

    train_dataset.blur_amount = 20

    train_dataloader = text2img_dataloader(train_dataset, train_batch_size, tokenizer)

    index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_ids[0]

    for tok_id in placeholder_token_ids:
        index_no_updates[tok_id] = False

    unet.requires_grad_(False)
    vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    # STEP 1 : Perform Inversion
    ti_optimizer = optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=ti_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay_ti,
    )

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=ti_optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps_ti,
    )

    train_inversion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        dataloader=train_dataloader,
        num_steps=max_train_steps_ti,
        scheduler=noise_scheduler,
        index_no_updates=index_no_updates,
        optimizer=ti_optimizer,
        save_steps=save_steps,
        placeholder_tokens=placeholder_tokens,
        placeholder_token_ids=placeholder_token_ids,
        save_path=output_dir,
        lr_scheduler=lr_scheduler,
        accum_iter=gradient_accumulation_steps,
        clip_ti_decay=clip_ti_decay,
    )

    del ti_optimizer

def parse_safeloras_embeds(safeloras) -> dict[str, torch.Tensor]:
    """
    Converts a loaded safetensor file that contains Textual Inversion embeds into
    a dictionary of embed_token: Tensor
    """
    embeds = {}
    metadata = safeloras.metadata()
    for key in safeloras.keys():
        # Only handle Textual Inversion embeds
        meta = metadata.get(key)
        if not meta or meta != EMBED_FLAG:
            continue

        embeds[key] = safeloras.get_tensor(key)

    return embeds

def apply_learned_embed_in_clip(
    learned_embeds,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    token: Optional[Union[str, list[str]]] = None,
    idempotent=False,
):
    if isinstance(token, str):
        trained_tokens = [token]
    elif isinstance(token, list):
        assert len(learned_embeds.keys()) == len(
            token
        ), "The number of tokens and the number of embeds should be the same"
        trained_tokens = token
    else:
        trained_tokens = list(learned_embeds.keys())

    for token in trained_tokens:
        print(token)
        embeds = learned_embeds[token]
        num_added_tokens = tokenizer.add_tokens(token)

        i = 1
        if not idempotent:
            while num_added_tokens == 0:
                print(f"The tokenizer already contains the token {token}.")
                token = f"{token[:-1]}-{i}>"
                print(f"Attempting to add the token {token}.")
                num_added_tokens = tokenizer.add_tokens(token)
                i += 1
        elif num_added_tokens == 0 and idempotent:
            print(f"The tokenizer already contains the token {token}.")
            print(f"Replacing {token} embedding.")

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token

class ForgetMeNotDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        size=512,
        center_crop=False,
        use_added_token= False,
        use_pooler=False,
        multi_concept=None,
        data_dir="fmn-data"
    ):  
        self.use_added_token = use_added_token
        self.use_pooler = use_pooler
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_path  = []
        self.instance_prompt  = []

        token_idx = 1
        for c, t, num_tok in multi_concept:
            p = Path(data_dir)
            if not p.exists():
                raise ValueError(f"Instance {p} images root doesn't exists.")                   
            
            image_paths = list(p.iterdir())
            self.instance_images_path += image_paths

            target_snippet = f"{''.join([ f'<s{token_idx + i}>' for i in range(num_tok)])}" if use_added_token else c.replace("-", " ")
            if t == "object":
                self.instance_prompt += [(f"a photo of {target_snippet}", target_snippet)] * len(image_paths)
            elif t == "style":
                self.instance_prompt += [(f"a photo in the style of {target_snippet}", target_snippet)] * len(image_paths)
            else:
                raise ValueError("unknown concept type!")
            if use_added_token:
                token_idx += num_tok
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_prompt, target_tokens = self.instance_prompt[index % self.num_instance_images]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_prompt"] = instance_prompt
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        prompt_ids = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length
        ).input_ids

        concept_ids = self.tokenizer(
            target_tokens,
            add_special_tokens=False
        ).input_ids             

        pooler_token_id = self.tokenizer(
            "<|endoftext|>",
            add_special_tokens=False
        ).input_ids[0]

        concept_positions = [0] * self.tokenizer.model_max_length
        for i, tok_id in enumerate(prompt_ids):
            if tok_id == concept_ids[0] and prompt_ids[i:i + len(concept_ids)] == concept_ids:
                concept_positions[i:i + len(concept_ids)] = [1]*len(concept_ids)
            if self.use_pooler and tok_id == pooler_token_id:
                concept_positions[i] = 1
        example["concept_positions"] = torch.tensor(concept_positions)[None]               

        return example

def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    concept_positions = [example["concept_positions"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    instance_prompts =  [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    concept_positions = torch.cat(concept_positions, dim=0).type(torch.BoolTensor)

    batch = {
        "instance_prompts": instance_prompts,
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "concept_positions": concept_positions
    }
    return batch

def attn_component(
    instance_data_dir: str,
    seed: int,
    output_dir: str,
    pretrained_model_name_or_path: str,
    multi_concept: list[str],
    scale_lr: bool,
    learning_rate: float,
    gradient_accumulation_steps: int,
    train_batch_size: int,
    only_optimize_ca: bool,
    resolution: int,
    center_crop: bool,
    use_pooler: bool,
    dataloader_num_workers: int,
    max_train_steps: int,
    num_train_epochs: int,
    lr_scheduler: str,
    lr_warmup_steps: int,
    lr_num_cycles: int,
    lr_power: float,
    no_real_image: bool,
    max_grad_norm: float,
    device: str="cuda:0"
):

    output_dir = output_dir.replace(" ", "-")
    
    if seed is not None:
        set_seed(seed)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    
    tok_idx = 1
    multi_concepts = []
    for c, t in multi_concept:
        token = None
        idempotent_token = True
        models_dir = output_dir.split("/")[0]
        weight_path = f"{models_dir}/{c.replace(' ', '-')}/fmn/{c.replace(' ', '-')}-ti/step_inv_500.safetensors"
        safeloras = safe_open(weight_path, framework="pt", device="cpu")
        tok_dict = parse_safeloras_embeds(safeloras)
        
        tok_dict = {f"<s{tok_idx + i}>":tok_dict[k] for i, k in enumerate(sorted(tok_dict.keys()))}
        tok_idx += len(tok_dict.keys())
        multi_concepts.append([c, t, len(tok_dict.keys())])

        print("---Adding Tokens---:",c, t)
        apply_learned_embed_in_clip(
            tok_dict,
            text_encoder,
            tokenizer,
            token=token,
            idempotent=idempotent_token,
        )
    multi_concept = multi_concepts

    class AttnController:
        def __init__(self) -> None:
            self.attn_probs = []
            self.logs = []
            self.concept_positions = None
        def __call__(self, attn_prob, m_name) -> Any:
            bs, _ = self.concept_positions.shape
            head_num = attn_prob.shape[0] // bs
            target_attns = attn_prob.masked_select(self.concept_positions[:,None,:].repeat(head_num, 1, 1)).reshape(-1, self.concept_positions[0].sum())
            self.attn_probs.append(target_attns)
            self.logs.append(m_name)
        def set_concept_positions(self, concept_positions):
            self.concept_positions = concept_positions
        def loss(self):
            return torch.cat(self.attn_probs).norm()
        def zero_attn_probs(self):
            self.attn_probs = []
            self.logs = []
            self.concept_positions = None

    class MyCrossAttnProcessor:
        def __init__(self, attn_controller: "AttnController", module_name) -> None:
            self.attn_controller = attn_controller
            self.module_name = module_name
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

            query = attn.to_q(hidden_states)
            query = attn.head_to_batch_dim(query)

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
        
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            self.attn_controller(attention_probs, self.module_name)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states

    attn_controller = AttnController()
    module_count = 0
    for n, m in unet.named_modules():
        if n.endswith('attn2'):
            m.set_processor(MyCrossAttnProcessor(attn_controller, n))
            module_count += 1
    print(f"cross attention module count: {module_count}")
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    if scale_lr:
        learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size

    # Optimizer creation
    if only_optimize_ca:
        params_to_optimize = (itertools.chain([p for n, p in unet.named_parameters() if 'attn2' in n]))
        print("only optimize cross attention...")
    else:
        params_to_optimize = (itertools.chain(unet.parameters()))
        print("optimize unet...")
    
    # these are default values except lr
    optimizer = optim.AdamW(
        params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )

    train_dataset = ForgetMeNotDataset(
        tokenizer=tokenizer,
        size=resolution,
        center_crop=center_crop,
        use_pooler=use_pooler,
        use_added_token=True,
        multi_concept=multi_concept,
        data_dir=instance_data_dir
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    unet.to(device)
    vae.to(device)
    text_encoder.to(device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * gradient_accumulation_steps

    print("***** Running training *****")
    print(f"{len(train_dataset)=}")
    print(f"{len(train_dataloader)=}")
    print(f"{num_train_epochs=}")
    print(f"{train_batch_size=}")
    print(f"{total_batch_size=}")
    print(f"{gradient_accumulation_steps=}")
    print(f"{max_train_steps=}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = trange(global_step, max_train_steps)
    progress_bar.set_description("Steps")

    debug_once = True
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # show
            if debug_once:
                print(batch["instance_prompts"][0])
                debug_once = False
            
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            if no_real_image:
                noisy_latents = noise_scheduler.add_noise(torch.zeros_like(noise), noise, timesteps)
            else:
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                text_embedding = text_encoder(batch["input_ids"].to(text_encoder.device))[0]
            
            attn_controller.set_concept_positions(batch["concept_positions"].to(unet.device))

            _ = unet(noisy_latents, timesteps, text_embedding).sample
            loss = attn_controller.loss()

            loss.backward()
            clip_grad_norm_(parameters=params_to_optimize, max_norm=max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=False)
            attn_controller.zero_attn_probs()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # output_dir: models/CONCEPT_NAME/fmn/CONCEPT_NAME-attn
    unet.save_pretrained(Path(output_dir).parent)

