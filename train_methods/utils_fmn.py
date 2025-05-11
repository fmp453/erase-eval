# Bootstrapped from:
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/cli_lora_pti.py


import os
import re
import math
import itertools
from typing import Optional, Literal, Any, Union
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.utils import set_seed
from safetensors.torch import safe_open
from tqdm import trange

from train_methods import train_utils
from train_methods.lora_save import save_all
from train_methods.data import ForgetMeNotDataset, FMNPivotalTuningDataset
from utils import Arguments

EMBED_FLAG = "<embed>"


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

    latents = vae.encode(batch["pixel_values"].to(unet.device)).latent_dist.sample()
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

                        current_norm = text_encoder.get_input_embeddings().weight[index_updates, :].norm(dim=-1)
                        
                        text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

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

    text_encoder, vae, unet, tokenizer, placeholder_token_ids = get_models(
        pretrained_model_name_or_path,
        placeholder_tokens,
        initializer_tokens,
        device=device,
    )
    noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")
    ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size if scale_lr else learning_rate_ti
    
    train_dataset = FMNPivotalTuningDataset(
        instance_data_root=instance_data_dir,
        token_map=token_map,
        use_template=use_template,
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
    learned_embeds: dict[str, torch.Tensor],
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    token: Optional[Union[str, list[str]]]=None,
    idempotent: bool=False,
):
    if isinstance(token, str):
        trained_tokens = [token]
    elif isinstance(token, list):
        assert len(learned_embeds.keys()) == len(token), "The number of tokens and the number of embeds should be the same"
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

        text_encoder.resize_token_embeddings(len(tokenizer))

        # get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token

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
    args: Arguments,
    output_dir: str,
    multi_concept: list[str],
    device: str="cuda:0"
):

    output_dir = output_dir.replace(" ", "-")
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer, text_encoder, vae, unet, _, noise_scheduler = train_utils.get_models(args)
    
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
    
    if args.fmn_scale_lr:
        learning_rate = args.fmn_lr_attn * args.fmn_gradient_accumulation_steps * args.fmn_train_batch_size

    # Optimizer creation
    if args.only_optimize_ca:
        params_to_optimize = (itertools.chain([p for n, p in unet.named_parameters() if 'attn2' in n]))
        print("only optimize cross attention...")
    else:
        params_to_optimize = (itertools.chain(unet.parameters()))
        print("optimize unet...")
    
    # these are default values except lr
    optimizer = optim.AdamW(
        params_to_optimize,
        lr=learning_rate,
    )

    train_dataset = ForgetMeNotDataset(
        tokenizer=tokenizer,
        size=args.image_size,
        center_crop=args.center_crop,
        use_pooler=args.use_pooler,
        multi_concept=multi_concept,
        data_dir=args.instance_data_dir
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.fmn_train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.fmn_dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.fmn_gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.fmn_lr_warmup_steps_attn * args.fmn_gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.fmn_gradient_accumulation_steps,
        num_cycles=args.fmn_lr_num_cycles_attn,
        power=args.fmn_lr_power_attn,
    )

    unet.to(device)
    vae.to(device)
    text_encoder.to(device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.fmn_gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    print("***** Running training *****")
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
                latents: torch.Tensor = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                text_embedding = text_encoder(batch["input_ids"].to(text_encoder.device))[0]
            
            attn_controller.set_concept_positions(batch["concept_positions"].to(unet.device))

            _ = unet(noisy_latents, timesteps, text_embedding).sample
            loss = attn_controller.loss()

            loss.backward()
            clip_grad_norm_(parameters=params_to_optimize, max_norm=args.max_grad_norm)
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
