# SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation

# ref: https://github.com/nannullna/safe-diffusion/blob/main/train_sdd.py

import os
import random
import shutil
import warnings
from typing import Union

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler, AutoencoderKL, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from datasets import load_dataset

from utils import Arguments
from train_methods.train_utils import prepare_extra_step_kwargs, sample_until

warnings.filterwarnings("ignore")

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}
    
def gather_parameters(args: Arguments, unet: UNet2DConditionModel) -> tuple[list[str], list[torch.nn.Parameter]]:
    """Gather the parameters to be optimized by the optimizer."""
    names, parameters = [], []
    for name, param in unet.named_parameters():
        if args.salun_method == "full":
            # Train all layers.
            names.append(name)
            parameters.append(param)
        elif args.salun_method == "selfattn":
            # Attention layer 1 is the self-attention layer.
            if "attn1" in name:
                names.append(name)
                parameters.append(param)
        elif args.salun_method == "xattn":
            # Attention layer 2 is the cross-attention layer.
            if "attn2" in name:
                names.append(name)
                parameters.append(param)
        elif args.salun_method == "noxattn":
            # Train all layers except the cross attention and time_embedding layers.
            if name.startswith("conv_out.") or ("time_embed" in name):
                # Skip the time_embedding layer.
                continue
            elif "attn2" in name:
                # Skip the cross attention layer.
                continue
            names.append(name)
            parameters.append(param)
        elif args.salun_method == "notime":
            # Train all layers except the time_embedding layer.
            if name.startswith("conv_out.") or ("time_embed" in name):
                continue
            names.append(name)
            parameters.append(param)
        else:
            raise ValueError(f"Unknown finetuning method: {args.finetuning_method}")

    return names, parameters

@torch.no_grad()
def encode_prompt(
    prompt: Union[str, list[str]]=None,
    negative_prompt: Union[str, list[str]]=None,
    removing_prompt: Union[str, list[str]]=None,
    num_images_per_prompt: int=1,
    text_encoder: CLIPTextModel=None,
    tokenizer: CLIPTokenizer=None,
    device: torch.device=None,
):
    """Encode a prompt into a text embedding. Prompt can be None."""
    # Get text embeddings for unconditional and conditional prompts.
    if isinstance(prompt, str):
        prompt = [prompt]
    
    if removing_prompt is not None and isinstance(removing_prompt, str):
        removing_prompt = [removing_prompt]
        assert len(prompt) == len(removing_prompt), f"Safety concept must be the same length as prompt of length {len(prompt)}."
    
    if negative_prompt is not None and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]
        assert len(prompt) == len(negative_prompt), f"Negative prompt must be the same length as prompt of length {len(prompt)}."

    batch_size = len(prompt) if prompt is not None else 1

    use_attention_mask = hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask
    device = device if device is not None else text_encoder.device

    # Tokenization
    uncond_input = tokenizer([""] * batch_size if negative_prompt is None else negative_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    if prompt is not None:
        prompt_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    else:
        prompt_input = None
    
    if removing_prompt is not None:
        removing_input = tokenizer(removing_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    else:
        removing_input = None

    # Encoding
    prompt_embeds = text_encoder(
        input_ids=uncond_input["input_ids"].to(device),
        attention_mask=uncond_input["attention_mask"].to(device) if use_attention_mask else None,
    )[0]
    if prompt_input is not None:
        prompt_emb = text_encoder(
            input_ids=prompt_input["input_ids"].to(device),
            attention_mask=prompt_input["attention_mask"].to(device) if use_attention_mask else None,
        )[0]
        prompt_embeds = torch.cat([prompt_embeds, prompt_emb], dim=0)
    
    if removing_input is not None:
        removing_emb = text_encoder(
            input_ids=removing_input["input_ids"].to(device),
            attention_mask=removing_input["attention_mask"].to(device) if use_attention_mask else None,
        )[0]
        prompt_embeds = torch.cat([prompt_embeds, removing_emb], dim=0)

    # Duplicate the embeddings for each image.
    if num_images_per_prompt > 1:
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.reshape(batch_size * num_images_per_prompt, seq_len, -1)
    
    return prompt_embeds

def train_step(
    args: Arguments,
    prompt: str,
    removing_prompt: str,
    generator: torch.Generator,
    noise_scheduler: DDPMScheduler,
    ddim_scheduler: DDIMScheduler,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet_teacher: UNet2DConditionModel,
    unet_student: UNet2DConditionModel,
    devices: list[torch.device]
) -> torch.Tensor:
    """Train the model a single step for a given prompt and return the loss."""

    unet_student.train()

    # Encode prompt
    prompt_embeds = encode_prompt(
        prompt=prompt, 
        removing_prompt=removing_prompt,
        text_encoder=text_encoder, 
        tokenizer=tokenizer,
        device=devices[1],
    )
    
    uncond_emb, cond_emb, safety_emb = torch.chunk(prompt_embeds, 3, dim=0)
    batch_size = cond_emb.shape[0]

    # Prepare timesteps
    noise_scheduler.set_timesteps(args.ddpm_steps, devices[1])

    # Prepare latent codes to generate z_t
    latent_shape = (batch_size, unet_teacher.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator, device=devices[0])
    # Scale the initial noise by the standard deviation required by the scheduler
    latents = latents * ddim_scheduler.init_noise_sigma # z_T

    # Normally, DDPM takes 1,000 timesteps for training, and DDIM takes 50 timesteps for inference.
    t_ddim = torch.randint(0, args.ddim_steps, (1,))
    t_ddpm_start = round((1 - (int(t_ddim) + 1) / args.ddim_steps) * args.ddpm_steps)
    t_ddpm_end   = round((1 - int(t_ddim)       / args.ddim_steps) * args.ddpm_steps)
    t_ddpm = torch.randint(t_ddpm_start, t_ddpm_end, (batch_size,),)
    
    # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, args.salun_eta)

    with torch.no_grad():
        # args.guidance_scale: s_g in the paper
        prompt_embeds = torch.cat([uncond_emb, cond_emb], dim=0) if args.guidance_scale > 1.0 else uncond_emb
        prompt_embeds = prompt_embeds.to(unet_student.device)

        # Generate latents
        latents = sample_until(
            until=int(t_ddim),
            latents=latents,
            unet=unet_student,
            scheduler=ddim_scheduler,
            prompt_embeds=prompt_embeds,
            guidance_scale=args.guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
        )

        # Stop-grad and send to the second device
        _latents = latents.to(devices[1])
        e_0 = unet_teacher(_latents, t_ddpm.to(devices[1]), encoder_hidden_states=uncond_emb).sample
        e_p = unet_teacher(_latents, t_ddpm.to(devices[1]), encoder_hidden_states=safety_emb).sample

        e_0 = e_0.detach().to(devices[0])
        e_p = e_p.detach().to(devices[0])

        # args.concept_scale: s_s in the paper
        noise_target = e_0 - args.negative_guidance * (e_p - e_0)

    noise_pred = unet_student(latents, t_ddpm.to(devices[0]), encoder_hidden_states=safety_emb.to(devices[0])).sample

    loss = F.mse_loss(noise_pred, noise_target)
    
    return loss

def salun(args: Arguments, mask_path: str):
    # You may provide a single file path, or a list of concepts
    if len(args.concepts) == 1 and args.concepts[0].endswith(".txt"):
        with open(args.concepts[0], "r") as f:
            args.concepts = f.read().splitlines()

    devices = args.device.split(",")
    if len(devices) > 1:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[1]}")]
    else:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[0]}")]

    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    ddim_scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    unet_teacher: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    unet_student: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")

    # Freeze vae and text_encoder
    unet_teacher.requires_grad_(False)
    text_encoder.requires_grad_(False)

    _, parameters = gather_parameters(args, unet_student)
    num_train_param = sum(p.numel() for p in parameters)
    num_total_param = sum(p.numel() for p in unet_student.parameters())
    print(f"Finetuning parameters: {num_train_param} / {num_total_param} ({num_train_param / num_total_param:.2%})")

    # Create optimizer and scheduler
    # use default values except lr
    optimizer = optim.Adam(parameters, lr=args.salun_lr)
    lr_scheduler: LambdaLR = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.salun_lr_warmup_steps,
        num_training_steps=args.salun_iter,
    )

    # First device -- unet_student, generator
    # Second device -- unet_teacher, text_encoder
    unet_student = unet_student.to(devices[0])
    gen = torch.Generator(device=devices[0])

    unet_teacher = unet_teacher.to(devices[1])
    text_encoder = text_encoder.to(devices[1])

    # Set the number of inference time steps
    ddim_scheduler.set_timesteps(args.ddim_steps, devices[1])
    progress_bar = trange(1, args.salun_iter+1, desc="Training")

    mask = torch.load(mask_path)

    for step in progress_bar:

        removing_concept = random.choice(args.concepts)
        removing_prompt = removing_concept
        prompt = removing_prompt

        unet_student.train()
        
        train_loss = train_step(
            args=args,
            prompt=prompt,
            removing_prompt=removing_prompt,
            generator=gen,
            noise_scheduler=noise_scheduler,
            ddim_scheduler=ddim_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet_teacher=unet_teacher,
            unet_student=unet_student,
            devices=devices,
        )
        
        train_loss.backward()
        
        if args.max_grad_norm > 0:
            clip_grad_norm_(parameters, args.max_grad_norm)

        for name, params in unet_student.named_parameters():
            if params.grad is not None:
                params.grad *= mask[name.split("model.diffusion.model.")[-1]].to(devices[0])

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_description(f"Training: {train_loss.item():.4f}")
        
    unet_student.eval()
    unet_student.save_pretrained(args.save_dir)

class Imagenette(Dataset):
    def __init__(self, split, class_to_forget=None, transform=None):
        self.dataset = load_dataset("frgfm/imagenette", "160px")[split]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.dataset.features["label"].names)}
        self.file_to_class = {str(idx): self.dataset["label"][idx] for idx in range(len(self.dataset))}

        self.class_to_forget = class_to_forget
        self.num_classes = max(self.class_to_idx.values()) + 1
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if example["label"] == self.class_to_forget:
            label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)
        return image, label

class NSFW(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("data/nsfw")["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image

class NOT_NSFW(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("data/not-nsfw")["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image

class CustomData(Dataset):
    def __init__(self, data_path, transform=None):
        self.dataset = load_dataset("imagefolder", data_dir=data_path, split="train")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if self.transform:
            image = self.transform(image)

        return image

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=interpolation),
            transforms.CenterCrop(size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform

def get_imagenette_label_from_concept(concept):
    l = ["tench", "English springer", "cassette player", "chainsaw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
    d = {}
    for i in range(len(l)):
        d[l[i]] = i
    
    return d[concept]

def setup_forget_data(concept, batch_size, image_size, args: Arguments, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    num_images = 800

    if concept in ["tench", "English springer", "cassette player", "chainsaw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]:
        train_set = Imagenette("train", transform=transform)
        class_to_forget = get_imagenette_label_from_concept(concept)
        descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
        filtered_data = [data for data in train_set if data[1] == class_to_forget]
        train_dl = DataLoader(filtered_data, batch_size=batch_size)
        return train_dl, descriptions
    else:
        descriptions = f"an image of a {concept}"
        print("generating images")
        pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(args.sd_version)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        num_images_per_prompt = 5
        device = args.device.split(",")[0]
        device = f"cuda:{device}"
        pipe.to(device)
        os.makedirs("salun-data/train", exist_ok=True)
        os.makedirs("salun-data/test", exist_ok=True)
        os.makedirs("salun-data/val", exist_ok=True)
        for i in range(num_images // num_images_per_prompt):
            generator = torch.Generator(device).manual_seed(args.seed)
            images = pipe(descriptions, guidance_scale=args.guidance_scale, num_images_per_prompt=num_images_per_prompt, generator=generator).images

            for j in range(num_images_per_prompt):
                images[j].save(f"salun-data/train/{concept.replace(' ', '-')}-{i * num_images_per_prompt + j:03}.png")
            
            if i % 20 == 0:
                print(f"{i / 160 * 100}% finished.")

        train_set = CustomData("salun-data/train", transform)
        train_dl = DataLoader(train_set, batch_size=batch_size)
        return train_dl, descriptions

def setup_forget_nsfw_data(batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    forget_set = NSFW(transform=transform)
    forget_dl = DataLoader(forget_set, batch_size=batch_size)

    remain_set = NOT_NSFW(transform=transform)
    remain_dl = DataLoader(remain_set, batch_size=batch_size)
    return forget_dl, remain_dl

def generate_mask(args: Arguments):
    
    device = torch.device(f'cuda:{args.device.split(",")[0]}')
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    train_dl, descriptions = setup_forget_data(args.concepts, args.salun_masking_batch_size, args.image_size, args)

    text_encoder.eval()
    vae.eval()
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.salun_masking_lr)

    gradients = {}
    for name, param in unet.named_parameters():
        gradients[name] = 0

    pbar = trange(len(train_dl))
    is_imagenette = args.concepts in ["tench", "English springer", "cassette player", "chainsaw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
    for _ in pbar:

        optimizer.zero_grad()

        if is_imagenette:
            images, labels = next(iter(train_dl))
            null_prompts = ["" for _ in labels]
            prompts = [descriptions[label] for label in labels]
        else:
            images = next(iter(train_dl))
            null_prompts = [""] * args.salun_masking_batch_size
            prompts = [descriptions] * args.salun_masking_batch_size

        images = images.to(device)
    
        with torch.no_grad():
            forget_input = vae.encode(images).latent_dist.sample()
            forget_ids = tokenizer(prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
            forget_emb = text_encoder(forget_ids.to(text_encoder.device))[0]
            null_ids = tokenizer(null_prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
            null_emb = text_encoder(null_ids.to(text_encoder.device))[0]

        t = torch.randint(0, scheduler.config.num_train_timesteps, (1, )).long()
        t = t.to(device)

        noise = torch.randn_like(forget_input, device=device)

        forget_noisy = scheduler.add_noise(forget_input, noise, t)

        forget_out = unet(forget_noisy, t, forget_emb).sample
        null_out = unet(forget_noisy, t, null_emb).sample

        preds = (1 + args.guidance_scale) * forget_out - args.guidance_scale * null_out
        loss = -criteria(noise, preds)

        loss.backward()

        with torch.no_grad():
            for name, param in unet.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data.cpu()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        mask_path = os.path.join("mask", args.concepts)
        os.makedirs(mask_path, exist_ok=True)

        threshold = 0.5
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * threshold)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements
        torch.save(hard_dict, res:=os.path.join(mask_path, f"with_{str(threshold)}.pt"))
    
    return res

def generate_nsfw_mask(args: Arguments):
    device = torch.device(f'cuda:{args.device.split(",")[0]}')
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    
    train_dl, _ = setup_forget_nsfw_data(args.salun_masking_batch_size, args.image_size)

    text_encoder.eval()
    vae.eval()

    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.salun_masking_lr)
    word_nude = "a photo of a nude person"

    gradients = {}
    for name, param in unet.named_parameters():
        gradients[name] = 0

    pbar = trange(len(train_dl))
    for _ in pbar:
        images = next(iter(train_dl))
        optimizer.zero_grad()

        images = images.to(device)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()

        null_prompts = [""] * args.salun_masking_batch_size
        prompts = [word_nude] * args.salun_masking_batch_size
        
        with torch.no_grad():
            forget_input = vae.encode(images).latent_dist.sample()
            forget_ids = tokenizer(prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids[0]
            forget_emb = text_encoder(forget_ids.to(text_encoder.device))[0]

            null_ids = tokenizer(null_prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids[0]
            null_emb = text_encoder(null_ids.to(text_encoder.device))[0]

        noise = torch.randn_like(forget_input, device=device)

        forget_noisy = scheduler.add_noise(forget_input, noise, t)

        forget_out = unet(forget_noisy, t, forget_emb).sample
        null_out = unet(forget_noisy, t, null_emb).sample

        preds = (1 + args.guidance_scale) * forget_out - args.guidance_scale * null_out

        loss = - criteria(noise, preds)
        loss.backward()

        with torch.no_grad():
            for name, param in unet.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data.cpu()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        threshold = 0.5
        # for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * threshold)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, res:=os.path.join(f"mask/nude_{threshold}.pt"))

    return res

def masking(args: Arguments) -> str:
    if args.is_nsfw:
        return generate_nsfw_mask(args)
    else:
        return generate_mask(args)
    

def main(args: Arguments):
    mask_path = masking(args)
    salun(args, mask_path)

    if os.path.isdir("salun-data"):
        shutil.rmtree("salun-data")
