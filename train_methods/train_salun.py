# SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation
# ref: https://github.com/nannullna/safe-diffusion/blob/main/train_sdd.py

import random
import shutil
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler, AutoencoderKL, StableDiffusionPipeline
from diffusers.optimization import get_scheduler

from utils import Arguments
from train_methods.consts import imagenette_labels
from train_methods.data import Imagenette, NSFW, SalUnDataset
from train_methods.train_utils import prepare_extra_step_kwargs, sample_until, gather_parameters, encode_prompt, get_devices, get_models, get_condition

warnings.filterwarnings("ignore")

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
) -> torch.Tensor:

    unet_student.train()
    prompt_embeds = encode_prompt(
        prompt=prompt, 
        removing_prompt=removing_prompt,
        text_encoder=text_encoder, 
        tokenizer=tokenizer,
        device=text_encoder.device,
    )

    uncond_emb, cond_emb, safety_emb = torch.chunk(prompt_embeds, 3, dim=0)
    batch_size = cond_emb.shape[0]

    noise_scheduler.set_timesteps(args.ddpm_steps, unet_student.device)

    latent_shape = (batch_size, unet_teacher.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator)
    latents = latents * ddim_scheduler.init_noise_sigma # z_T

    t_ddim = torch.randint(0, args.ddim_steps, (1,))
    t_ddpm_start = round((1 - (int(t_ddim) + 1) / args.ddim_steps) * args.ddpm_steps)
    t_ddpm_end   = round((1 - int(t_ddim) / args.ddim_steps) * args.ddpm_steps)
    t_ddpm = torch.randint(t_ddpm_start, t_ddpm_end, (batch_size,))
    
    extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, args.salun_eta)

    with torch.no_grad():
        # args.guidance_scale: s_g in the paper
        prompt_embeds = torch.cat([uncond_emb, cond_emb], dim=0) if args.guidance_scale > 1.0 else uncond_emb
        prompt_embeds = prompt_embeds.to(unet_student.device)

        latents = sample_until(
            until=int(t_ddim),
            latents=latents.to(unet_student.device),
            unet=unet_student,
            scheduler=ddim_scheduler,
            prompt_embeds=prompt_embeds,
            guidance_scale=args.guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
        )

        # Stop-grad and send to the second device
        e_0: torch.Tensor = unet_teacher(latents.to(unet_teacher.device), t_ddpm.to(unet_teacher.device), encoder_hidden_states=uncond_emb.to(unet_teacher.device)).sample
        e_p: torch.Tensor = unet_teacher(latents.to(unet_teacher.device), t_ddpm.to(unet_teacher.device), encoder_hidden_states=safety_emb.to(unet_teacher.device)).sample

        e_0 = e_0.detach()
        e_p = e_p.detach()

        # args.concept_scale: s_s in the paper
        noise_target = e_0 - args.negative_guidance * (e_p - e_0)

    noise_pred = unet_student(latents.to(unet_student.device), t_ddpm.to(unet_student.device), encoder_hidden_states=safety_emb.to(unet_student.device)).sample

    return F.mse_loss(noise_pred, noise_target.to(unet_student.device))


def salun(args: Arguments, mask_path: str):
    # You may provide a single file path, or a list of concepts
    if len(args.concepts) == 1 and args.concepts[0].endswith(".txt"):
        with open(args.concepts[0], "r") as f:
            args.concepts = f.read().splitlines()

    devices = get_devices(args)
    unet_student: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    tokenizer, text_encoder, _, unet_teacher, ddim_scheduler, noise_scheduler = get_models(args)

    unet_teacher.requires_grad_(False)
    text_encoder.requires_grad_(False)

    _, parameters = gather_parameters(args.salun_method, unet_student)
    num_train_param = sum(p.numel() for p in parameters)
    num_total_param = sum(p.numel() for p in unet_student.parameters())
    print(f"Finetuning parameters: {num_train_param} / {num_total_param} ({num_train_param / num_total_param:.2%})")

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

def _convert_image_to_rgb(image: Image.Image):
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

def get_imagenette_label_from_concept(concept) -> int:
    d = {}
    for i in range(len(imagenette_labels)):
        d[imagenette_labels[i]] = i
    
    return d[concept]

def setup_forget_data(args: Arguments, device: torch.device):
    transform = get_transform(size=args.image_size)

    if args.concepts in imagenette_labels:
        train_set = Imagenette("train", transform=transform)
        class_to_forget = get_imagenette_label_from_concept(args.concepts)
        descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
        filtered_data = [data for data in train_set if data[1] == class_to_forget]
        train_dl = DataLoader(filtered_data, batch_size=args.salun_masking_batch_size)
        return train_dl, descriptions
    else:
        descriptions = f"an image of a {args.concepts}"
        print("generating images")
        pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(args.sd_version)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        num_images_per_prompt = 5
        pipe.to(device)
        Path("salun-data/train").mkdir(exist_ok=True)
        for i in trange(800 // num_images_per_prompt):
            generator = torch.Generator(device).manual_seed(args.seed)
            images = pipe(descriptions, guidance_scale=args.guidance_scale, num_images_per_prompt=num_images_per_prompt, generator=generator).images

            for j in range(num_images_per_prompt):
                images[j].save(f"salun-data/train/{args.concepts.replace(' ', '-')}-{i * num_images_per_prompt + j:03}.png")

        train_set = SalUnDataset("salun-data/train", transform)
        train_dl = DataLoader(train_set, batch_size=args.salun_masking_batch_size)
        return train_dl, descriptions

def generate_mask(args: Arguments):
    
    device = get_devices(args)[0]
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    train_dl, descriptions = setup_forget_data(args, device)

    text_encoder.eval()
    vae.eval()
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    criteria = nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=args.salun_masking_lr)

    gradients: dict[str, nn.Parameter] = {}
    for name, param in unet.named_parameters():
        gradients[name] = 0

    pbar = trange(len(train_dl))
    is_imagenette = args.concepts in imagenette_labels
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
            forget_emb = get_condition(prompts, tokenizer, text_encoder)
            null_emb = get_condition(null_prompts, tokenizer, text_encoder)

        t = torch.randint(0, scheduler.config.num_train_timesteps, (1, )).long()
        t = t.to(device)

        noise = torch.randn_like(forget_input, device=device)
        forget_noisy = scheduler.add_noise(forget_input, noise, t)
        forget_out = unet(forget_noisy, t, forget_emb).sample
        null_out = unet(forget_noisy, t, null_emb).sample

        preds = (1 + args.guidance_scale) * forget_out - args.guidance_scale * null_out
        loss: torch.Tensor = -criteria(noise, preds)
        loss.backward()

        with torch.no_grad():
            for name, param in unet.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data.cpu()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        mask_path = Path("mask", args.concepts)
        Path(mask_path).mkdir(exist_ok=True)

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
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements
        torch.save(hard_dict, res:=Path(mask_path, f"with_{str(threshold)}.pt"))

    return res

def generate_nsfw_mask(args: Arguments):
    device = get_devices(args)[0]
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")

    forget_set = NSFW(transform=get_transform(size=args.image_size))
    train_dl = DataLoader(forget_set, batch_size=args.salun_masking_batch_size)

    text_encoder.eval()
    vae.eval()

    criteria = nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=args.salun_masking_lr)
    word_nude = "a photo of a nude person"

    gradients: dict[str, torch.Tensor] = {}
    for name, param in unet.named_parameters():
        gradients[name] = 0

    pbar = trange(len(train_dl))
    for _ in pbar:
        images: torch.Tensor = next(iter(train_dl))
        optimizer.zero_grad()

        images = images.to(device)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()

        null_prompts = [""] * args.salun_masking_batch_size
        prompts = [word_nude] * args.salun_masking_batch_size
        
        with torch.no_grad():
            forget_input = vae.encode(images).latent_dist.sample()
            forget_emb = get_condition(prompts, tokenizer, text_encoder)
            null_emb = get_condition(null_prompts, tokenizer, text_encoder)

        noise = torch.randn_like(forget_input, device=device)
        forget_noisy = scheduler.add_noise(forget_input, noise, t)
        forget_out = unet(forget_noisy, t, forget_emb).sample
        null_out = unet(forget_noisy, t, null_emb).sample
        
        preds = (1 + args.guidance_scale) * forget_out - args.guidance_scale * null_out
        
        loss: torch.Tensor = - criteria(noise, preds)
        loss.backward()

        with torch.no_grad():
            for name, param in unet.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data.cpu()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

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
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, res:=Path(f"mask/nude_{threshold}.pt"))

    return res

def masking(args: Arguments) -> str:
    if args.is_nsfw:
        return generate_nsfw_mask(args)
    else:
        return generate_mask(args)
    

def main(args: Arguments):
    mask_path = masking(args)
    salun(args, mask_path)

    if Path("salun-data").is_dir():
        shutil.rmtree("salun-data")
