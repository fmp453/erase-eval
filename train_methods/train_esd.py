# Erasing Concepts from Diffusion Models (ESD)

# ref: https://github.com/nannullna/safe-diffusion/blob/main/train_sdd.py

import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import trange
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler

from utils import Arguments
from train_methods.train_utils import prepare_extra_step_kwargs, sample_until, gather_parameters, encode_prompt, get_devices, get_models


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
    extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, args.esd_eta)

    with torch.no_grad():
        # args.start_guidance: s_g in the paper
        prompt_embeds = torch.cat([uncond_emb, cond_emb], dim=0) if args.start_guidance > 1.0 else uncond_emb
        prompt_embeds = prompt_embeds.to(unet_student.device)

        # Generate latents
        latents = sample_until(
            until=int(t_ddim),
            latents=latents,
            unet=unet_student,
            scheduler=ddim_scheduler,
            prompt_embeds=prompt_embeds,
            guidance_scale=args.start_guidance,
            extra_step_kwargs=extra_step_kwargs,
        )

        # Stop-grad and send to the second device
        _latents = latents.to(devices[1])
        e_0: torch.Tensor = unet_teacher(_latents, t_ddpm.to(devices[1]), encoder_hidden_states=uncond_emb).sample
        e_p: torch.Tensor = unet_teacher(_latents, t_ddpm.to(devices[1]), encoder_hidden_states=safety_emb).sample

        e_0 = e_0.detach().to(devices[0])
        e_p = e_p.detach().to(devices[0])

        # args.negative_guidance: s_s in the paper
        noise_target = e_0 - args.negative_guidance * (e_p - e_0)

    noise_pred: torch.Tensor = unet_student(latents, t_ddpm.to(devices[0]), encoder_hidden_states=safety_emb.to(devices[0])).sample
    return F.mse_loss(noise_pred, noise_target.to(noise_pred.device))


def main(args: Arguments):
    
    # You may provide a single file path, or a list of concepts
    if len(args.concepts) == 1 and args.concepts[0].endswith(".txt"):
        with open(args.concepts[0], "r") as f:
            args.concepts = f.read().splitlines()

    # This script requires two CUDA devices
    # Sample latents on the first device, and train the unet on the second device
    devices = get_devices(args)

    unet_student: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    tokenizer, text_encoder, _, unet_teacher, ddim_scheduler, noise_scheduler = get_models(args)

    unet_teacher.requires_grad_(False)
    text_encoder.requires_grad_(False)

    names, parameters = gather_parameters(args.esd_method, unet_student)
    print(f"Finetuning parameters: {names}")
    num_train_param = sum(p.numel() for p in parameters)
    num_total_param = sum(p.numel() for p in unet_student.parameters())
    print(f"Finetuning parameters: {num_train_param} / {num_total_param} ({num_train_param / num_total_param:.2%})")

    # Create optimizer and scheduler
    optimizer = optim.Adam(parameters, lr=args.esd_lr)
    lr_scheduler: LambdaLR = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.esd_lr_warmup_steps,
        num_training_steps=args.esd_iter,
    )

    # First device -- unet_student, generator
    # Second device -- unet_teacher, text_encoder
    unet_student = unet_student.to(devices[0])
    gen = torch.Generator(device=devices[0])

    unet_teacher = unet_teacher.to(devices[1])
    text_encoder = text_encoder.to(devices[1])

    # Set the number of inference time steps
    ddim_scheduler.set_timesteps(args.ddim_steps, devices[1])
    progress_bar = trange(1, args.esd_iter+1, desc="Training")

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
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_description(f"Training: {train_loss.item():.4f}")
        
    unet_student.save_pretrained(args.save_dir)
