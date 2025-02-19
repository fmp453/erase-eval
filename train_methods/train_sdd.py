import random
from typing import Union

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from utils import Arguments
from train_methods.train_utils import prepare_extra_step_kwargs, sample_until

# sdd (safe self-distill diffusion)
# ref: https://github.com/nannullna/safe-diffusion


def gather_parameters(args: Arguments, unet: UNet2DConditionModel) -> tuple[list[str], list[torch.nn.Parameter]]:
    """Gather the parameters to be optimized by the optimizer."""
    names, parameters = [], []
    for name, param in unet.named_parameters():
        if args.sdd_method == "full":
            # Train all layers.
            names.append(name)
            parameters.append(param)
        elif args.sdd_method == "selfattn":
            # Attention layer 1 is the self-attention layer.
            if "attn1" in name:
                names.append(name)
                parameters.append(param)
        elif args.sdd_method == "xattn":
            # Attention layer 2 is the cross-attention layer.
            if "attn2" in name:
                names.append(name)
                parameters.append(param)
        elif args.sdd_method == "noxattn":
            # Train all layers except the cross attention and time_embedding layers.
            if name.startswith("conv_out.") or ("time_embed" in name):
                # Skip the time_embedding layer.
                continue
            elif "attn2" in name:
                # Skip the cross attention layer.
                continue
            names.append(name)
            parameters.append(param)
        elif args.sdd_method == "notime":
            # Train all layers except the time_embedding layer.
            if name.startswith("conv_out.") or ("time_embed" in name):
                continue
            names.append(name)
            parameters.append(param)
        else:
            raise ValueError(f"Unknown finetuning method: {args.sdd_method}")

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
    uncond_input = tokenizer(
        [""] * batch_size if negative_prompt is None else negative_prompt,
        padding="max_length", 
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    if prompt is not None:
        prompt_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt",
        )
    else:
        prompt_input = None
    
    if removing_prompt is not None:
        removing_input = tokenizer(
            removing_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt",
        )
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
    noise_scheduler.set_timesteps(1000, devices[1])

    # Prepare latent codes to generate z_t
    latent_shape = (batch_size, unet_teacher.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator, device=devices[1])
    # Scale the initial noise by the standard deviation required by the scheduler
    latents = latents * ddim_scheduler.init_noise_sigma # z_T

    # Normally, DDPM takes 1,000 timesteps for training, and DDIM takes 50 timesteps for inference.
    t_ddim = torch.randint(0, 50, (1,))
    t_ddpm_start = round((1 - (int(t_ddim) + 1) / 50) * 1000)
    t_ddpm_end   = round((1 - int(t_ddim)       / 50) * 1000)
    t_ddpm = torch.randint(t_ddpm_start, t_ddpm_end, (batch_size,),)
    
    # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, 0.0)

    with torch.no_grad():
        latents = sample_until(
            until=int(t_ddim),
            latents=latents,
            unet=unet_teacher,
            scheduler=ddim_scheduler,
            prompt_embeds=torch.cat([uncond_emb, cond_emb], dim=0) if args.start_guidance > 1.0 else uncond_emb,
            guidance_scale=args.start_guidance,
            extra_step_kwargs=extra_step_kwargs,
        )

    latents = latents.to(unet_student.device)
    t_ddpm = t_ddpm.to(unet_student.device)
    c_0 = uncond_emb.to(unet_student.device)
    c_s = safety_emb.to(unet_student.device)

    with torch.no_grad():
        e_0 = unet_student(latents, t_ddpm, encoder_hidden_states=c_0).sample
    e_s = unet_student(latents, t_ddpm, encoder_hidden_states=c_s).sample

    loss = F.mse_loss(e_0.detach(), e_s)
    return loss

def main(args: Arguments):    
    
    devices = args.device.split(",")
    if len(devices) > 1:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[1]}")]
    else:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[0]}")]

    # Load pretrained model
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    unet_teacher: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    unet_student: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    ddim_scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")

    # Freeze vae and text_encoder
    unet_teacher.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    names, parameters = gather_parameters(args, unet_student)
    print(f"Finetuning parameters: {names}")
    num_train_param = sum(p.numel() for p in parameters)
    num_total_param = sum(p.numel() for p in unet_student.parameters())
    print(f"Finetuning parameters: {num_train_param} / {num_total_param} ({num_train_param / num_total_param:.2%})")

    # Create optimizer and scheduler
    # hyperparameters are from official paper. 1st paragraph in appendix B
    optimizer = optim.AdamW(
        parameters,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )
    lr_scheduler: LambdaLR = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.sdd_num_steps
    )

    # First device -- unet_student
    # Second device -- unet_teacher, vae, text_encoder
    unet_student = unet_student.to(devices[0])

    unet_teacher = unet_teacher.to(devices[1])
    text_encoder = text_encoder.to(devices[1])
    vae = vae.to(devices[1])
    gen = torch.Generator(device=devices[1])
    
    # Set the number of inference time steps
    ddim_scheduler.set_timesteps(50, devices[1])
    ema_decay = 0.998
    max_grad_norm = 1.0

    step = 0
    progress_bar = tqdm(range(1, args.sdd_num_steps + 1), desc="Training")

    for step in progress_bar:

        # Sample a concept to remove
        if args.sdd_concept_method == "composite":
            # concat all strings separated by commas in removing_concepts
            removing_concept = ", ".join(args.concepts)
        elif args.sdd_concept_method == "random":
            # randomly choose a concept to remove
            removing_concept = random.choice(args.concepts)
        elif args.sdd_concept_method == "iterative":
            # iteratively choose a concept to remove
            removing_concept = args.concepts[(step-1) % len(args.concepts)]
        elif args.sdd_concept_method == "sequential":
            # choose a concept to remove in a continual manner
            removing_concept = args.concepts[(step-1) // args.sdd_num_steps]

        removing_prompt = removing_concept
        prompt = ", ".join(args.concepts)

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
            devices=devices
        )

        train_loss.backward()
        
        if max_grad_norm > 0:
            clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Update unet_teacher with EMA
        with torch.no_grad():
            for param, ema_param in zip(unet_student.parameters(), unet_teacher.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data.to(devices[1]), alpha=1 - ema_decay)

        progress_bar.set_description(f"Training: {train_loss.item():.4f} on c_p: {prompt} - c_s: {removing_concept}")
        
        if step % 100 == 0:
            print(f"Step: {step} | Loss: {train_loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.4e}")

    # Save final checkpoint
    unet_teacher.save_pretrained(args.save_dir)
    
