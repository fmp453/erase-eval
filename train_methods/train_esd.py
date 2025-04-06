# Erasing Concepts from Diffusion Models (ESD)

# ref: https://github.com/nannullna/safe-diffusion/blob/main/train_sdd.py

import inspect
import random
import warnings
from typing import Union, Any, Optional

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

warnings.filterwarnings("ignore")

def gather_parameters(args: Arguments, unet: UNet2DConditionModel) -> tuple[list[str], list[torch.nn.Parameter]]:
    """Gather the parameters to be optimized by the optimizer."""
    names, parameters = [], []
    for name, param in unet.named_parameters():
        if args.esd_method == "full":
            # Train all layers.
            names.append(name)
            parameters.append(param)
        elif args.esd_method == "selfattn":
            # Attention layer 1 is the self-attention layer.
            if "attn1" in name:
                names.append(name)
                parameters.append(param)
        elif args.esd_method == "xattn":
            # Attention layer 2 is the cross-attention layer.
            if "attn2" in name:
                names.append(name)
                parameters.append(param)
        elif args.esd_method == "noxattn":
            # Train all layers except the cross attention and time_embedding layers.
            if name.startswith("conv_out.") or ("time_embed" in name):
                # Skip the time_embedding layer.
                continue
            elif "attn2" in name:
                # Skip the cross attention layer.
                continue
            names.append(name)
            parameters.append(param)
        elif args.esd_method == "notime":
            # Train all layers except the time_embedding layer.
            if name.startswith("conv_out.") or ("time_embed" in name):
                continue
            names.append(name)
            parameters.append(param)
        else:
            raise ValueError(f"Unknown finetuning method: {args.esd_method}")

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

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    
    return extra_step_kwargs

# Sample latents from unet and DDIM scheduler until the given timestep.
@torch.no_grad()
def sample_until(
    until: int,
    latents: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
    extra_step_kwargs: Optional[dict[str, Any]]=None,
):
    """Sample latents until t for a given prompt."""
    timesteps = scheduler.timesteps

    do_guidance = abs(guidance_scale) > 1.0

    # Denoising loop
    for i, t in enumerate(timesteps):
        latent_model_input = (torch.cat([latents] * 2) if do_guidance else latents)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

        # perform guidance
        if do_guidance:
            noise_pred_out = torch.chunk(noise_pred, 2, dim=0)
            noise_pred_uncond, noise_pred_prompt = noise_pred_out[0], noise_pred_out[1]
            
            cond_guidance = noise_pred_prompt - noise_pred_uncond
            noise_pred = noise_pred_uncond + (guidance_scale * cond_guidance)

        latents = scheduler.step(model_output=noise_pred, timestep=t, sample=latents, **extra_step_kwargs).prev_sample

        if i == until - 1:
            break

    return latents

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
    # print(f"t_ddim: {t_ddim}, t_ddpm: {t_ddpm}")

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
        e_0 = unet_teacher(_latents, t_ddpm.to(devices[1]), encoder_hidden_states=uncond_emb).sample
        e_p = unet_teacher(_latents, t_ddpm.to(devices[1]), encoder_hidden_states=safety_emb).sample

        e_0 = e_0.detach().to(devices[0])
        e_p = e_p.detach().to(devices[0])

        # args.negative_guidance: s_s in the paper
        noise_target = e_0 - args.negative_guidance * (e_p - e_0)

    noise_pred = unet_student(latents, t_ddpm.to(devices[0]), encoder_hidden_states=safety_emb.to(devices[0])).sample

    loss = F.mse_loss(noise_pred, noise_target)
    
    return loss

def main(args: Arguments):
    
    # You may provide a single file path, or a list of concepts
    if len(args.concepts) == 1 and args.concepts[0].endswith(".txt"):
        with open(args.concepts[0], "r") as f:
            args.concepts = f.read().splitlines()

    # This script requires two CUDA devices
    # Sample latents on the first device, and train the unet on the second device
    devices = args.device.split(",")
    if len(devices) > 1:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[1]}")]
    else:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[0]}")]

    # Load pretrained models
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    ddim_scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")

    unet_teacher: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    unet_student: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")

    # Freeze vae and text_encoder
    unet_teacher.requires_grad_(False)
    text_encoder.requires_grad_(False)

    names, parameters = gather_parameters(args, unet_student)
    print(f"Finetuning parameters: {names}")
    num_train_param = sum(p.numel() for p in parameters)
    num_total_param = sum(p.numel() for p in unet_student.parameters())
    print(f"Finetuning parameters: {num_train_param} / {num_total_param} ({num_train_param / num_total_param:.2%})")

    # Create optimizer and scheduler
    # use default values except lr
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

