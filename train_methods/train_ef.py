# https://github.com/Abhiramkns/EraseFlow
# EraseFlow (NeurIPS 2025)

import time
import contextlib
from pathlib import Path

import torch
import torch.optim as optim
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, get_peft_model_state_dict
from torch.nn.utils import clip_grad_norm_
from transformers import CLIPTextModel, CLIPTokenizer

from train_methods.train_utils import get_models, get_condition, get_devices
from train_methods.utils_ef import ddim_step_with_logprob, pipeline_with_logprob
from utils import Arguments

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if is_compiled_module(model) else model

def save_lora_checkpoint(unet: UNet2DConditionModel, output_dir: Path, epoch: int):
    save_path = Path(output_dir, f"checkpoint_epoch{epoch}")
    save_path.mkdir(exist_ok=True)

    unwrapped = unwrap_model(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unwrapped)
    )

    StableDiffusionPipeline.save_lora_weights(
        save_directory=save_path,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )

def setup_optimizer_and_scaler(unet: UNet2DConditionModel, args: Arguments):
    """
    Create optimizer (8-bit AdamW if requested) over
    LoRA parameters + z_model parameter, plus optional GradScaler.
    """
    if args.ef_use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes (pip install bitsandbytes) for 8-bit Adam.")
    else:
        optimizer_cls = optim.AdamW

    # z_model is a single scalar parameter to learn the flow constant
    z_model = torch.nn.Parameter(
        torch.tensor(-0.1953, device=unet.device, dtype=torch.float32, requires_grad=True)
    )

    # collect LoRA layers
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    param_groups = [
        {"params": lora_layers, "lr": args.ef_lr},
        {"params": z_model, "lr": args.ef_flow_lr},
    ]

    optimizer = optimizer_cls(
        param_groups,
        betas=(args.ef_adam_beta1, args.ef_adam_beta2),
        weight_decay=args.ef_adam_weight_decay,
        eps=args.ef_adam_epsilon,
    )

    return optimizer, z_model


def train_eraseflow_step(
    sample: dict[str, torch.Tensor],
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    optimizer: optim.Optimizer,
    z_model,
    args: Arguments,
    train_neg_prompt_embeds: torch.Tensor,
):
    """
    Given a single `sample` dict (with latents, next_latents, embeddings, timesteps):
      1. Optionally concatenate unconditional + conditional embeddings if args.cfg
      2. Loop over a subset of timesteps to compute log-forward & log-backward
      3. Compute flow-loss and z_loss, backpropagate, and step optimizer (with GradScaler if needed)
    Returns the scalar loss value (float).
    """
    # build “embeds” for UNet input
    if args.start_guidance > 1.0:
        # classifier-free: concat neg + pos prompt embeddings
        embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
    else:
        embeds = sample["prompt_embeds"]

    total_loss = 0.0

    # choose timesteps indices (same as original: 10 random in [0..39], plus [40..49])
    indices = torch.cat([torch.randperm(40)[:10], torch.arange(40, 50)]).to(sample["latents"].device)

    for j in indices:
        j = int(j.item())
        j_latent = sample["latents"][:, j]
        next_j_latent = sample["next_latents"][:, j]

        with contextlib.nullcontext():
            if args.start_guidance > 1.0:
                noise_pred: torch.Tensor = unet(
                    torch.cat([j_latent] * 2),
                    torch.cat([sample["timesteps"][:, j]] * 2),
                    embeds,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.start_guidance * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = unet(
                    j_latent,
                    sample["timesteps"][:, j],
                    embeds,
                ).sample

            # compute log‐forward (log_pf) and log‐backward (log_pb) under the DDIM step
            _, log_pf, log_pb = ddim_step_with_logprob(
                scheduler,
                noise_pred,
                sample["timesteps"][:, j],
                j_latent,
                eta=args.ef_eta,
                prev_sample=next_j_latent,
            )

        # GFlowNet loss = (log_pf - log_pb)
        loss_flow = log_pf - log_pb  # shape (batch_size,)
        total_loss = total_loss + loss_flow

        # free up intermediate tensors
        torch.cuda.empty_cache()

    # compute z‐loss: encourages z_model ≈ log(beta). Here z_model models the log(Z).
    z_target = args.ef_logbeta or 2.5
    z_loss = z_model - z_target
    total_loss: torch.Tensor = total_loss + z_loss

    # mean‐squared: (sum over batch & timesteps + z_loss).pow(2).mean()
    total_loss = torch.mean(total_loss.pow(2))

    total_loss.backward()
    clip_grad_norm_(unet.parameters(), args.max_grad_norm)
    clip_grad_norm_(z_model, args.max_grad_norm)
    optimizer.step()

    optimizer.zero_grad()

    return total_loss.item()


def sample_epoch(
    args: Arguments,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    scheduler: DDIMScheduler,
    neg_prompt_embed: torch.Tensor
) -> list[dict[str, torch.Tensor]]:
    """
    For each epoch: 
      - encode anchor_prompt
      - run DDIM sampling w/ log-prob until latent sequence
      - stack latents, prepare train prompt embeddings + timesteps
      - return a list of dicts containing latents, next_latents, embeddings, etc.
    """
    samples = []

    # Anchor prompts → their embeddings
    prompts = [args.anchor_concept] * args.ef_batch_size
    anchor_prompt_embeds = get_condition(prompts, tokenizer, text_encoder)

    # Expand negative‐prompt embeddings to match batch size
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(len(anchor_prompt_embeds), 1, 1)

    # Inform scheduler how many diffusion steps we take this epoch
    scheduler.set_timesteps(args.ddim_steps, device=text_encoder.device)

    # Perform one pass of inference‐mode sampling (no gradients)
    with torch.inference_mode():
        ret_tuple = pipeline_with_logprob(
            tokenizer,
            text_encoder,
            unet,
            vae,
            num_inference_steps=args.ddim_steps,
            guidance_scale=args.start_guidance,
            num_images_per_prompt=args.ef_batch_size,
            eta=args.ef_eta,
            prompt_embeds=anchor_prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            output_type="pt",
            return_unetoutput=False,
        )
        _, latents, _ = ret_tuple

    # latents has shape (batch_size, num_steps+1, 4, 64, 64)
    latents = torch.stack(latents, dim=1)

    # encode train (target) prompts
    train_prompts = [args.concepts] * args.ef_batch_size
    train_prompt_embeds = get_condition(train_prompts, tokenizer, text_encoder)

    # Build a timesteps tensor: shape (batch_size, num_steps)
    timesteps = scheduler.timesteps.repeat(len(train_prompt_embeds), 1)

    samples.append({
        "prompt_embeds": train_prompt_embeds,
        "anchor_prompt_embeds": anchor_prompt_embeds,
        "timesteps": timesteps,
        # we drop the last timestep for “latents” and shift 1 for next_latents
        "latents": latents[:, :-1],
        "next_latents": latents[:, 1:],
    })

    return samples

def train(args: Arguments):

    device = get_devices(args)[0]
    tokenizer, text_encoder, vae, unet, scheduler, _ = get_models(args)
    text_encoder.eval()
    text_encoder.to(device)
    vae.eval()
    vae.to(device)

    # 2) Attach LoRA to UNet and freeze base weights
    unet.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)
    unet.to(device)

    unet_lora_config = LoraConfig(
        r=args.ef_lora_rank,
        lora_alpha=args.ef_lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # 3) Build optimizer (LoRA + z_model) and potential GradScaler
    optimizer, z_model = setup_optimizer_and_scaler(unet, args)

    # 5) Precompute negative prompt embedding (for classifier‐free guidance)
    neg_prompt = get_condition("", tokenizer, text_encoder)
    train_neg_prompt_embeds = neg_prompt.repeat(1, 1, 1)

    output_dir = Path(args.save_dir)
    output_dir.mkdir(exist_ok=True)

    global_step = 0
    last_samples = None

    start_time = time.time()
    for epoch in range(args.ef_num_epochs):
        print(f"Starting epoch {epoch}/{args.ef_num_epochs - 1}")

        # Determine whether to sample fresh latents or reuse previous
        if args.ef_switch_epoch is None or epoch <= args.ef_switch_epoch:
            # SAMPLE PHASE (no gradients)
            unet.zero_grad()
            unet.eval()
            last_samples = sample_epoch(
                args,
                tokenizer,
                text_encoder,
                unet,
                vae,
                scheduler,
                neg_prompt
            )
        else:
            # Use samples from the last sampling epoch
            print(f"Epoch {epoch} > switch_epoch ({args.ef_switch_epoch}): reusing previous samples")
            # last_samples remains unchanged

        # TRAIN PHASE
        if last_samples is not None:
            unet.train()
            for sample in last_samples:
                loss_val = train_eraseflow_step(
                    sample,
                    unet,
                    scheduler,
                    optimizer,
                    z_model,
                    args,
                    train_neg_prompt_embeds,
                )
                global_step += 1
                if global_step % 10 == 0:
                    print(f"Epoch {epoch} | step {global_step} | loss {loss_val:.4f}")

    save_lora_checkpoint(unet, output_dir, epoch)
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    seconds = elapsed
    print("Training complete.")
    print(f"Total training time: {hours:.2f} hours ({seconds:.1f} seconds).")

def main(args: Arguments):
    train(args)
