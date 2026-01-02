# https://github.com/Abhiramkns/EraseFlow
# EraseFlow (NeurIPS 2025)

import time
import contextlib
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

from train_methods.train_utils import get_models


def generate_negative_prompt_embeddings(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel) -> torch.Tensor:
    """
    Create a single-token negative prompt embedding (just an empty string)
    to use for classifier-free guidance.
    """
    with torch.no_grad():
        neg = text_encoder(
            tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).input_ids.to(text_encoder.device)
        )[0]
    return neg

def save_lora_checkpoint(unet: UNet2DConditionModel, output_dir, epoch):
    save_path = Path(output_dir) / Path(f"checkpoint_epoch{epoch}")
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

def setup_optimizer_and_scaler(unet: UNet2DConditionModel, args):
    """
    Create optimizer (8-bit AdamW if requested) over
    LoRA parameters + z_model parameter, plus optional GradScaler.
    """
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes (pip install bitsandbytes) for 8-bit Adam.")
    else:
        optimizer_cls = torch.optim.AdamW

    # z_model is a single scalar parameter to learn the flow constant
    z_model = torch.nn.Parameter(
        torch.tensor(-0.1953, device=unet.device, dtype=torch.float32, requires_grad=True)
    )

    # collect LoRA layers
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    param_groups = [
        {"params": lora_layers, "lr": args.learning_rate},
        {"params": z_model, "lr": args.flow_learning_rate},
    ]

    optimizer = optimizer_cls(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    scaler = None

    return optimizer, scaler, z_model



def train_eraseflow_step(
    sample,
    unet,
    pipeline,
    optimizer,
    scaler,
    z_model,
    args,
    train_neg_prompt_embeds
):
    """
    Given a single `sample` dict (with latents, next_latents, embeddings, timesteps):
      1. Optionally concatenate unconditional + conditional embeddings if args.cfg
      2. Loop over a subset of timesteps to compute log-forward & log-backward
      3. Compute flow-loss and z_loss, backpropagate, and step optimizer (with GradScaler if needed)
    Returns the scalar loss value (float).
    """
    # build “embeds” for UNet input
    if args.cfg:
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

        with args.autocast():
            if args.cfg:
                # forward twice: uncond + text
                noise_pred = unet(
                    torch.cat([j_latent] * 2),
                    torch.cat([sample["timesteps"][:, j]] * 2),
                    embeds,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = unet(
                    j_latent,
                    sample["timesteps"][:, j],
                    embeds,
                ).sample

            # compute log‐forward (log_pf) and log‐backward (log_pb) under the DDIM step
            _, log_pf, log_pb = ddim_step_with_logprob(
                pipeline.scheduler,
                noise_pred,
                sample["timesteps"][:, j],
                j_latent,
                eta=args.eta,
                prev_sample=next_j_latent,
                calculate_pb=True,
            )

        # GFlowNet loss = (log_pf - log_pb)
        loss_flow = log_pf - log_pb  # shape (batch_size,)
        total_loss = total_loss + loss_flow

        # free up intermediate tensors
        torch.cuda.empty_cache()

    # compute z‐loss: encourages z_model ≈ log(beta). Here z_model models the log(Z).
    z_target = args.logbeta
    z_loss = z_model - z_target
    total_loss = total_loss + z_loss

    # mean‐squared: (sum over batch & timesteps + z_loss).pow(2).mean()
    total_loss = torch.mean(total_loss.pow(2))

    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(z_model, args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(z_model, args.max_grad_norm)
        optimizer.step()

    optimizer.zero_grad()

    return total_loss.item()


def sample_epoch(
    pipeline,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    scheduler: DDIMScheduler,
    args,
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
    prompts = [args.anchor_prompt] * args.batch_size
    prompt_ids = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids

    with torch.no_grad():
        anchor_prompt_embeds = text_encoder(prompt_ids.to(text_encoder.device))[0]

    # Expand negative‐prompt embeddings to match batch size
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(len(anchor_prompt_embeds), 1, 1)

    # Inform scheduler how many diffusion steps we take this epoch
    scheduler.set_timesteps(args.num_steps, device=text_encoder.device)

    # Perform one pass of inference‐mode sampling (no gradients)
    with torch.inference_mode():
        ret_tuple = pipeline_with_logprob(
            pipeline,
            prompt_embeds=anchor_prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            output_type="pt",
            return_unetoutput=False,
            num_images_per_prompt=1,
        )
        _, _, latents, _ = ret_tuple

    # latents has shape (batch_size, num_steps+1, 4, 64, 64)
    latents = torch.stack(latents, dim=1)

    # Now encode train (target) prompts
    train_prompts = [args.target_prompt] * args.batch_size
    train_prompt_ids = tokenizer(
        train_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids
    with torch.no_grad():
        train_prompt_embeds = text_encoder(train_prompt_ids.to(text_encoder.device))[0]

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

def train(args):

    device = args.device

    tokenizer, text_encoder, _, unet, ddim_scheduler, _ = get_models(args)
    text_encoder.eval()
    text_encoder.to(device)

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
    optimizer, scaler, z_model = setup_optimizer_and_scaler(unet, args)

    # 4) Decide which autocast to use
    args.autocast = contextlib.nullcontext

    # 5) Precompute negative prompt embedding (for classifier‐free guidance)
    neg_prompt = generate_negative_prompt_embeddings(tokenizer, text_encoder)
    train_neg_prompt_embeds = neg_prompt.repeat(1, 1, 1)

    output_dir = Path(args.save_dir) / Path(args.name)
    output_dir.mkdir(exist_ok=True)

    global_step = 0
    last_samples = None

    # ──────────── Start timing ────────────
    start_time = time.time()
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch}/{args.num_epochs - 1}")

        # Determine whether to sample fresh latents or reuse previous
        if args.switch_epoch is None or epoch <= args.switch_epoch:
            # SAMPLE PHASE (no gradients)
            unet.zero_grad()
            unet.eval()
            last_samples = sample_epoch(pipeline, args, device, neg_prompt)
        else:
            # Use samples from the last sampling epoch
            print(f"Epoch {epoch} > switch_epoch ({args.switch_epoch}): reusing previous samples")
            # last_samples remains unchanged

        # TRAIN PHASE
        if last_samples is not None:
            unet.train()
            for sample in last_samples:
                loss_val = train_eraseflow_step(
                    sample,
                    unet,
                    pipeline,
                    optimizer,
                    scaler,
                    z_model,
                    args,
                    train_neg_prompt_embeds,
                )
                global_step += 1
                if global_step % 10 == 0:
                    print(f"Epoch {epoch} | step {global_step} | loss {loss_val:.4f}")

        # SAVE CHECKPOINT
        if epoch % args.save_freq == 0 or epoch == args.num_epochs - 1:
            save_lora_checkpoint(unet, output_dir, epoch)

    # ────────── End timing ──────────
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    seconds = elapsed
    print("Training complete.")
    print(f"Total training time: {hours:.2f} hours ({seconds:.1f} seconds).")

def main():
    args = get_args()
    train(args)
