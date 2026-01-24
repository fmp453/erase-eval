# Ablating Concepts in Text-to-Image Diffusion Models (AC)
# model-based concept ablation
# ref: https://huggingface.co/spaces/nupurkmr9/concept-ablation/blob/main/concept-ablation-diffusers/train.py

import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict

from tqdm import trange
from pathlib import Path
from torch.utils.data import DataLoader

from diffusers import StableDiffusionPipeline

from train_methods.data import AblatingConceptDataset
from train_methods.train_utils import collate_fn, get_devices, get_models
from utils import Arguments


def train(args: Arguments):
    
    tokenizer, text_encoder, vae, unet, scheduler, _ = get_models(args)
    
    device = get_devices(args)[0]

    text_encoder.eval()
    vae.eval()
    unet.to(device)
    vae.to(device)
    text_encoder.to(device)
    
    for name, params in unet.named_parameters():
        if args.ac_method == "xattn":
            if "attn2" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif args.ac_method == "full":
            params.requires_grad = True
        else:
            raise ValueError(f"Unknown finetuning method: {args.ac_method}")

    cnt = 0
    tot = 0
    for param in unet.parameters():
        tot += param.numel()
        if param.requires_grad:
            cnt += param.numel()
    
    print(f"{cnt / tot * 100}% parameters are updated.")

    optimizer = optim.Adam(unet.parameters(), lr=args.ac_lr, weight_decay=1e-2)
    dataset = AblatingConceptDataset(
        concept_type=args.ac_concept_type,
        image_dir=args.ac_img_dir,
        prompt_path=args.ac_prompt_path,
        tokenizer=tokenizer,
        concept=args.concepts,
        anchor_concept=args.anchor_concept,
        aug=(args.ac_concept_type != "style")
    )

    dataloader = DataLoader(dataset, batch_size=args.ac_batch_size, num_workers=2, shuffle=True, collate_fn=lambda examples: collate_fn(examples))

    print(f"{len(dataloader)=}")

    pbar = trange(0, 1 * len(dataloader), desc="step")
    unet.train()

    for _ in pbar:
        optimizer.zero_grad()
        batch = next(iter(dataloader))

        with torch.no_grad():
            latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
            text_embedding = text_encoder(batch["input_ids"].to(device))[0]
            anchor_embedding = text_encoder(batch["input_anchor_ids"].to(device))[0]
            latents = latents * vae.config.scaling_factor
        
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noisy_latens = scheduler.add_noise(latents, noise, timesteps)
        noise_pred = unet(noisy_latens, timesteps, text_embedding).sample

        with torch.no_grad():
            anchor_pred = unet(noisy_latens[:anchor_embedding.size(0)], timesteps[:anchor_embedding.size(0)], anchor_embedding).sample
        
        mask: torch.Tensor = batch["mask"].to(device)

        loss: torch.Tensor = F.mse_loss(noise_pred, anchor_pred, reduction="none")
        loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
    
        loss.backward()
        optimizer.step()
        pbar.set_postfix(OrderedDict(loss=loss.detach().item()))
    
    unet.save_pretrained(args.save_dir)

def generation(args: Arguments):
    print("generate images for Ablating Concepts")

    device = args.device.split(",")[0]
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_version).to(device=f"cuda:{device}")
    pipe.safety_checker = None
    df = pd.read_csv(args.ac_prompt_path)
    prompts = df["prompt"].tolist()
    Path(args.ac_img_dir).mkdir(exist_ok=True)
    
    for i in range(200):
        images = pipe(prompts[i], num_images_per_prompt=5).images
        for j in range(5):
            images[j].save(f"{args.ac_img_dir}/{i:03}-{j}.png")

    del pipe
    torch.cuda.empty_cache()

def main(args: Arguments):
    # generate images for Ablating Concepts
    generation(args)
    # main part of Ablating Concepts
    train(args)
