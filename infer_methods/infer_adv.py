import os

import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel

from utils import Arguments

def infer(args: Arguments):
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_version)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe._progress_bar_config = {"disable": True}
    pipe.text_encoder = CLIPTextModel.from_pretrained(args.erased_model_dir)

    device = f"cuda:{args.device.split(',')[0]}"
    pipe = pipe.to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    images = pipe(args.prompt, guidance_scale=args.guidance_scale, num_images_per_prompt=args.num_images_per_prompt, generator=generator).images

    os.makedirs(args.images_dir, exist_ok=True)
    for i in range(len(images)):
        images[i].save(f"{args.images_dir}/{i:02}.png")

def main(args: Arguments):
    infer(args)

