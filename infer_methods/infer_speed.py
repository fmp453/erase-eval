from copy import deepcopy
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline

from utils import Arguments


def infer(args: Arguments):
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_version)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe._progress_bar_config = {"disable": True}
    unet: UNet2DConditionModel = pipe.unet

    unet_edit = deepcopy(unet)
    edit_path = args.erased_model_file
    unet_edit.load_state_dict(torch.load(edit_path, map_location='cpu'), strict=False)

    pipe.unet = unet_edit

    device = f"cuda:{args.device.split(',')[0]}"
    pipe = pipe.to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    images = pipe(args.prompt, guidance_scale=args.guidance_scale, num_images_per_prompt=args.num_images_per_prompt, generator=generator).images

    Path(args.images_dir).mkdir(exist_ok=True)
    for i in range(len(images)):
        images[i].save(f"{args.images_dir}/{i:02}.png")

def main(args: Arguments):
    infer(args)
