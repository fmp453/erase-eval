import os
import warnings

import torch

from diffusers import EulerDiscreteScheduler

from train_methods.mce_models import (
    SD2PipelineForCheckpointing,
    SD3PipelineForCheckpointing,
    SDXLPipelineForCheckpointing,
    DiTPipelineForCheckpointing,
    FluxPipelineForCheckpointing,
    ReverseDPMSolverMultistepScheduler,
    Pipeline
)
from utils import Arguments

warnings.filterwarnings("ignore")



def load_pipeline(model_str: str) -> Pipeline:
    if model_str == "sd1":
        pipe = SD2PipelineForCheckpointing.from_pretrained("CompVis/stable-diffusion-v1-4", include_entities=False)
    elif model_str == "sd2":
        # Use the Euler scheduler here instead
        scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="scheduler")
        pipe = SD2PipelineForCheckpointing.from_pretrained("stabilityai/stable-diffusion-2-base", scheduler=scheduler)
    elif model_str == "sdxl":
        pipe = SDXLPipelineForCheckpointing.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
    elif model_str == "sdxl_turbo":
        pipe = SDXLPipelineForCheckpointing.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True)
        pipe.set_distilled()
    elif model_str == "sd3":
        pipe = SD3PipelineForCheckpointing.from_pretrained("stabilityai/stable-diffusion-3.5-large")
    elif model_str == "dit":
        pipe = DiTPipelineForCheckpointing.from_pretrained("facebook/DiT-XL-2-256")
        pipe.scheduler = ReverseDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_str == "flux":
        pipe = FluxPipelineForCheckpointing.from_pretrained("black-forest-labs/FLUX.1-schnell")
    elif model_str == "flux_dev":
        pipe = FluxPipelineForCheckpointing.from_pretrained("black-forest-labs/FLUX.1-dev")
    else:
        raise ValueError(f"Model {model_str} not supported")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def infer(args: Arguments):
    pipe = load_pipeline(args.sd_version)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe._progress_bar_config = {"disable": True}

    device = f"cuda:{args.device.split(',')[0]}"
    pipe = pipe.to(device)

    generator = torch.Generator(device).manual_seed(args.seed)

    images = pipe(args.prompt, guidance_scale=args.guidance_scale, num_images_per_prompt=args.num_images_per_prompt, generator=generator).images

    os.makedirs(args.images_dir, exist_ok=True)
    for i in range(len(images)):
        images[i].save(f"{args.images_dir}/{i:02}.png")

def main(args: Arguments):
    infer(args)
