from pathlib import Path
from typing import Literal

import safetensors
import torch
from diffusers import LMSDiscreteScheduler, AutoencoderKL
from PIL import Image
from safetensors.torch import load_file
from tqdm.auto import tqdm

from train_methods.train_ace import ACELayer, ACENetwork
from train_methods.train_utils import get_models, get_devices, tokenize, get_condition
from utils import Arguments


def calculate_matching_score(
    prompt_tokens: list[torch.Tensor],
    prompt_embeds: torch.Tensor,
    erased_prompt_tokens: list[torch.Tensor],
    erased_prompt_embeds: torch.Tensor,
    matching_metric: Literal["clipcos", "tokenuni"],
    special_token_ids: set,
):
    scores = []
    if "clipcos" in matching_metric:
        clipcos = torch.cosine_similarity(
            prompt_embeds.flatten(1, 2),
            erased_prompt_embeds.flatten(1, 2),
            dim=-1).cpu()
        scores.append(clipcos)
    if "tokenuni" in matching_metric:
        prompt_set = set(prompt_tokens[0].tolist()) - special_token_ids
        tokenuni = []
        for ep in erased_prompt_tokens:
            ep_set = set(ep.tolist()) - special_token_ids
            tokenuni.append(len(prompt_set.intersection(ep_set)) / len(ep_set))
        scores.append(torch.tensor(tokenuni).to("cpu"))
    return torch.max(torch.stack(scores), dim=0)[0]


@torch.no_grad()
def get_images(latents: torch.Tensor, vae: AutoencoderKL):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]


def save_images(pil_images: list[Image.Image], folder_path: str):
    Path(folder_path).mkdir(exist_ok=True)
    for num, im in enumerate(pil_images):
        im.save(f"{folder_path}/{num:02}.png")

def load_state_dict(file_name: str):
    if file_name.endswith(".safetensors"):
        sd = load_file(file_name)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if isinstance(sd[key], torch.Tensor):
            sd[key] = sd[key].to(dtype=torch.float32)

    return sd, metadata


def load_metadata_from_safetensors(safetensors_file: str) -> dict[str, str]:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if not safetensors_file.endswith(".safetensors"):
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


@torch.no_grad()
def generate_images(args: Arguments):
    tokenizer, text_encoder, vae, unet, _, _ = get_models(args)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))
    device = get_devices(args)[0]

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    spm_paths = [lp for lp in args.erased_model_dir.split(",")]
    used_multipliers = []
    network = ACENetwork(
        unet,
        rank=args.ace_lora_rank,
        alpha=1.0,
        module=ACELayer,
    ).to(device)
    aces, metadatas = zip(*[load_state_dict(spm_model_path) for spm_model_path in spm_paths])

    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    erased_prompts_count = [len(ep) for ep in erased_prompts]
    erased_prompts_flatten = [item for sublist in erased_prompts for item in sublist]
    erased_prompt_tokens = tokenize(erased_prompts_flatten, tokenizer)
    erased_prompt_embeds = text_encoder(erased_prompt_tokens.input_ids.to(text_encoder.device))[0]
    
    print(f"erased_prompts is {erased_prompts}")
    
    Path(args.images_dir).mkdir(exist_ok=True)
    prompt = [args.prompt] * args.num_images_per_prompt
    seed = args.seed
    weighted_ace = dict.fromkeys(aces[0].keys())
    prompt_tokens = tokenize([prompt], tokenizer)
    prompt_embeds = text_encoder(prompt_tokens.input_ids.to(text_encoder.device))[0]
    multipliers = calculate_matching_score(
        prompt_tokens,
        prompt_embeds,
        erased_prompt_tokens,
        erased_prompt_embeds,
        matching_metric=args.matching_metric,
        special_token_ids=special_token_ids
    )
    multipliers = torch.split(multipliers, erased_prompts_count)
    ace_multipliers = torch.tensor(multipliers).to("cpu")
    for ace, multiplier in zip(aces, ace_multipliers):
        max_multiplier = torch.max(multiplier)
        for key, value in ace.items():
            if weighted_ace[key] is None:
                weighted_ace[key] = value * max_multiplier
            else:
                weighted_ace[key] += value * max_multiplier
        used_multipliers.append(max_multiplier.item())
    network.load_state_dict(weighted_ace)
    
    num_inference_steps = args.ddim_steps
    guidance_scale = args.guidance_scale
    generator = torch.manual_seed(seed)
    batch_size = len(prompt)
    text_embeddings = get_condition(prompt, tokenizer, text_encoder)
    uncond_embeddings = get_condition([""], tokenizer, text_encoder)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = torch.randn(
        (batch_size, unet.config.in_channels, args.image_size // 8, args.image_size // 8),
        generator=generator,
    ).to(device=unet.device)

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        # predict the noise residual
        with torch.no_grad():
            with network:
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    save_images(
        pil_images=get_images(latents, vae),
        folder_path=f"{args.images_dir}/ace"
    )

def main(args: Arguments):
    generate_images(args)
