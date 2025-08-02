import random, os
from pathlib import Path
from typing import Optional

import safetensors
import torch
import numpy as np
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from utils import Arguments
from train_methods.utils_cpe import CPELayer_ResAG, CPENetwork_ResAG

UNET_NAME = "unet"
TEXT_ENCODER_NAME = "text_encoder"

def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata

def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata

def load_checkpoint_model(checkpoint_path: str, v2: bool = False, clip_skip: Optional[int] = None, device = "cuda") -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, StableDiffusionPipeline]:
    print(f"Loading checkpoint from {checkpoint_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        checkpoint_path,
        upcast_attention=True if v2 else False,
    ).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    if clip_skip is not None:
        if v2:
            text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
        else:
            text_encoder.config.num_hidden_layers = 12 - (clip_skip - 1)

    return tokenizer, text_encoder, unet, pipe

def text_tokenize(tokenizer: CLIPTokenizer, prompts: list[str]):
    return tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids

@torch.no_grad()
def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.to(text_encoder.device))[0]

def encode_prompts(tokenizer: CLIPTokenizer, text_encoder: CLIPTokenizer, prompts: list[str], return_tokens: bool = False):
    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)

    if return_tokens:
        return text_embeddings, torch.unique(text_tokens, dim=1)
    return text_embeddings

def infer_with_cpe(
    args: Arguments,
    model_path: list[str],
    config: GenerationConfig,
):

    model_paths = model_path
    device = f"cuda:{args.device.split(',')[0]}"

    tokenizer, text_encoder, unet, pipe = load_checkpoint_model(args.sd_version, "v2" in args.sd_version)
    text_encoder.to(device)
    text_encoder.eval()

    unet.to(device)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    cpes, metadatas = zip(*[load_state_dict(model_path, torch.float32) for model_path in model_paths])
        
    # check if CPEs are compatible
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])

    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    print(f"Erased prompts: {erased_prompts}")

    print(metadatas[0])
    network = CPENetwork_ResAG(
        unet,
        text_encoder,
        rank=int(float(metadatas[0]["rank"])),
        multiplier=1.0,
        alpha=float(metadatas[0]["alpha"]),
        module=CPELayer_ResAG,
        continual=True,
        task_id=10,
        continual_rank=config.gate_rank,
        hidden_size=config.gate_rank,
        init_size=config.gate_rank,  
        n_concepts=len(model_paths),
    ).to(device)
    
    for k,v in network.named_parameters():
        for idx in range(len(cpes)):
            if len(v.shape) > 1:
                v.data[idx,:] = cpes[idx][k]
            else:
                v.data[idx] = cpes[idx][k]
                
    network.to(device)
    network.eval()
    network.set_inference_mode()
    
    prompt = args.prompt
    with torch.no_grad():
        prompt_embeds = encode_prompts(tokenizer, text_encoder, [prompt])
        network.reset_cache_attention_gate()
        seed_everything(args.seed)

        with network:
            images = pipe(
                negative_prompt=args.negative_prompt,
                width=args.image_size,
                height=args.image_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.cuda.manual_seed(args.seed),
                num_images_per_prompt=args.num_images_per_prompt,
                prompt_embeds=prompt_embeds,
            ).images
        
        os.makedirs(args.images_dir, exist_ok=True)
        for i, image in enumerate(images):
            image[i].save(f"{args.images_dir}/{i:02}.png")
        
        network.reset_cache_attention_gate()

def infer(args: Arguments):
    spm_path = [lp for lp in args.erased_model_dir.split(",")]
    for i in range(len(spm_path)):
        concept = str(Path(spm_path[i])).split("/")[1]
        spm_path[i] = Path(f"{spm_path[i]}/{concept}")
        
    generation_config = GenerationConfig(**{
        "prompts": [args.prompt],
        "generate_num": 5,
        "save_path": args.images_dir
    })
            
    infer_with_cpe(spm_path, generation_config, args)

def main(args):
    infer(args)


def main(args):
    concepts_folder = os.listdir(args.model_path[0])    
    concepts_ckpt = []
    
    for folder in concepts_folder:
        for ckpt in os.listdir(os.path.join(args.model_path[0],folder)):
            if ("last.safetensors" in ckpt) and ("adv_prompts" not in ckpt):
                concepts_ckpt.append(os.path.join(args.model_path[0],folder,ckpt))

    model_path = [Path(lp) for lp in concepts_ckpt]
    
    generation_config = load_config_from_yaml(args.config)

    if args.st_prompt_idx != -1:
        generation_config.st_prompt_idx = args.st_prompt_idx
    if args.end_prompt_idx != -1:
        generation_config.end_prompt_idx = args.end_prompt_idx
    if args.gate_rank != -1:
        generation_config.gate_rank = args.gate_rank
    
    
    generation_config.save_path = os.path.join("/".join(generation_config.save_path.split("/")[:-3]), args.save_env, "/".join(generation_config.save_path.split("/")[-2:]))

    infer_with_cpe(
        model_path,
        generation_config,
        base_model=args.base_model,
        v2=args.v2,
        precision=args.precision,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/generation.yaml",
        help="Base configs for image generation.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        nargs="*",
        help="CPE model to use.",
    )
    # model configs
    parser.add_argument(
        "--base_model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base model for generation.",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the 2.x version of the SD.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision for the base model.",
    )
    parser.add_argument(
        "--save_env",
        type=str,
        default="",
        help="Precision for the base model.",
    )    
    
    parser.add_argument(
        "--st_prompt_idx",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--end_prompt_idx",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--gate_rank",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    main(args)

