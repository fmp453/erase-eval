import gc
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, StableDiffusionPipeline

from utils import Arguments
from infer_methods.infer_utils import load_state_dict
from train_methods.utils_spm import SPMNetwork, SPMLayer

UNET_NAME = "unet"
TEXT_ENCODER_NAME = "text_encoder"
MATCHING_METRICS = Literal["clipcos", "clipcos_tokenuni", "tokenuni"]

class GenerationConfig(BaseModel):
    prompts: list[str] = []
    negative_prompt: str = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    unconditional_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 2024
    generate_num: int = 1

    save_path: str = None  # can be a template, e.g. "path/to/img_{}.png",
    # then the generated images will be saved as "path/to/img_0.png", "path/to/img_1.png", ...

    def dict(self):
        results = {}
        for attr in vars(self):
            if not attr.startswith("_"):
                results[attr] = getattr(self, attr)
        return results
    
    @staticmethod
    def fix_format(cfg):
        for k, v in cfg.items():
            if isinstance(v, list):
                cfg[k] = v[0]
            elif isinstance(v, torch.Tensor):
                cfg[k] = v.item()


def load_checkpoint_model(checkpoint_path: str, v2: bool = False, clip_skip: int | None = None, device="cuda") -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, StableDiffusionPipeline]:
    print(f"Loading checkpoint from {checkpoint_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        checkpoint_path,
        upcast_attention=True if v2 else False,
    ).to(device)

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder: CLIPTextModel = pipe.text_encoder
    if clip_skip is not None:
        if v2:
            text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
        else:
            text_encoder.config.num_hidden_layers = 12 - (clip_skip - 1)

    return tokenizer, text_encoder, unet, pipe


def text_tokenize(tokenizer: CLIPTokenizer, prompts: list[str]):
    return tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids


@torch.no_grad()
def text_encode(text_encoder: CLIPTextModel, tokens: torch.Tensor):
    return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(tokenizer: CLIPTokenizer, text_encoder: CLIPTokenizer, prompts: list[str], return_tokens: bool = False):
    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)

    if return_tokens:
        return text_embeddings, torch.unique(text_tokens, dim=1)
    return text_embeddings

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def calculate_matching_score(
    prompt_tokens: torch.Tensor,
    prompt_embeds: torch.Tensor,
    erased_prompt_tokens: torch.Tensor,
    erased_prompt_embeds: torch.Tensor,
    matching_metric: MATCHING_METRICS,
    special_token_ids: set[int],
):
    scores = []
    if "clipcos" in matching_metric:
        clipcos = torch.cosine_similarity(prompt_embeds.flatten(1, 2), erased_prompt_embeds.flatten(1, 2), dim=-1).cpu()
        scores.append(clipcos)
    if "tokenuni" in matching_metric:
        prompt_set = set(prompt_tokens[0].tolist()) - special_token_ids
        tokenuni = []
        for ep in erased_prompt_tokens:
            ep_set = set(ep.tolist()) - special_token_ids
            tokenuni.append(len(prompt_set.intersection(ep_set)) / len(ep_set))
        scores.append(torch.tensor(tokenuni).to("cpu"))
    return torch.max(torch.stack(scores), dim=0)[0]

def infer_with_spm(spm_paths: list[Path], config: GenerationConfig, args: Arguments):
    
    base_model = args.sd_version
    v2 = "v2" in args.sd_version
    assigned_multipliers = None
    matching_metric = args.matching_metric
    save_path = config.save_path

    Path(save_path).mkdir(exist_ok=True)
    device = f"cuda:{args.device.split(',')[0]}"

    spm_model_paths = [lp / f"{lp.name}_last.safetensors" if lp.is_dir() else lp for lp in spm_paths]

    tokenizer, text_encoder, unet, pipe = load_checkpoint_model(base_model, v2=v2, device=device)
    pipe._progress_bar_config = {"disable": True}
    special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))

    text_encoder.to(device)
    text_encoder.eval()

    unet.to(device)
    unet.requires_grad_(False)
    unet.eval()

    # load the SPM modules
    spms, metadatas = zip(*[load_state_dict(spm_model_path, torch.float32) for spm_model_path in spm_model_paths])
    # check if SPMs are compatible
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])

    # get the erased concept
    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    erased_prompts_count = [len(ep) for ep in erased_prompts]
    print(f"Erased prompts: {erased_prompts}")

    erased_prompts_flatten = [item for sublist in erased_prompts for item in sublist]
    erased_prompt_embeds, erased_prompt_tokens = encode_prompts(tokenizer, text_encoder, erased_prompts_flatten, return_tokens=True)

    network = SPMNetwork(
        unet,
        rank=int(float(metadatas[0]["rank"])),
        alpha=float(metadatas[0]["alpha"]),
        module=SPMLayer,
    ).to(device)

    with torch.no_grad():
        for prompt in config.prompts:
            prompt += config.unconditional_prompt
            print(f"Generating for prompt: {prompt}")
            prompt_embeds, prompt_tokens = encode_prompts(tokenizer, text_encoder, [prompt], return_tokens=True)
            if assigned_multipliers is not None:
                multipliers = torch.tensor(assigned_multipliers).to("cpu")
                if assigned_multipliers == [0,0,0]:
                    matching_metric = "aazeros"
                elif assigned_multipliers == [1,1,1]:
                    matching_metric = "zzone"
            else:
                multipliers = calculate_matching_score(
                    prompt_tokens,
                    prompt_embeds, 
                    erased_prompt_tokens, 
                    erased_prompt_embeds, 
                    matching_metric=matching_metric,
                    special_token_ids=special_token_ids
                )
                multipliers = torch.split(multipliers, erased_prompts_count)
            print(f"multipliers: {multipliers}")
            weighted_spm = dict.fromkeys(spms[0].keys())
            used_multipliers = []
            for spm, multiplier in zip(spms, multipliers):
                max_multiplier = torch.max(multiplier)
                for key, value in spm.items():
                    if weighted_spm[key] is None:
                        weighted_spm[key] = value * max_multiplier
                    else:
                        weighted_spm[key] += value * max_multiplier
                used_multipliers.append(max_multiplier.item())
            network.load_state_dict(weighted_spm)
            with network:
                images = pipe(
                    negative_prompt=config.negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=torch.cuda.manual_seed(config.seed),
                    num_images_per_prompt=config.generate_num,
                    prompt_embeds=prompt_embeds,
                ).images
            
            for i in range(len(images)):
                images[i].save(f"{save_path}/{i:02}.png")
                
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
            
    infer_with_spm(spm_path, generation_config, args)

def main(args):
    infer(args)
