from pathlib import Path

import safetensors
import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from utils import Arguments
from train_methods.train_utils import seed_everything, tokenize
from train_methods.utils_cpe import CPELayer_ResAG, CPENetwork_ResAG

UNET_NAME = "unet"
TEXT_ENCODER_NAME = "text_encoder"


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
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

def load_state_dict(file_name: str, dtype: torch.dtype):
    if file_name.endswith(".safetensors"):
        sd = load_file(file_name)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata

def load_checkpoint_model(checkpoint_path: str, v2: bool = False, clip_skip: int | None = None, device = "cuda") -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, StableDiffusionPipeline]:
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


@torch.no_grad()
def text_encode(text_encoder: CLIPTextModel, tokens: torch.Tensor):
    return text_encoder(tokens.to(text_encoder.device))[0]

def encode_prompts(tokenizer: CLIPTokenizer, text_encoder: CLIPTokenizer, prompts: list[str], return_tokens: bool = False):
    text_tokens = tokenize(prompts, tokenizer).input_ids
    text_embeddings = text_encode(text_encoder, text_tokens)

    if return_tokens:
        return text_embeddings, torch.unique(text_tokens, dim=1)
    return text_embeddings

def infer_with_cpe(args: Arguments, model_path: list[str]):

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

    network = CPENetwork_ResAG(
        unet,
        text_encoder,
        rank=int(float(metadatas[0]["rank"])),
        multiplier=1.0,
        alpha=float(metadatas[0]["alpha"]),
        module=CPELayer_ResAG,
        continual=True,
        task_id=10,
        continual_rank=args.cpe_gate_rank,
        hidden_size=args.cpe_gate_rank,
        init_size=args.cpe_gate_rank,
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
    Path(args.images_dir).mkdir(exist_ok=True)

    with torch.no_grad():
        prompt_embeds = encode_prompts(tokenizer, text_encoder, [args.prompt])
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

        for i, image in enumerate(images):
            image[i].save(f"{args.images_dir}/{i:02}.png")

        network.reset_cache_attention_gate()

def infer(args: Arguments):
    cpe_path = [lp for lp in args.erased_model_dir.split(",")]
    for i in range(len(cpe_path)):
        concept = str(Path(cpe_path[i])).split("/")[1]
        cpe_path[i] = Path(f"{cpe_path[i]}/{concept}")

    infer_with_cpe(args, cpe_path)

def main(args):
    infer(args)
