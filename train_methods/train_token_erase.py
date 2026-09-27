# https://github.com/xszz666/TokenErase
import argparse
import time
import logging
import math
import os
import random
import shutil
import yaml
import pandas as pd
from copy import deepcopy
from pathlib import Path

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange, repeat
from transformers import CLIPTextModel, CLIPTokenizer

from tqdm.auto import tqdm

from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler

from train_methods.train_utils import get_models, get_devices, predict_noise, seed_everything
from train_methods.train_spm import PromptEmbedsCache, PromptEmbedsPair, PromptSettings, get_random_noise
from train_methods.utils_token_erase import process_reference_images, create_mrsa_from_references
from utils import Arguments

logger = get_logger(__name__)


def hack_self_attention_to_mrsa(model, mrsa):
    """
    Hack the original self-attention module to multi-reference self-attention(MRSA) mechanism
    """
    def mrsa_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None, **kwargs):
            """
            The msra is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            # 这里和原始的attention模块一样
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            # 如果注意力有掩膜，则应用
            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)


            # the only difference,注入参考图
            out = mrsa(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale, **kwargs)

            return to_out(out)

        return forward
    
    def hack_attention_module(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':
                net.forward = mrsa_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = hack_attention_module(subnet, count, place_in_unet)
        return count
  
    cross_att_count = 0
    for net_name, net in model.named_children():
        if "down" in net_name:
            cross_att_count += hack_attention_module(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += hack_attention_module(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += hack_attention_module(net, 0, "up")
    mrsa.num_att_layers = cross_att_count

def load_prompts_from_yaml(path, attributes = []):
    with Path(path).open("r") as f:
        prompts = yaml.safe_load(f)
    if len(prompts) == 0:
        raise ValueError("prompts file is empty")
    if len(attributes) != 0:
        newprompts = []
        for i in range(len(prompts)):
            for att in attributes:
                copy_ = deepcopy(prompts[i])
                copy_['target'] = f"{att} {copy_['target']}"
                copy_['positive'] = f"{att} {copy_['positive']}"
                copy_['neutral'] = f"{att} {copy_['neutral']}"
                copy_['unconditional'] = f"{att} {copy_['unconditional']}"
                newprompts.append(copy_)
    else:
        newprompts = deepcopy(prompts)
    return [PromptSettings(**prompt) for prompt in newprompts]

def load_coco_prompts(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        prompts = df['prompt'].tolist()
        logger.info(f"Loaded {len(prompts)} COCO prompts for regularization")
        return prompts
    except Exception as e:
        logger.warning(f"Failed to load COCO prompts: {e}")
        # 如果加载失败，返回一些默认的通用prompt
        return [
            "a photo of a person",
            "a photo of an object", 
            "a photo of a building",
            "a photo of a landscape",
            "a photo of an animal",
            "a picture of a car",
            "an image of a tree",
            "a photo of a room",
            "a picture of food",
            "an image of water"
        ]

def text_tokenize(tokenizer: CLIPTokenizer, prompts: list[str]) -> torch.Tensor:
    return tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids


def text_encode(text_encoder: CLIPTextModel, tokens: torch.Tensor):
    return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(tokenizer: CLIPTokenizer, text_encoder: CLIPTokenizer, prompts: list[str], return_tokens: bool = False):
    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)

    if return_tokens:
        return text_embeddings, torch.unique(text_tokens, dim=1)
    return text_embeddings


def encode_prompts_slider(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTokenizer,
    prompts: list[str],
    sc: float = 1.0,
) -> torch.Tensor:
    text_tokens = text_tokenize(tokenizer, prompts)
    idx = text_tokens.argmax(-1)
    text_embeddings = text_encode(text_encoder, text_tokens)
    batch_indices = torch.arange(len(text_tokens))
    text_embeddings[batch_indices, idx, :] = sc * text_embeddings[batch_indices, idx, :]
    return text_embeddings

def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2
    step = 64
    min_step = min_resolution // step
    max_step = max_resolution // step
    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width

def predict_noise_with_reference(
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    timestep: int,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    ref_latents_noisy: torch.FloatTensor | None = None,
    guidance_scale: float = 7.5,
    **kwargs,
) -> torch.FloatTensor:
    
    if ref_latents_noisy is None:
        return predict_noise(unet, scheduler, timestep, latents, text_embeddings, guidance_scale, **kwargs)
        
    combined_latents = torch.cat([latents, ref_latents_noisy], dim=0)

    latent_model_input = torch.cat([combined_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        **kwargs,
    ).sample
    
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    
    return guided_target[:1]

@torch.no_grad()
def diffusion(
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    total_timesteps,
    guidance_scale=7.5,
    record=False,
    record_type: str | None =None,
    desc=None,
) -> tuple[torch.Tensor, dict[str, dict[str, nn.Parameter]]] | torch.Tensor:

    visualize_map_withstep = {key: {} for key in record_type.strip().split(',')} if record_type is not None else {}

    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[:total_timesteps], desc=desc):

        noise_pred = predict_noise(unet, scheduler, timestep, latents, text_embeddings, guidance_scale)

        if record:
            for type in record_type.strip().split(','):
                for value in unet.attn_processors.values():  # This 'value' is different from the 'value' in CA/SA.
                    for k, v in value.records[type].items():
                        visualize_map_withstep[type][f'{timestep.item()}.{k}'] = v
        
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return (latents, visualize_map_withstep) if record else latents

@torch.no_grad()
def diffusion_with_reference(
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    ref_latents_z_0: torch.FloatTensor | None = None,
    total_timesteps: int = 1000,
    start_timesteps: int = 0,
    guidance_scale: float = 7.5,
    **kwargs,
):
    if ref_latents_z_0 is None:
        return diffusion(
            unet, scheduler, latents, text_embeddings,  total_timesteps, start_timesteps, guidance_scale=guidance_scale, **kwargs
        )
    
    device = latents.device
    dtype = latents.dtype

    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        noise = torch.randn_like(ref_latents_z_0, device=device, dtype=dtype)
        
        if isinstance(timestep, int):
            timestep_tensor = torch.tensor([timestep], device=device)
        else:
            timestep_tensor = timestep
            
        ref_latents_noisy = scheduler.add_noise(ref_latents_z_0, noise, timestep_tensor)
        
        noise_pred = predict_noise_with_reference(
            unet, scheduler, timestep, latents, text_embeddings,
            ref_latents_noisy=ref_latents_noisy, guidance_scale=guidance_scale, **kwargs
        )

        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
    
    return latents

def save_progress(
    text_encoder: CLIPTextModel,
    placeholder_token_ids,
    args,
    save_path,
    safe_serialization=True
):
    logger.info("Saving embeddings")
    learned_embeds = (
        text_encoder.get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--prompts_file", type=str, default=None, help="prompt file.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    #*COCO 添加正则化相关参数
    parser.add_argument(
        "--coco_prompts_path",
        type=str,
        default="textsliders/data/coco_10k.csv",
        help="Path to COCO prompts CSV file for regularization.",
    )
    parser.add_argument(
        "--regularization_weight",
        type=float,
        default=0.1,
        help="Weight for regularization loss to prevent overfitting.",
    )
    parser.add_argument(
        "--use_coco_regularization",
        action="store_true",
        help="Whether to use COCO prompts for regularization. If False, regularization will be disabled.",
    )

    parser.add_argument(
        "--text_train_steps",
        type=int,
        default=200,
        help="Total number of text training steps to perform.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args: Arguments):

    device = get_devices(args)[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    seed_everything(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # load models
    tokenizer, text_encoder, vae, unet, _, noise_scheduler = get_models(args)

    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.encoder.requires_grad_(False)
    text_encoder.final_layer_norm.requires_grad_(False)
    text_encoder.embeddings.position_embedding.requires_grad_(False)

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.train_batch_size

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    #*COCO 加载COCO prompts用于正则化
    if args.use_coco_regularization:
        coco_prompts = load_coco_prompts(args.coco_prompts_path)
        logger.info(f"COCO regularization enabled with weight {args.regularization_weight}")
    else:
        coco_prompts = []
        logger.info("COCO regularization disabled")
    
    attributes = []
    prompts = load_prompts_from_yaml(args.prompts_file, attributes)
    placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)))
    criteria = torch.nn.MSELoss()

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    #! prompt存在cache里
    with torch.no_grad():
        for settings in prompts:
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:

                if isinstance(prompt, list):
                    if prompt == settings.positive:
                        key_setting = 'positive'
                    else:
                        key_setting = 'attributes'
                    if len(prompt) == 0:
                        cache[key_setting] = []
                    else:
                        if cache[key_setting] is None:
                            cache[key_setting] = encode_prompts(
                                tokenizer, text_encoder, prompt
                            )
                else:
                    if cache[prompt] == None:
                        cache[prompt] = encode_prompts(
                            tokenizer, text_encoder, [prompt]
                        )

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(args.max_train_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
    )

    text_encoder.train()
    weight_dtype = torch.float32

    # Move vae and unet to device and cast to weight_dtype
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    ref_data = None
    mrsa = None
    # 检查是否有prompt配置
    if len(prompt_pairs) > 0:
        prompt_pair = prompt_pairs[0]
        first_prompt_settings = prompt_pair.settings
        # 检查YAML配置中是否包含reference_images
        if first_prompt_settings is not None and hasattr(first_prompt_settings, 'reference_images'):
            
            ref_config = first_prompt_settings.reference_images
            
            if isinstance(ref_config, dict) and ref_config.get('image_dir'):
                # 处理参考图像
                ref_data = process_reference_images(  # 使用新函数
                    first_prompt_settings.reference_images, 
                    vae, 
                    device,
                    weight_dtype
                )
                
                if ref_data:
                    # 从YAML获取MRSA配置参数
                    mrsa_start_step = 0
                    mrsa_end_step = 50
                    mrsa_layer_idx = [10,11,12,13,14,15]
                    style_fidelity = 1.0
                    
                    # 如果YAML中有mrsa_config，使用它覆盖默认值
                    if hasattr(first_prompt_settings, 'mrsa_config'):
                        mrsa_config = first_prompt_settings.mrsa_config
                        if isinstance(mrsa_config, dict):
                            mrsa_start_step = mrsa_config.get('start_step', mrsa_start_step)
                            mrsa_end_step = mrsa_config.get('end_step', mrsa_end_step)
                            mrsa_layer_idx = mrsa_config.get('layer_idx', mrsa_layer_idx)
                            style_fidelity = mrsa_config.get('style_fidelity', style_fidelity)
                        
                    
                    # 创建一个临时的args对象来传递MRSA参数
                    class MRSAConfig:
                        def __init__(self):
                            self.mrsa_start_step = mrsa_start_step
                            self.mrsa_end_step = mrsa_end_step
                            self.mrsa_layer_idx = mrsa_layer_idx
                            self.style_fidelity = style_fidelity
                    
                    mrsa_config_obj = MRSAConfig()
                    
                    # 创建MRSA对象
                    mrsa = create_mrsa_from_references(ref_data, mrsa_config_obj)
                    
                    if mrsa:
                        # Hack UNet注意力
                        hack_self_attention_to_mrsa(unet, mrsa)
                        mrsa.enabled = True
                    else:
                        logger.error("Failed to create MRSA object")
                else:
                    logger.error("Failed to process reference images")
            else:
                logger.warning("reference_images configuration is invalid or empty")
        else:
            logger.info("No reference_images found in YAML configuration - using standard training")
    else:
        logger.warning("No prompt pairs found")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(1)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {args.max_train_steps}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )

    # keep original embeddings as reference
    orig_embeds_params = (text_encoder).get_input_embeddings().weight.data.clone()
    run_start_ts = time.perf_counter()
    for _ in range(first_epoch, args.num_train_epochs):
        text_encoder.train()

        # 前text_train_steps个epoch只使用文本，后面的epoch使用图片
        if global_step < args.text_train_steps:
            use_reference = False 
        else:
            use_reference = True

        if use_reference and ref_data["ref_prompts"] and (ref_data["ref_latents_z_0"] is not None):
            num_ref_images = len(ref_data["ref_latents_z_0"])
            selected_ref_idx = random.randint(0, num_ref_images - 1)
        
            current_ref_latents = ref_data["ref_latents_z_0"][selected_ref_idx]
            selected_prompt = random.choice(ref_data["ref_prompts"])

            # 获取当前参考图像的mask
            current_ref_mask = None
            if "ref_masks" in ref_data and ref_data["ref_masks"]:
                current_ref_mask = ref_data["ref_masks"][selected_ref_idx]
                mrsa.ref_masks = [current_ref_mask]
                mrsa.mask_weights = [1.0]
        else:
            selected_prompt = None
            current_ref_latents = None

        with torch.no_grad():
            noise_scheduler.set_timesteps(50, device=device)

            optimizer.zero_grad()

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            # select strength from 0, 1, 2
            sc = float(random.choice([idx for idx in range(3)]))

            # 1 ~ 49 からランダム
            timesteps_to = torch.randint(1, 50, (1,)).item()

            height, width = (
                prompt_pair.resolution,
                prompt_pair.resolution,
            )
            if prompt_pair.dynamic_resolution:
                height, width = get_random_resolution_in_bucket(prompt_pair.resolution)

            latents = (get_random_noise(prompt_pair.batch_size, height, width) * noise_scheduler.init_noise_sigma).to(device, dtype=weight_dtype)

            # 使用目标提示词（含占位符）进行部分去噪，得到随机时间步的隐状态
            #! 修改这里：使用支持参考图像的diffusion函数
            
            if args.use_coco_regularization and coco_prompts:
                coco_prompt = random.choice(coco_prompts)
                coco_prompt_with_placeholder = coco_prompt + f', {placeholder_token}'

                coco_prompt_embeddings_with_placeholder_wo_grad = encode_prompts_slider(
                    tokenizer, text_encoder, [coco_prompt_with_placeholder], sc=sc,
                )
                coco_prompt_embeddings_wo_placeholder_wo_grad = encode_prompts(
                    tokenizer, text_encoder, [coco_prompt],
                )
            else:
                coco_prompt = None
                coco_prompt_embeddings_with_placeholder_wo_grad = None
                coco_prompt_embeddings_wo_placeholder_wo_grad = None

            #*4 使用随机选择的ref img
            # * 这里使用prompt_pair的缓存  prompt_pair.unconditional
            # * 只有target使用 train_util.encode_prompts_slider  sc=sc
            # 编码目标提示词
            target_prompt_text_wo_grad =  prompt_pair.settings.target + f', {placeholder_token}'
            target_prompt_embeddings_wo_grad = encode_prompts_slider(
                tokenizer, text_encoder, [target_prompt_text_wo_grad], sc=sc,
            )
            # 确保embeddings的数据类型
            target_prompt_embeddings_wo_grad = target_prompt_embeddings_wo_grad.to(dtype=weight_dtype)


            # 非参考图文本
            all_prompt_embeddings = [target_prompt_embeddings_wo_grad] 
            all_positive_prompt_embeddings = [prompt_pair.positive.to(device, dtype=weight_dtype)]
            all_unconditional_embeddings = [prompt_pair.unconditional.to(device, dtype=weight_dtype)]

            #* 这里使用随机选择的selected_prompt
            # 参考图文本
            if selected_prompt:
                ref_embed = encode_prompts(
                    tokenizer, text_encoder, [selected_prompt],
                ).to(dtype=weight_dtype)  # 确保数据类型一致
                all_prompt_embeddings.append(ref_embed)
                all_positive_prompt_embeddings.append(ref_embed)
                all_unconditional_embeddings.append(prompt_pair.unconditional.to(device, dtype=weight_dtype))

            # 拼接所有embeddings
            combined_prompt_embeddings = torch.cat(all_prompt_embeddings, dim=0).to(device, dtype=weight_dtype)
            combined_positive_prompt_embeddings = torch.cat(all_positive_prompt_embeddings, dim=0).to(device, dtype=weight_dtype)
            combined_unconditional = torch.cat(all_unconditional_embeddings, dim=0).to(device, dtype=weight_dtype)

            #! 在这里降噪到指定的时间步
            if use_reference:
                mrsa.use_reference = True
            else:
                mrsa.use_reference = False

            # 原始的单次降噪路径
            denoised_latents = diffusion_with_reference(
                unet,
                noise_scheduler,
                latents,
                torch.cat([ 
                    combined_unconditional,
                    combined_prompt_embeddings,
                ]).repeat(
                    prompt_pair.batch_size, dim=0,
                ),
                ref_latents_z_0=current_ref_latents,
                start_timesteps=0,
                total_timesteps=timesteps_to,
                guidance_scale=3,
            )

            if args.use_coco_regularization and coco_prompt_embeddings_with_placeholder_wo_grad is not None:
                # 生成新的latents用于正则化
                coco_latents = (get_random_noise(1, height, width) * noise_scheduler.init_noise_sigma).to(device, dtype=weight_dtype)

                # 对COCO prompt进行相同的去噪过程
                coco_denoised_latents = diffusion_with_reference(
                    unet,
                    noise_scheduler,
                    coco_latents,
                    torch.cat([ 
                        prompt_pair.unconditional.to(device, dtype=weight_dtype)[:1],  # 只取一个batch
                        coco_prompt_embeddings_with_placeholder_wo_grad,
                    ]).repeat(
                        1, dim=0,
                    ),
                    ref_latents_z_0=None,  # COCO不使用参考图像
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )
            else:
                coco_denoised_latents = None
            #! ==== 正则化分支：COCO prompt ==== end

            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 /50)
            ]

            # 在当前时间步的噪声的基础上进行噪声预测
            #! 这里是positive_latents
            if use_reference:
                mrsa.use_reference = True
            else:
                mrsa.use_reference = False

            positive_latents = predict_noise_with_reference(
                unet, noise_scheduler, current_timestep, denoised_latents,
                torch.cat([ 
                    combined_unconditional,
                    combined_positive_prompt_embeddings,
                ]).repeat(
                    prompt_pair.batch_size, dim=0
                ),
                ref_latents_noisy=current_ref_latents,
                guidance_scale=1,
            )[:1]

            positive_latents = positive_latents.to(device, dtype=weight_dtype)

            # 中性和无条件不使用参考图像
            if mrsa is not None:
                mrsa.use_reference = False

            #! 中性 （如"a person"）
            neutral_latents = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                torch.cat([ 
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                ]).repeat(
                    prompt_pair.batch_size, dim=0,
                ).to(device, dtype=weight_dtype),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

            #! unconditional: 无条件参考
            unconditional_latents = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                torch.cat([ 
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                ]).repeat(
                    prompt_pair.batch_size, dim=0,
                ).to(device, dtype=weight_dtype),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

            if args.use_coco_regularization and coco_denoised_latents is not None:
                coco_original_noise = predict_noise(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    coco_denoised_latents,
                    torch.cat([ 
                        prompt_pair.unconditional.to(device, dtype=weight_dtype)[:1],
                        coco_prompt_embeddings_wo_placeholder_wo_grad,
                    ]).repeat(
                        1, dim=0,
                    ),
                    guidance_scale=1,
                ).to(device, dtype=weight_dtype)
            else:
                coco_original_noise = None

        #! 需要训练的参数在这里 target_latents
        with accelerator.accumulate(text_encoder):
            # 编码目标提示词
            #!!!!!!!!!!!修改了这里
            target_prompt_text_with_grad = prompt_pair.settings.target + f', {placeholder_token}'
            target_prompt_embeddings_with_grad = encode_prompts_slider(
                tokenizer, text_encoder, [target_prompt_text_with_grad], sc=sc,
            )
            
            all_prompt_embeddings_with_grad = [target_prompt_embeddings_with_grad]
            # 如果使用参考图像，重新编码所有参考图像的提示词（带梯度）
            if selected_prompt:
                ref_embed_with_grad = encode_prompts(
                    tokenizer, text_encoder, [selected_prompt]
                )
                all_prompt_embeddings_with_grad.append(ref_embed_with_grad)

            combined_prompt_embeddings_with_grad = torch.cat(all_prompt_embeddings_with_grad, dim=0).to(device, dtype=weight_dtype)
            
            if use_reference:
                mrsa.use_reference = True             
            else:
                mrsa.use_reference = False

            # 使用带参考图的噪声预测
            target_latents = predict_noise_with_reference(
                unet, noise_scheduler, current_timestep, denoised_latents,
                torch.cat([ 
                    combined_unconditional,
                    combined_prompt_embeddings_with_grad,
                ]).repeat(
                    prompt_pair.batch_size, dim=0,
                ),
                ref_latents_noisy=current_ref_latents,
                guidance_scale=1,
            )[:1]
            
            target_latents = target_latents.to(device, dtype=weight_dtype)

            #!====== 计算COCO prompt使用当前可学习token的噪声预测（带梯度）==== begin
            if args.use_coco_regularization and coco_denoised_latents is not None:
                if mrsa is not None:
                    mrsa.use_reference = False

                coco_prompt_embeddings_with_placeholder_with_grad = encode_prompts_slider(
                    tokenizer, text_encoder, [coco_prompt_with_placeholder], sc=sc,
                )
                # todo 这里使用附带可学习token的coco prompt
                coco_learned_noise = predict_noise(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    coco_denoised_latents,
                    torch.cat([
                        prompt_pair.unconditional.to(device, dtype=weight_dtype)[:1],
                        coco_prompt_embeddings_with_placeholder_with_grad,
                    ]).repeat(
                        1, dim=0,
                    ),
                    guidance_scale=1,
                ).to(device, dtype=weight_dtype)
            else:
                coco_learned_noise = None
            #!====== 计算COCO prompt使用当前可学习token的噪声预测（带梯度）==== end

            positive_latents.requires_grad = False
            neutral_latents.requires_grad = False
            unconditional_latents.requires_grad = False

            main_loss = prompt_pair.loss(
                target_latents=target_latents,
                positive_latents=positive_latents,
                neutral_latents=neutral_latents,
                unconditional_latents=unconditional_latents,
                scale=sc,
            )

            # 计算正则化损失：让加载token和不加载token时的预测趋于一致
            if args.use_coco_regularization and coco_learned_noise is not None and coco_original_noise is not None:
                coco_original_noise.requires_grad = False
                regularization_loss = F.mse_loss(coco_learned_noise, coco_original_noise)
                total_loss = main_loss + args.regularization_weight * regularization_loss
            else:
                regularization_loss = torch.tensor(0.0, device=device)
                total_loss = main_loss

            total_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

            with torch.no_grad():
                (text_encoder).get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]

                if global_step % args.save_steps == 0:
                    weight_name = (
                        f"learned_embeds-steps-{global_step}.bin"
                        if args.no_safe_serialization
                        else f"learned_embeds-steps-{global_step}.safetensors"
                    )
                    save_path = Path(args.save_dir) / weight_name
                    save_progress(
                        text_encoder,
                        placeholder_token_ids,
                        args,
                        save_path,
                        safe_serialization=not args.no_safe_serialization,
                    )

                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.save_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = Path(args.save_dir) / removing_checkpoint
                                shutil.rmtree(removing_checkpoint)

                    save_path = Path(args.save_dir) / f"checkpoint-{global_step}"
                    text_encoder.save_pretrained(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": total_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
    save_path = os.path.join(args.save_dir, weight_name)
    save_progress(
        text_encoder,
        placeholder_token_ids,
        args,
        save_path,
        safe_serialization=not args.no_safe_serialization,
    )

    total_secs = time.perf_counter() - run_start_ts
    logger.info(f"Total training time: {total_secs/3600:.3f} h ({total_secs/60:.2f} m)")


if __name__ == "__main__":
    main()
