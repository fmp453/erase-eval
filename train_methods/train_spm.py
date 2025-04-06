# One-dimensional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing Applications (SPM)

# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py

import gc
import random

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, model_validator

import torch
import bitsandbytes as bnb
from tqdm import tqdm

from train_methods.utils_spm import SPMNetwork, SPMLayer

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, SchedulerMixin, PNDMScheduler
from diffusers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType

from utils import Arguments

UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

ACTION_TYPES = Literal["erase", "erase_with_la"]

class PromptEmbedsCache:
    prompts: dict[str, torch.FloatTensor] = {}

    def __setitem__(self, __name: str, __value: torch.FloatTensor) -> None:
        self.prompts[__name] = __value

    def __getitem__(self, __name: str):
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None

class PromptSettings(BaseModel):  # yaml
    target: str
    positive: str = None  # if None, target will be used
    unconditional: str = ""  # default is ""
    neutral: str = None  # if None, unconditional will be used
    action: ACTION_TYPES = "erase"  # default is "erase"
    guidance_scale: float = 1.0  # default is 1.0
    resolution: int = 512  # default is 512
    dynamic_resolution: bool = False  # default is False
    batch_size: int = 1  # default is 1
    dynamic_crops: bool = False  # default is False. only used when model is XL
    use_template: bool = False  # default is False
    
    la_strength: float = 1000.0
    sampling_batch_size: int = 4

    seed: int = None
    case_number: int = 0

    @model_validator(mode='before')
    def fill_prompts(cls, values):
        keys = values.keys()
        if "target" not in keys:
            raise ValueError("target must be specified")
        if "positive" not in keys:
            values["positive"] = values["target"]
        if "unconditional" not in keys:
            values["unconditional"] = ""
        if "neutral" not in keys:
            values["neutral"] = values["unconditional"]

        return values

class PromptEmbedsPair:
    target: torch.FloatTensor  # the concept that do not want to generate 
    positive: torch.FloatTensor  # generate the concept
    unconditional: torch.FloatTensor  # uncondition (default should be empty)
    neutral: torch.FloatTensor  # base condition (default should be empty)
    use_template: bool = False  # use clip template or not

    guidance_scale: float
    resolution: int
    batch_size: int
    dynamic_crops: bool

    loss_fn: torch.nn.Module
    action: ACTION_TYPES

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        target: torch.FloatTensor,
        positive: torch.FloatTensor,
        unconditional: torch.FloatTensor,
        neutral: torch.FloatTensor,
        settings=None #: PromptSettings,
    ) -> None:
        self.loss_fn = loss_fn
        self.target = target
        self.positive = positive
        self.unconditional = unconditional
        self.neutral = neutral

        if settings is None:
            # PromptSettingのデフォルト値を採用
            self.use_template = False
            self.guidance_scale = 1.0
            self.resolution = 512
            self.batch_size = 1
            self.dynamic_crops = False
            self.action = "erase_with_la"
            self.la_strength = 1000.0
            self.sampling_batch_size = 4        
        else:
            self.use_template = settings.use_template
            self.guidance_scale = settings.guidance_scale
            self.resolution = settings.resolution
            self.dynamic_resolution = settings.dynamic_resolution
            self.batch_size = settings.batch_size
            self.dynamic_crops = settings.dynamic_crops
            self.action = settings.action
            self.la_strength = settings.la_strength
            self.sampling_batch_size = settings.sampling_batch_size
    
    def _erase(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        neutral_latents: torch.FloatTensor,  # ""
        **kwargs,
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""

        erase_loss = self.loss_fn(
            target_latents,
            neutral_latents
            - self.guidance_scale * (positive_latents - neutral_latents),
        )
        losses = {
            "loss": erase_loss,
            "loss/erase": erase_loss,
        }
        return losses
    
    def _erase_with_la(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        neutral_latents: torch.FloatTensor,  # ""
        anchor_latents: torch.FloatTensor, 
        anchor_latents_ori: torch.FloatTensor, 
        **kwargs,
    ):
        anchoring_loss = self.loss_fn(anchor_latents, anchor_latents_ori)
        erase_loss = self._erase(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
        )["loss/erase"]
        losses = {
            "loss": erase_loss + self.la_strength * anchoring_loss,
            "loss/erase": erase_loss,
            "loss/anchoring": anchoring_loss
        }
        return losses

    def loss(self, **kwargs,):
        if self.action == "erase":
            return self._erase(**kwargs)
        else:
            return self._erase_with_la(**kwargs)
        
def flush():
    torch.cuda.empty_cache()
    gc.collect()

def get_scheduler_fix(optimizer, iterations, lr_scheduler_num_cycles, lr_warmup_steps, num_processes: int = 1):
    num_training_steps = iterations * num_processes  
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[SchedulerType("cosine_with_restarts")]
    return schedule_func(optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=num_training_steps, num_cycles=lr_scheduler_num_cycles)

def text_tokenize(tokenizer: CLIPTokenizer, prompts: list[str]):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

def encode_prompts(tokenizer, text_encoder, prompts: list[str], return_tokens: bool = False):
    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encoder(text_tokens.to(text_encoder.device))[0]

    if return_tokens:
        return text_embeddings, torch.unique(text_tokens, dim=1)
    return text_embeddings

def get_random_noise(batch_size: int, height: int, width: int, generator: torch.Generator=None) -> torch.Tensor:
    return torch.randn(
        (batch_size, UNET_IN_CHANNELS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR),
        generator=generator,
        device="cpu",
    )

def get_initial_latents(scheduler, n_imgs, height, width, n_prompts, generator=None):
    noise = get_random_noise(n_imgs, height, width, generator=generator).repeat(n_prompts, 1, 1, 1)
    latents = noise * scheduler.init_noise_sigma
    return latents

def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    guidance_scale=7.5,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # predict the noise residual
    noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return guided_target

@torch.no_grad()
def diffusion(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor, 
    text_embeddings: torch.FloatTensor,
    total_timesteps: int = 1000,
    start_timesteps=0,
    guidance_scale = 7.5
):

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise(unet, scheduler, timestep, latents, text_embeddings, guidance_scale=guidance_scale)
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return latents

def concat_embeddings(unconditional: torch.FloatTensor, conditional: torch.FloatTensor, n_imgs: int):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)

def sample(prompt_pair: PromptEmbedsPair):
    samples = []
    while len(samples) < prompt_pair.sampling_batch_size:
        while True:
            # sample from gaussian distribution
            noise = torch.randn_like(prompt_pair.target)
            # normalize the noise
            noise = noise / noise.view(-1).norm(dim=-1)
            # compute the similarity
            sim = torch.cosine_similarity(prompt_pair.target.view(-1), noise.view(-1), dim=-1)
            # the possibility of accepting the sample = 1 - sim
            if random.random() < 1 - sim:
                break
        scale = random.random() * 0.4 + 0.8
        sample = scale * noise * prompt_pair.target.view(-1).norm(dim=-1)
        samples.append(sample)
    
    samples = [torch.cat([prompt_pair.unconditional, s]) for s in samples]
    samples = torch.cat(samples, dim=0)
    return samples

def train(
    args: Arguments,
    save_path: str,
    network_rank: int=1,
    network_alpha: float=1.0,
    text_encoder_lr: float = 5e-5,
    lr: float = 1e-4,
    unet_lr: float = 1e-4,
    iterations: int = 3000,
    lr_warmup_steps: int = 500,
    lr_scheduler_num_cycles: int = 3,
    max_grad_norm: float = 0.0,
    max_denoising_steps: int = 30,
    resolution: int=512,
    save_name: str = "untitled",
    prompts: list[PromptSettings] = []
):
    model_metadata = {
        "prompts": ",".join([prompt.target for prompt in prompts]),
        "rank": str(network_rank),
        "alpha": str(network_alpha),
    }
    save_path = Path(save_path)

    noise_scheduler = PNDMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    device = torch.device(f'cuda:{args.device.split(",")[0]}')
    
    text_encoder.to(device)
    text_encoder.eval()

    unet.to(device)
    unet.requires_grad_(False)
    unet.eval()

    network = SPMNetwork(
        unet,
        rank=network_rank,
        multiplier=1.0,
        alpha=network_alpha,
        module=SPMLayer,
    ).to(device)

    trainable_params = network.prepare_optimizer_params(text_encoder_lr, unet_lr, lr)
    
    optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr)
    
    lr_scheduler = get_scheduler_fix(optimizer, iterations, lr_scheduler_num_cycles, lr_warmup_steps)
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    cache = PromptEmbedsCache()
    prompt_pairs = [] # list[PromptEmbedsPair]

    with torch.no_grad():
        for settings in prompts:
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] == None:
                    cache[prompt] = encode_prompts(tokenizer, text_encoder, [prompt])

            prompt_pair = PromptEmbedsPair(
                criteria,
                cache[settings.target],
                cache[settings.positive],
                cache[settings.unconditional],
                cache[settings.neutral],
                settings,
            )
            assert prompt_pair.sampling_batch_size % prompt_pair.batch_size == 0
            prompt_pairs.append(prompt_pair)
            print(f"norm of target: {prompt_pair.target.norm()}")

    flush()
    pbar = tqdm(range(iterations))
    loss = None

    for i in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(max_denoising_steps, device=device)
            optimizer.zero_grad()

            prompt_pair: PromptEmbedsPair = prompt_pairs[torch.randint(0, len(prompt_pairs), (1,)).item()]

            timesteps_to = torch.randint(1, max_denoising_steps, (1,)).item()

            height, width = (resolution, resolution)
            latents = get_initial_latents(noise_scheduler, prompt_pair.batch_size, height, width, 1).to(device)

            with network:
                denoised_latents = diffusion(
                    unet,
                    noise_scheduler,
                    latents,
                    concat_embeddings(prompt_pair.unconditional, prompt_pair.target, prompt_pair.batch_size,),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[int(timesteps_to * 1000 / max_denoising_steps)]

            positive_latents = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu")
            neutral_latents = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                concat_embeddings(prompt_pair.unconditional, prompt_pair.neutral, prompt_pair.batch_size),
                guidance_scale=1,
            ).to("cpu")

        with network:
            target_latents = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                concat_embeddings(prompt_pair.unconditional, prompt_pair.target, prompt_pair.batch_size,),
                guidance_scale=1,
            ).to("cpu")

        # ------------------------- latent anchoring part -----------------------------

        if prompt_pair.action == "erase_with_la":
            # noise sampling
            anchors = sample(prompt_pair)

            # get latents
            repeat = prompt_pair.sampling_batch_size // prompt_pair.batch_size
            with network:
                anchor_latents = predict_noise(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents.repeat(repeat, 1, 1, 1),
                    anchors,
                    guidance_scale=1,
                ).to("cpu")

            with torch.no_grad():
                anchor_latents_ori = predict_noise(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents.repeat(repeat, 1, 1, 1),
                    anchors,
                    guidance_scale=1,
                ).to("cpu")
            anchor_latents_ori.requires_grad_ = False

        else:
            anchor_latents = None
            anchor_latents_ori = None

        # --------------------------------------------------------------

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False

        loss = prompt_pair.loss(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            anchor_latents=anchor_latents,
            anchor_latents_ori=anchor_latents_ori,
        )

        loss["loss"].backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm, norm_type=2)
        optimizer.step()
        lr_scheduler.step()

        pbar.set_description(f"Loss*1k: {loss['loss'].item()*1000:.4f}")

        del (
            positive_latents,
            neutral_latents,
            target_latents,
            latents,
            anchor_latents,
            anchor_latents_ori,
        )
        flush()

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(save_path / f"{save_name}_last.safetensors", metadata=model_metadata)

    del (unet, noise_scheduler, loss, optimizer, network)
    flush()
    print("Done.")


def main(args: Arguments):
    args.concepts = args.concepts.split(",")[0]
    train(
        args,
        save_path=f"{args.save_dir}/{args.concepts.replace(' ', '-')}",
        save_name=args.concepts.replace(" ", "-"),
        prompts=[PromptSettings(
            target=args.concepts,
            positive=args.concepts,
            unconditional="",
            action="erase_with_la",
            guidance_scale=1.0,
            resolution=args.image_size,
            batch_size=1,
            dynamic_resolution=True,
            la_strength=1000,
            sampling_batch_size=4
        )]
    )


