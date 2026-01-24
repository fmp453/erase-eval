# ACE: Anti-Editing Concept Erasure in Text-to-Image Models

import gc
import json
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from safetensors.torch import save_file
from diffusers import UNet2DConditionModel, DDIMScheduler
from pathlib import Path
from tqdm import tqdm

from utils import Arguments
from train_methods.data import AnchorsDataset
from train_methods.train_utils import get_condition, get_models, predict_noise, get_devices


class ACELayer(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.

    """

    def __init__(
        self,
        ace_name: str,
        org_module: nn.Linear | nn.Conv2d,
        multiplier: float=1.0,
        dim: int=4,
        alpha: int=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.dim = dim
        self.ace_name = ace_name

        if org_module.__class__.__name__ == "Linear" and isinstance(org_module, nn.Linear):
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, dim, bias=False)
            self.lora_up = nn.Linear(dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv2d" and isinstance(org_module, nn.Conv2d):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(in_dim, self.dim, kernel_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            raise ValueError("org_module must be Linear or Conv2d")

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().numpy()
        alpha = dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale


class ACENetwork(nn.Module):
    UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
        "Transformer2DModel",
    ]
    UNET_TARGET_REPLACE_MODULE_CONV = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]

    ACE_PREFIX_UNET = "lora_unet"  # aligning with SD webui usage
    DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        module = ACELayer,
        module_kwargs = None,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.dim = rank
        self.alpha = alpha

        self.module = module
        self.module_kwargs = module_kwargs or {}

        # unet ace
        self.unet_ace_layers = self.create_modules(
            ACENetwork.ACE_PREFIX_UNET,
            unet,
            ACENetwork.DEFAULT_TARGET_REPLACE,
            self.dim,
            self.multiplier,
        )
        print(f"Create lora for U-Net: {len(self.unet_ace_layers)} modules.")

        ace_names = set()
        for ace_layer in self.unet_ace_layers:
            assert (
                ace_layer.ace_name not in ace_names
            ), f"duplicated lora layer name: {ace_layer.ace_name}. {ace_names}"
            ace_names.add(ace_layer.ace_name)

        for ace_layer in self.unet_ace_layers:
            ace_layer.apply_to()
            self.add_module(ace_layer.ace_name, ace_layer,)

        del unet
        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: list[str],
        rank: int,
        multiplier: float,
    ) -> list[ACELayer]:
        ace_layers = []

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d"] and "ip" not in child_name:
                        ace_name = f"{prefix}.{name}.{child_name}".replace(".", "_")
                        ace_layer = self.module(ace_name, child_module, multiplier, rank, self.alpha, **self.module_kwargs)
                        ace_layers.append(ace_layer)

        return ace_layers

    def prepare_optimizer_params(self, default_lr: float) -> list[dict[str, list[nn.Parameter] | float]]:
        all_params = []

        if self.unet_ace_layers:
            params = []
            [params.extend(ace_layer.parameters()) for ace_layer in self.unet_ace_layers]
            param_data = {"params": params}
            if default_lr is not None:
                param_data["lr"] = default_lr
            all_params.append(param_data)

        return all_params

    def save_weights(self, file: str, metadata: dict | None=None):
        state_dict = self.state_dict()

        for key in list(state_dict.keys()):
            if not key.startswith("lora"):
                del state_dict[key]

        if file.endswith(".safetensors"):
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def __enter__(self):
        for ace_layer in self.unet_ace_layers:
            ace_layer.multiplier = 1.0

    def __exit__(self, exc_type, exc_value, tb):
        for ace_layer in self.unet_ace_layers:
            ace_layer.multiplier = 0        


class InfiniteDataLoader(DataLoader):
    def __iter__(self):
        return self.iter_function()

    def iter_function(self):
        while True:
            for batch in super().__iter__():
                yield batch

@torch.no_grad()
def diffusion_to_get_x_t(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    latents: torch.Tensor, 
    text_embeddings: torch.Tensor,
    total_timesteps: int = 1000,
    get_t: int = 1000,
) -> torch.Tensor:
    step = 0
    for timestep in scheduler.timesteps[:total_timesteps]:
        if step is get_t:
            break
        noise_pred = predict_noise(unet, scheduler, timestep, latents, text_embeddings)

        # compute the previous noisy sample x_t -> x_t-1
        latents: torch.Tensor = scheduler.step(noise_pred, timestep, latents).prev_sample
        step = step + 1

    return latents


def train(args: Arguments):

    torch.autograd.set_detect_anomaly(True)
    devices = get_devices(args)
    prompt = args.concepts

    if args.seperator is not None:
        words = prompt.split(args.seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]

    tokenizer, text_encoder, _, unet, noise_scheduler, _ = get_models(args.sd_version)

    unet.to(devices[0])
    text_encoder.to(devices[1])
    scheduler_ori = deepcopy(noise_scheduler)
    scheduler_ori.set_timesteps(1000)
    unet.requires_grad_(False)
    unet.eval()
    text_encoder.eval()
    lora_alpha = 1.0
    network = ACENetwork(
        unet,
        rank=args.ace_lora_rank,
        multiplier=1.0,
        alpha=lora_alpha,
        module=ACELayer,
    ).to(device=devices[0])
    model_metadata = {
        "prompts": ",".join(words),
        "rank": f"{args.ace_lora_rank}",
        "alpha": f"{lora_alpha}",
    }

    unet_lora_params = network.prepare_optimizer_params(args.ace_lr)

    losses = []
    opt = torch.optim.Adam(unet_lora_params, lr=args.ace_lr)
    criteria = torch.nn.MSELoss()
    history = []
    is_sc_clip = args.ace_surrogate_concept_clip_path is not None
    if not is_sc_clip:
        sc_clip = None
    else:
        args.ace_surrogate_concept_clip_path = args.ace_surrogate_concept_clip_path.replace("CONCEPT", args.concepts)
        with open(args.ace_surrogate_concept_clip_path, "r") as f:
            sc_clip = json.load(f)

    anchor_dataset = AnchorsDataset(prompt_path=args.ace_anchor_prompt_path, concept=prompt)

    pbar = tqdm(range(args.ace_iterations))
    anchor_dataloader = InfiniteDataLoader(anchor_dataset, args.ace_anchor_batch_size, shuffle=True)
    
    for _, data in zip(pbar, anchor_dataloader):
        word = random.sample(words, 1)[0]
        # get text embeddings for unconditional and conditional prompts
        emb_0 = get_condition([''], tokenizer, text_encoder)
        emb_p = get_condition(word, tokenizer, text_encoder)
        emb_n = get_condition(word, tokenizer, text_encoder)
        emb_anchor = get_condition(data, tokenizer, text_encoder)
        emb_anchor = torch.cat([emb_0.repeat(len(data), 1, 1), emb_anchor], dim=0)
        opt.zero_grad()

        t_end = torch.randint(int((1 - args.ace_change_step_rate) * args.ddim_steps), args.ddim_steps, (1,), device=devices[0])
        init_latent = torch.randn((1, 4, 64, 64)).to(devices[0])
        init_latent = init_latent * noise_scheduler.init_noise_sigma
        with torch.no_grad():
            noise_scheduler.set_timesteps(args.ddim_steps)
            # generate an image with the concept from ESD model
            # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z
            with network:
                latent_t: torch.Tensor = diffusion_to_get_x_t(
                    unet,
                    noise_scheduler,
                    init_latent,
                    torch.cat([emb_0, emb_p]).repeat_interleave(1, dim=0),
                    total_timesteps=args.ddim_steps,
                    guidance_scale=args.start_guidance,
                    get_t=t_end,
                )
            # set training timestep
            noise_scheduler.set_timesteps(1000)
            current_timestep = noise_scheduler.timesteps[int(t_end * 1000 / args.ddim_steps)]
            e_prior_ori = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                latent_t.repeat(len(data), 1, 1, 1),
                emb_anchor,
            ).to(devices[0])
            e_prior_ori.requires_grad = False
            e_0 = predict_noise(
                unet,
                scheduler_ori,
                current_timestep,
                latent_t,
                torch.cat([emb_0, emb_0]).repeat_interleave(1, dim=0),
            ).to(devices[0])
            conditional_embedding = torch.cat([emb_0, emb_p]).repeat_interleave(1, dim=0)
            e_p = predict_noise(unet, scheduler_ori, current_timestep, latent_t, conditional_embedding).to(devices[0])
        # get conditional score from ESD model
        with network:
            e_n_0 = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                latent_t,
                torch.cat([emb_0, emb_0]).repeat_interleave(1, dim=0),
            ).to(devices[0])
        with network:
            e_n = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                latent_t,
                torch.cat([emb_0, emb_n]).repeat_interleave(1, dim=0),
            ).to(devices[0])
        with network:
            e_prior = predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                latent_t.repeat(len(data), 1, 1, 1),
                emb_anchor,
            ).to(devices[0])

        e_p.requires_grad = False

        surrogate_guidance = torch.zeros_like(latent_t)
        if is_sc_clip:
            own_clip = sc_clip[0][prompt]
        for j in range(len(data)):
            if is_sc_clip:
                sc_clip_tem = sc_clip[0][data[j]]
                clip_scale = sc_clip_tem / own_clip
                surrogate_guidance += clip_scale * (e_prior_ori[j] - e_0)
            else:
                surrogate_guidance += e_prior_ori[j] - e_0
        
        loss_erase_null: torch.Tensor = criteria(e_n_0, e_0 + args.negative_guidance * (e_p - e_0) - args.ace_surrogate_guidance_scale * surrogate_guidance)
        loss_erase_cond: torch.Tensor = criteria(e_n, e_0 - (args.negative_guidance * (e_p - e_0)))

        loss_erase = args.ace_null_weight * loss_erase_null + (1 - args.ace_null_weight) * loss_erase_cond
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        # loss = criteria(e_n, e_0) works the best try 5000 epochs

        loss_prior: torch.Tensor = criteria(e_prior, e_prior_ori)
        loss = (1 - args.ace_pl_weight) * loss_erase + args.ace_pl_weight * loss_prior
        # update weights to erase the concept
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix({
            "loss": loss.item(),
            "cond_loss": loss_erase_cond.item(),
            "null_loss": loss_erase_null.item(),
            "loss_prior": loss_prior.item()
        })
        history.append(loss.item())
        opt.step()
        del latent_t, e_prior, e_prior_ori, e_p, emb_anchor
        torch.cuda.empty_cache()
        gc.collect()

    folder_path = f'{args.save_dir}/ace/{args.concepts.replace(" ", "_")}'
    Path(folder_path).mkdir(exist_ok=True)
    network.save_weights(
        Path(folder_path, "model.safetensors"),
        metadata=model_metadata,
    )

def main(args: Arguments):
    train(args)
