from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock

from utils import Arguments

def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class EraserControlMixin:
    _use_eraser = True

    @property
    def use_eraser(self):
        return self._use_eraser

    @use_eraser.setter
    def use_eraser(self, state):
        if not isinstance(state, bool):
            raise AttributeError(f'state should be bool, but got {type(state)}.')
        self._use_eraser = state


class AdapterEraser(nn.Module, EraserControlMixin):
    def __init__(self, dim, mid_dim):
        super().__init__()
        self.down = nn.Linear(dim, mid_dim)
        self.act = nn.GELU()
        self.up = zero_module(nn.Linear(mid_dim, dim))

    def forward(self, hidden_states):
        return self.up(self.act(self.down(hidden_states)))


def load_adapter_params(eraser_ckpt_path):
    adapter_params = {}
    with safe_open(eraser_ckpt_path, framework="pt") as f:
        for key in f.keys():
            if "adapter" in key:
                adapter_params[key] = f.get_tensor(key)
    return adapter_params


def inject_eraser(unet: UNet2DConditionModel, adapter_params, eraser_rank):
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlock):
            print(f'Load eraser at: {name}')
            attn_w_eraser = AttentionWithEraser(module.attn2, eraser_rank)
            # load adapters' parameters
            attn_w_eraser.adapter.load_state_dict({
                'down.weight': adapter_params[f'{name}.adapter.down.weight'],
                'down.bias': adapter_params[f'{name}.adapter.down.bias'],
                'up.weight': adapter_params[f'{name}.adapter.up.weight'],
                'up.bias': adapter_params[f'{name}.adapter.up.bias']
            })
            module.attn2 = attn_w_eraser

class AttentionWithEraser(nn.Module):
    def __init__(self, attn, eraser_rank):
        super().__init__()
        self.attn = attn
        self.adapter = AdapterEraser(attn.to_out[0].weight.shape[1], eraser_rank)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        attn_outputs = self.attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        return self.adapter(attn_outputs) + attn_outputs

def infer(args: Arguments):
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_version)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe._progress_bar_config = {"disable": True}

    unet = UNet2DConditionModel.from_pretrained(args.erased_model_dir)
    adapter_params = load_adapter_params(f"{args.erased_model_dir}/diffusion_pytorch_model.safetensors")
    inject_eraser(unet, adapter_params=adapter_params, eraser_rank=args.receler_rank)
    pipe.unet = unet

    device = f"cuda:{args.device.split(',')[0]}"
    pipe = pipe.to(device)
    generator = torch.Generator(device).manual_seed(args.seed)

    images = pipe(args.prompt, guidance_scale=args.guidance_scale, num_images_per_prompt=args.num_images_per_prompt, generator=generator).images

    Path(args.images_dir).mkdir(exist_ok=True)
    for i in range(len(images)):
        images[i].save(f"{args.images_dir}/{i:02}.png")

def main(args: Arguments):
    infer(args)
