# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import json
from typing import Optional, Type

import torch
import torch.nn as nn
from safetensors.torch import save_file as safe_save

TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}
UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}
DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE
EMBED_FLAG = "<embed>"

class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}")
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return (self.linear(input) + self.dropout(self.lora_up(self.selector(self.lora_down(input)))) * self.scale)

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(self.lora_up.weight.device).to(self.lora_up.weight.dtype)


class LoraInjectedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 4,
        dropout_p: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            raise ValueError(f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}")
        self.r = r
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.lora_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return (self.conv(input) + self.dropout(self.lora_up(self.selector(self.lora_down(input)))) * self.scale)

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Conv2d(
            in_channels=self.r,
            out_channels=self.r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector.weight.data = torch.diag(diag)

        # same device + dtype as lora_up
        self.selector.weight.data = self.selector.weight.data.to(self.lora_up.weight.device).to(self.lora_up.weight.dtype)

@torch.no_grad()
def inspect_lora(model: nn.Module):
    moved = {}

    for name, _module in model.named_modules():
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            ups = _module.lora_up.weight.data.clone()
            downs = _module.lora_down.weight.data.clone()

            wght: torch.Tensor = ups.flatten(1) @ downs.flatten(1)

            dist = wght.flatten().abs().mean().item()
            if name in moved:
                moved[name].append(dist)
            else:
                moved[name] = [dist]

    return moved

def _find_modules(
    model: nn.Module,
    ancestor_class: Optional[set[str]] = None,
    search_class: list[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[list[Type[nn.Module]]] = [
        LoraInjectedLinear,
        LoraInjectedConv2d,
    ],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (module for module in model.modules() if module.__class__.__name__ in ancestor_class)
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any([isinstance(parent, _class) for _class in exclude_children_of]):
                    continue
                # Otherwise, yield it
                yield parent, name, module

def extract_lora_ups_down(model, target_replace_module=DEFAULT_TARGET_REPLACE):

    loras = []

    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        loras.append((_child_module.lora_up, _child_module.lora_down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras

def save_lora_weight(
    model,
    path="./lora.pt",
    target_replace_module=DEFAULT_TARGET_REPLACE,
):
    weights = []
    for _up, _down in extract_lora_ups_down(
        model, target_replace_module=target_replace_module
    ):
        weights.append(_up.weight.to("cpu").to(torch.float16))
        weights.append(_down.weight.to("cpu").to(torch.float16))

    torch.save(weights, path)


def _text_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def _ti_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["ti", "pt"])

def save_all(
    unet,
    text_encoder,
    save_path,
    placeholder_token_ids=None,
    placeholder_tokens=None,
    save_lora=True,
    save_ti=True,
    target_replace_module_text=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    target_replace_module_unet=DEFAULT_TARGET_REPLACE,
    safe_form=True,
):
    if not safe_form:
        # save ti
        if save_ti:
            ti_path = _ti_lora_path(save_path)
            learned_embeds_dict = {}
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                # print(f"Current Learned Embeddings for {tok}:, id {tok_id} ", learned_embeds[:4])
                learned_embeds_dict[tok] = learned_embeds.detach().cpu()

            torch.save(learned_embeds_dict, ti_path)
            print(f"Ti saved to {ti_path}")

        # save text encoder
        if save_lora:

            save_lora_weight(unet, save_path, target_replace_module=target_replace_module_unet)
            print(f"Unet saved to {save_path}")

            save_lora_weight(text_encoder, _text_lora_path(save_path), target_replace_module=target_replace_module_text,)
            print(f"Text Encoder saved to {_text_lora_path(save_path)}")

    else:
        assert save_path.endswith(
            ".safetensors"
        ), f"Save path : {save_path} should end with .safetensors"

        loras = {}
        embeds = {}

        if save_lora:

            loras["unet"] = (unet, target_replace_module_unet)
            loras["text_encoder"] = (text_encoder, target_replace_module_text)

        if save_ti:
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                # print(f"Current Learned Embeddings for {tok}:, id {tok_id} ", learned_embeds[:4])
                embeds[tok] = learned_embeds.detach().cpu()

        save_safeloras_with_embeds(loras, embeds, save_path)

def extract_lora_as_tensor(model, target_replace_module=DEFAULT_TARGET_REPLACE, as_fp16=True):

    loras = []

    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        up, down = _child_module.realize_as_lora()
        if as_fp16:
            up = up.to(torch.float16)
            down = down.to(torch.float16)

        loras.append((up, down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras

def save_safeloras_with_embeds(
    modelmap: dict[str, tuple[nn.Module, set[str]]] = {},
    embeds: dict[str, torch.Tensor] = {},
    outpath="./lora.safetensors",
):
    """
    Saves the Lora from multiple modules in a single safetensor file.

    modelmap is a dictionary of {
        "module name": (module, target_replace_module)
    }
    """
    weights = {}
    metadata = {}

    for name, (model, target_replace_module) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))

        for i, (_up, _down) in enumerate(
            extract_lora_as_tensor(model, target_replace_module)
        ):
            rank = _down.shape[0]

            metadata[f"{name}:{i}:rank"] = str(rank)
            weights[f"{name}:{i}:up"] = _up
            weights[f"{name}:{i}:down"] = _down

    for token, tensor in embeds.items():
        metadata[token] = EMBED_FLAG
        weights[token] = tensor

    print(f"Saving weights to {outpath}")
    safe_save(weights, outpath, metadata)
    