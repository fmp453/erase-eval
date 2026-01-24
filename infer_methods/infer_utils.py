import torch
from safetensors.torch import load_file
from safetensors import safe_open


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    if not safetensors_file.endswith(".safetensors"):
        return {}

    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


def load_state_dict(file_name: str):
    if file_name.endswith(".safetensors"):
        sd = load_file(file_name)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in sd.keys():
        if isinstance(sd[key], torch.Tensor):
            sd[key] = sd[key]

    return sd, metadata
