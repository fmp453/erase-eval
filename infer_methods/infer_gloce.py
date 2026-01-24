import os
import random
from pathlib import Path

import numpy as np
import safetensors
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

from utils import Arguments
from train_methods.train_utils import get_models, get_condition, get_devices
from train_methods.train_gloce import GLoCELayerOutProp, GLoCENetworkOutProp
from train_methods.train_gloce import get_module_name_type, get_modules_list


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    if not safetensors_file.endswith(".safetensors"):
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
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


def infer_with_gloce(args: Arguments):
    args.gloce_method = args.gloce_method.split(",")
    device = get_devices(args)[0]
    model_paths = [lp for lp in args.erased_model_dir.split(",")]
    for i in range(len(model_paths)):
        concept = str(Path(model_paths[i])).split("/")[1]
        model_paths[i] = Path(f"{model_paths[i]}/{concept}")
    
    tokenizer, text_encoder, _, unet, _, _ = get_models(args)
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_version)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    text_encoder.to(device)
    text_encoder.eval()
    unet.to(device)
    unet.requires_grad_(False)
    unet.eval()

    # register org modules
    module_types = []
    module_names = []
    org_modules_all = []
    module_name_list_all = []
    
    for find_module_name in args.gloce_method:
        module_name, module_type = get_module_name_type(find_module_name)
        org_modules, module_name_list = get_modules_list(unet, text_encoder, find_module_name, module_name, module_type)
        module_names.append(module_name)
        module_types.append(module_type)
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)

    cpes, metadatas = zip(*[load_state_dict(model_path) for model_path in model_paths])

    # check if CPEs are compatible
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])
    
    network = GLoCENetworkOutProp(
        unet,
        multiplier=1.0,
        alpha=float(metadatas[0]["alpha"]),
        module=GLoCELayerOutProp,
        degen_rank=args.gloce_degen_rank,
        gate_rank=args.gloce_gate_rank,
        update_rank=args.gloce_update_rank,
        n_concepts=len(model_paths),
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names = args.gloce_method,
        last_layer=args.gloce_last_layer,
        st_step=args.gloce_start_timestep,
    ).to(device)

    for n_concept in range(len(cpes)):
        print(f"loaded concepts: {n_concept + 1}")
        for k, m in network.named_modules():
            if m.__class__.__name__ == "GLoCELayerOutProp":
                m.eta = args.gloce_eta
                        
                for k_child, m_child in m.named_children():
                    module_name = f"{k}.{k_child}"
                    if ("lora_update" in k_child) or ("lora_degen" in k_child):
                        m_child.weight.data[n_concept] = cpes[n_concept][module_name+'.weight']
                        print(f"{module_name+'.weight':100}", cpes[n_concept][module_name+'.weight'].shape)

                    elif "bias" in k_child:
                        m_child.weight.data[:,n_concept:n_concept+1,:] = cpes[n_concept][module_name+'.weight']

                    elif "selector" in k_child:
                        m_child.select_weight.weight.data[n_concept] = cpes[n_concept][module_name+'.select_weight.weight'].squeeze(0)
                        m_child.select_mean_diff.weight.data[n_concept] = cpes[n_concept][module_name+'.select_mean_diff.weight'].squeeze(0)
                        m_child.imp_center[n_concept] = cpes[n_concept][module_name+'.imp_center']
                        m_child.imp_slope[n_concept] = cpes[n_concept][module_name+'.imp_slope']

    network.to(device)
    network.eval()

    with torch.no_grad():
        prompt_embeds = get_condition([args.prompt], tokenizer, text_encoder)

        seed_everything(args.seed)
        with network:
            images = pipe(
                width=args.image_size,
                height=args.image_size,
                num_inference_steps=args.ddim_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.cuda.manual_seed(args.seed),
                num_images_per_prompt=args.num_images_per_prompt,
                prompt_embeds=prompt_embeds,
            ).images

        Path(args.images_dir, "gloce").mkdir(exist_ok=True)
        for i, image in enumerate(images):
            image.save(f"{args.images_dir}/gloce/{args.prompt.replace(' ', '-')}.png")

def main(args):
    infer_with_gloce(args)
