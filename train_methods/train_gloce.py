# GLoCE: Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation
# https://github.com/Hyun1A/GLoCE/tree/main

import os
import math
import random
import yaml
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pydantic import BaseModel, model_validator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

from utils import Arguments
from train_methods.train_utils import get_condition, get_devices

class PromptSettings(BaseModel):  # yaml
    target: str
    positive: str = None  # if None, target will be used
    unconditional: str = ""  # default is ""
    neutral: str = None  # if None, unconditional will be used

    @model_validator(mode='before')
    def fill_prompts(cls, values: dict[str, str]):
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
    
def load_prompts_from_yaml(path) -> list[PromptSettings]:
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)

    if len(prompts) == 0:
        raise ValueError("prompts file is empty")

    return [PromptSettings(**prompt) for prompt in prompts]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class ParamModule(nn.Module):
    def __init__(self, size):
        super(ParamModule, self).__init__()
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x * self.weight

    def __repr__(self):
        return f"ParameterModule(param_shape={tuple(self.weight.shape)})"

class SimpleSelectorOutProp(nn.Module):
    def __init__(
            self, 
            gate_rank: int,
            d_model,
            dropout: float=0.25,
            n_concepts: int=1,
            is_last_layer: bool=False
        ):
        super().__init__()        
        
        self.d_model = d_model
        self.gate_rank = gate_rank
        self.n_concepts = n_concepts
        self.temperature = 1.0

        self.select_weight = ParamModule((n_concepts, d_model, gate_rank))
        nn.init.kaiming_uniform_(self.select_weight.weight, a=math.sqrt(5))
        self.select_weight.weight.data = self.select_weight.weight.data / (d_model**2)    
        
        self.select_mean_diff = ParamModule((n_concepts, d_model))
        
        nn.init.kaiming_uniform_(self.select_mean_diff.weight, a=math.sqrt(5))
        self.select_mean_diff.weight.data = self.select_mean_diff.weight.data / (d_model**2)    
        
        self.register_buffer("imp_center", torch.zeros(n_concepts))
        self.register_buffer("imp_slope", torch.zeros(n_concepts))

        self.dropout = nn.Dropout(dropout)
        self.is_last_layer = is_last_layer
        
    def forward(self, x: torch.Tensor, mask=None):
        ## x: (B,T,D)
        
        x = x.unsqueeze(1)
        x_diff = x - self.select_mean_diff.weight.unsqueeze(0).unsqueeze(2) # BxNxTxD
        x_diff_norm =  x_diff / x_diff.norm(dim=3, keepdim=True)

        Vh_gate = self.select_weight.weight # (N,D,1)        
        cont = torch.einsum("nds,bntd->bnts", Vh_gate,x_diff_norm)**2
        
        select_scale = torch.sigmoid(self.imp_slope.unsqueeze(0).unsqueeze(-1)*( \
                    cont.sum(dim=-1) - self.imp_center.unsqueeze(0).unsqueeze(-1)) ) # BN

        select_scale, select_idx = select_scale.max(dim=1, keepdim=True) #

        return select_idx, select_scale

    def reset_select_cache(self):
        self.sel_prop.select_scale_cache = None
        self.sel_prop.prop_num = 0

class GLoCELayerOutProp(nn.Module):
    def __init__(
        self,
        find_name: str,
        gloce_name: str,
        gloce_org_name: str,
        org_module: nn.Module,
        multiplier: float=1.0,
        alpha: float=1,
        gate_rank: int=1,
        update_rank: int=4,
        degen_rank: int=4,
        n_concepts: int=1,
        last_layer_name: str="",
        use_update: bool=True,
        use_degen: bool=True,
        use_bias: bool=True,
        use_gate: bool=True,
        st_step: int=10,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.find_name = find_name
        self.gloce_name = gloce_name
        self.gloce_org_name = gloce_org_name
        self.eta = 1.0
        self.st_step = st_step
        self.n_step = 51

        self.use_update = use_update
        self.use_degen = use_degen
        self.use_bias = use_bias
        self.use_gate = use_gate

        if org_module.__class__.__name__ == "Linear":
 
            out_dim = org_module.out_features

            self.lora_update = ParamModule((n_concepts, out_dim, degen_rank))
            self.lora_degen = ParamModule((n_concepts, out_dim, degen_rank))

            self.bias = ParamModule((1, n_concepts, out_dim))
            self.debias = ParamModule((1, n_concepts, out_dim))


        # same as microsoft's
        nn.init.zeros_(self.lora_update.weight)    
        nn.init.zeros_(self.lora_degen.weight)    
            
        if isinstance(alpha, torch.Tensor): 
            alpha = alpha.detach().numpy()
        alpha = update_rank if alpha is None or alpha == 0 else alpha
        self.scale = alpha / update_rank
        self.register_buffer("alpha", torch.tensor(alpha))
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        is_last_layer = True if gloce_org_name == last_layer_name else False

        self.selector = SimpleSelectorOutProp(
            gate_rank=gate_rank, 
            d_model=out_dim,
            dropout=0.25, 
            n_concepts=n_concepts, 
            is_last_layer=is_last_layer
        )
        
        self.use_prompt_tuning = False    
        self.t_counter = 0

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # x.shape: (B, 77, 768)
        x = self.org_forward(x)
        self.t_counter +=1

        if not self.use_prompt_tuning:
            return x

        if self.t_counter <= self.st_step:
            return x

        select_idx, select_scale = self.selector(x) # (BxT)
                            
        debias = self.debias.weight.squeeze(0)[select_idx.squeeze(1)]
        x_debias = x-debias # BxTxD

        update_mat_sel = self.lora_update.weight[select_idx.squeeze(1)]
        degen_mat_sel = self.lora_degen.weight[select_idx.squeeze(1)]
                            
        mod_x = torch.einsum("btdh,btd->bth", update_mat_sel, x_debias)
        degen_up = torch.einsum("btdh,bth->btd", degen_mat_sel, mod_x)

        bias = self.bias.weight.squeeze(0)[select_idx.squeeze(1)]
        mod_x_bias = self.eta * (bias + degen_up) # BxNxTxD

        if not self.use_gate:
            select_scale = torch.ones_like(select_scale).to(x.device)
        
        if self.t_counter == self.n_step:
            self.t_counter = 0

        return (1-select_scale.permute(0,2,1))*x + select_scale.permute(0,2,1)*mod_x_bias 

class GLoCENetworkOutProp(nn.Module):
    TARGET_REPLACE_MODULE_TRANSFORMER = [
        "Transformer2DModel",
    ]
    TARGET_REPLACE_MODULE_CONV = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]

    GLoCE_PREFIX = "lora_gloce"  # aligning with SD webui usage
    DEFAULT_TARGET_REPLACE = TARGET_REPLACE_MODULE_TRANSFORMER

    def __init__(
        self,
        diffusion_model,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        module = GLoCELayerOutProp,
        module_kwargs = None,
        delta=1e-5,
        gate_rank=4,
        update_rank=4,
        degen_rank=4,
        n_concepts=1,
        org_modules_all=None,
        module_name_list_all=None,  
        find_module_names = None,
        last_layer = "",
        st_step = 10,
    ) -> None:
        
        super().__init__()
        
        self.n_concepts = n_concepts
        
        self.multiplier = multiplier
        self.alpha = alpha
        self.delta = delta 
        
        self.module = module
        self.module_kwargs = module_kwargs or {}
        
        self.gate_rank = gate_rank
        self.update_rank = update_rank
        self.degen_rank = degen_rank

        self.find_module_names = find_module_names
        self.org_modules_all=org_modules_all
        self.module_name_list_all=module_name_list_all
        self.last_layer = last_layer
        self.st_step = st_step


        self.gloce_layers = self.create_modules(
            GLoCENetworkOutProp.GLoCE_PREFIX,
            self.multiplier,
        )

        print(f"Create GLoCE for U-Net: {len(self.gloce_layers)} modules.")

        gloce_names = set()
        for gloce_layer in self.gloce_layers:
            assert (
                gloce_layer.gloce_name not in gloce_names
            ), f"duplicated GLoCE layer name: {gloce_layer.gloce_name}. {gloce_names}"
            gloce_names.add(gloce_layer.gloce_name)

        # Add: printing modified text encoder module
        for gloce_layer in self.gloce_layers:
            gloce_layer.apply_to()
            self.add_module(
                gloce_layer.gloce_name,
                gloce_layer,
            )
        
        del diffusion_model
        

    def load_gloce_lora_models(self, model_paths):
        for layer in self.gloce_layers:
            self.attention.encoder_layer.add_slf_attn(model_paths, layer.gloce_name)
        
    def create_modules(self, prefix: str, multiplier: float) -> list[GLoCELayerOutProp]:
        gloce_layers = []

        for find_name, org_modules in zip(self.find_module_names, self.org_modules_all):
            for module_name, org_module in org_modules.items():
                gloce_org_name = module_name
                gloce_name = prefix + "." + module_name
                gloce_name = gloce_name.replace(".", "_")
                print(f"{gloce_name=}")

                gloce_layer = self.module(
                    find_name, gloce_name, gloce_org_name, org_module, multiplier, self.alpha, \
                    gate_rank=self.gate_rank, update_rank=self.update_rank, degen_rank = self.degen_rank, \
                    n_concepts=self.n_concepts,
                    last_layer_name=self.last_layer,
                    st_step=self.st_step,
                    **self.module_kwargs
                )
                gloce_layers.append(gloce_layer)

        return gloce_layers
    
    def save_weights(self, file: str, dtype=None, metadata: dict | None = None):
        state_dict: dict[str, torch.Tensor] = self.state_dict()
        
        state_dict_save = dict()
        if dtype is not None:
            for key in state_dict.keys():
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict_save[key] = v
                
        if file.endswith(".safetensors"):
            save_file(state_dict_save, file, metadata)
        else:
            torch.save(state_dict_save, file)

    def __enter__(self):
        for gloce_layer in self.gloce_layers:
            gloce_layer.multiplier = 1.0
            gloce_layer.use_prompt_tuning = True

            
    def __exit__(self, exc_type, exc_value, tb):
        for gloce_layer in self.gloce_layers:
            gloce_layer.multiplier = 0
            gloce_layer.use_prompt_tuning = False

@torch.no_grad()
def get_registered_buffer(
    args: Arguments,
    module_name_list_all,
    org_modules_all,
    st_timestep,
    end_timestep,
    n_avail_tokens: int,
    prompts: list[str],
    embeddings: torch.Tensor,
    device: torch.device,
    register_buffer_path,
    register_buffer_fn,
    register_func,
    **kwargs
) -> dict[str, Any]:

    registered_buffer = dict()
    hooks = []

    registered_buffer, hooks = globals()[register_func](
        args,
        module_name_list_all,
        org_modules_all,
        registered_buffer,
        hooks,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        **kwargs
    )

    embs_batchsize = 1
    embs_batch: list[torch.Tensor] = []
    prompts_batch = []
    len_embs_batch = embeddings.size(0)

    Path(register_buffer_path).mkdir(exist_ok=True)

    if Path(f"{register_buffer_path}/{register_buffer_fn}").is_file():
        print(f"load precomputed registered_buffer for original models ... {register_buffer_path}/{register_buffer_fn}")
        registered_buffer = torch.load(f"{register_buffer_path}/{register_buffer_fn}", map_location=torch.device(device))

    else:
        print(f"compute registered_buffer for original models ... {register_buffer_path}/{register_buffer_fn}")
        for batch_idx in range(int(math.ceil(float(len_embs_batch)/embs_batchsize))):
            if embs_batchsize * (batch_idx + 1) <= len_embs_batch:
                embs_batch.append(embeddings[embs_batchsize * batch_idx : embs_batchsize * (batch_idx + 1)])
                prompts_batch.append(prompts[embs_batchsize * batch_idx : embs_batchsize * (batch_idx + 1)])
            
            else:
                embs_batch.append(embeddings[embs_batchsize * batch_idx:])
                prompts_batch.append(prompts[embs_batchsize * batch_idx:])

        for step, (embs, prompts) in enumerate(zip(embs_batch, prompts_batch)):
            if step % 10 == 0:
                print(f"{step}/{len(embs_batch)}")

            for seed in range(args.num_images_per_prompt):
                for find_module_name, module_name_list in zip(args.gloce_method, module_name_list_all):
                    for n in module_name_list:
                        if "seed" in registered_buffer[find_module_name][n].keys():
                            registered_buffer[find_module_name][n]["seed"] = seed

                if len(embs.size()) == 4:
                    B, C, T, D = embs.size()
                    embs = embs.reshape(B * C, T, D)

                for find_module_name, module_name_list in zip(args.gloce_method, module_name_list_all):
                    for n in module_name_list:
                        registered_buffer[find_module_name][n]["t"] = 0

        for find_module_name, module_name_list in zip(args.gloce_method, module_name_list_all):
            for n in module_name_list:
                registered_buffer[find_module_name][n]["t"] = 0

        if register_func != "register_norm_buffer_save_activation_sel":
            torch.save(registered_buffer, f"{register_buffer_path}/{register_buffer_fn}")

    for hook in hooks:
        hook.remove()

    return registered_buffer

@torch.no_grad()
def prepare_text_embedding_token(
    args: Arguments,
    prompts_target: list[PromptSettings],
    prompts_surround: list[PromptSettings],
    prompts_update: list[PromptSettings],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    emb_cache_path,
    n_avail_tokens=8,
    n_anchor_concepts=5
) -> dict[str, torch.Tensor]:
    prompt_scripts_path = f"configs/train_{args.gloce_replace_word}/prompt_templates.csv"

    prompt_scripts_df = pd.read_csv(prompt_scripts_path)
    prompt_scripts_list: list[str] = prompt_scripts_df['prompt'].to_list()
    replace_word = args.gloce_replace_word

    if replace_word == "artist":
        prmpt_temp_sel_base = f"An image in the style of {replace_word}" 
    elif replace_word == "celeb":
        prmpt_temp_sel_base = f"A face of {replace_word}"
    elif replace_word == "explicit":
        prmpt_temp_sel_base = replace_word

    prompt_scripts_list.append(prmpt_temp_sel_base)
    if args.gloce_use_emb_cache and Path(f"{emb_cache_path}/{args.gloce_emb_cache_fn}").is_file():
        print("load pre-computed text emb cache...")
        emb_cache = torch.load(f"{emb_cache_path}/{args.gloce_emb_cache_fn}", map_location=torch.device(text_encoder.device))
        
    else:
        print("compute text emb cache...")

        # compute target and update concept embeddings
        simWords_target = [prompt.target for prompt in prompts_target]
        prmpt_sel_base_target = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_target] 
        embeddings_target_sel_base = get_condition(prmpt_sel_base_target, tokenizer, text_encoder)

        simWords_update = [prompt.target for prompt in prompts_update]
        prmpt_sel_base_update = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_update] 
        embeddings_update_sel_base = get_condition(prmpt_sel_base_update, tokenizer, text_encoder)
                
        # Compute and select anchor and surrogate concept embeddings
        simWords_surround = [prompt.target for prompt in prompts_surround]
        for simW_erase in simWords_target:
            simWords_surround = [item.lower() for item in simWords_surround if simW_erase not in item.lower()]
            
        prmpt_sel_base_surround = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_surround] 

        len_surround = len(prmpt_sel_base_surround)
        text_encode_batch = 100
        simWords_surround_batch = [
            prmpt_sel_base_surround[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_surround) / text_encode_batch)))
        ]

        embeddings_surround_sel_base = []
        for simW_batch in simWords_surround_batch:
            emb_surround = get_condition(simW_batch, tokenizer, text_encoder)
            embeddings_surround_sel_base.append(emb_surround)
        embeddings_surround_sel_base = torch.cat(embeddings_surround_sel_base, dim=0)
        # Compute similarity
        embeddings_target_norm = embeddings_target_sel_base / embeddings_target_sel_base.norm(2, dim=-1, keepdim=True)
        embeddings_surround_norm = embeddings_surround_sel_base / embeddings_surround_sel_base.norm(2, dim=-1, keepdim=True)

        similarity = torch.einsum(
            "ijk,njk->inj", 
            embeddings_target_norm[:, 1:(1 + n_avail_tokens), :],
            embeddings_surround_norm[:, 1:(1 + n_avail_tokens), :]
        ).mean(dim=2)

        similarity = similarity.mean(dim=0)
        _, ind_sorted = similarity.sort()
        ind_sorted_list = ind_sorted.cpu().numpy().tolist()
        
        simWords_anchor = [simWords_surround[sim_idx] for sim_idx in ind_sorted_list[-n_anchor_concepts:]]
        embeddings_anchor_sel_base = embeddings_surround_sel_base[ind_sorted_list[-n_anchor_concepts:]]

        if replace_word == "celeb": # or args.opposite_for_map:        
            simWords_surrogate = [simWords_surround[sim_idx] for sim_idx in ind_sorted_list[:n_anchor_concepts]]
            embeddings_surrogate_sel_base = embeddings_surround_sel_base[ind_sorted_list[:n_anchor_concepts]]  
            
        else:
            simWords_surrogate = [prompts_target[0].neutral]
            prmpt_sel_base_surrogate = [prmpt_temp_sel_base.replace(replace_word, word) for word in simWords_surrogate] 
            embeddings_surrogate_sel_base = get_condition(prmpt_sel_base_surrogate, tokenizer, text_encoder)

        # Prepare for surrogate token cache
        print("compute emb cache...")
        
        embeddings_surrogate_cache = []
        prmpt_scripts_sur = []

        for simWord in simWords_surrogate:
            for prompt_script in prompt_scripts_list:
                pr_in_script_sur = prompt_script.replace(replace_word, simWord)
                pr_in_script_sur = pr_in_script_sur.replace(replace_word.lower(), simWord)
                prmpt_scripts_sur.append(pr_in_script_sur)
                
        len_surrogate = len(prmpt_scripts_sur)
        text_encode_batch = 100
        prmpt_scripts_sur_batch = [
            prmpt_scripts_sur[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_surrogate) / text_encode_batch)))
        ]

        for prmpt_batch in prmpt_scripts_sur_batch:
            embeddings_sur = get_condition(prmpt_batch, tokenizer, text_encoder)
            embeddings_surrogate_cache.append(embeddings_sur)

        embeddings_surrogate_cache = torch.cat(embeddings_surrogate_cache, dim=0)        

        # Prepare for target token cache
        embeddings_target_cache = []
        prmpt_scripts_tar = []
        for simWord in simWords_target:
            for prompt_script in prompt_scripts_list:
                pr_in_script_tar = prompt_script.replace(replace_word, simWord)
                pr_in_script_tar = pr_in_script_tar.replace(replace_word.lower(), simWord)
                prmpt_scripts_tar.append(pr_in_script_tar)

        len_target = len(prmpt_scripts_tar)
        text_encode_batch = 100
        prmpt_scripts_tar_batch = [
            prmpt_scripts_tar[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_target) / text_encode_batch)))
        ]

        for prmpt_batch in prmpt_scripts_tar_batch:
            embeddings_tar = get_condition(prmpt_batch, tokenizer, text_encoder)
            embeddings_target_cache.append(embeddings_tar)

        embeddings_target_cache = torch.cat(embeddings_target_cache, dim=0)

        # Prepare for anchor token cache
        embeddings_anchor_cache = []
        prmpt_scripts_anc = []

        for simWord in simWords_anchor:
            for prompt_script in prompt_scripts_list:
                pr_in_script_anc = prompt_script.replace(replace_word, simWord)
                pr_in_script_anc = pr_in_script_anc.replace(replace_word.lower(), simWord)
                prmpt_scripts_anc.append(pr_in_script_anc)

        len_anchor = len(prmpt_scripts_anc)
        text_encode_batch = 100
        prmpt_scripts_anc_batch = [
            prmpt_scripts_anc[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_anchor) / text_encode_batch)))
        ]
        for prmpt_batch in prmpt_scripts_anc_batch:
            embeddings_anc = get_condition(prmpt_batch, tokenizer, text_encoder)
            embeddings_anchor_cache.append(embeddings_anc)

        embeddings_anchor_cache = torch.cat(embeddings_anchor_cache, dim=0)

        # Prepare for update token cache
        embeddings_update_cache = []
        prmpt_scripts_upd = []

        for simWord in simWords_update:
            for prompt_script in prompt_scripts_list:
                pr_in_script_upd = prompt_script.replace(replace_word, simWord)
                prmpt_scripts_upd.append(pr_in_script_upd)

        len_update = len(prmpt_scripts_upd)
        text_encode_batch = 100
        prmpt_scripts_upd_batch = [
            prmpt_scripts_upd[text_encode_batch * batch_idx:text_encode_batch * (batch_idx + 1)]
            for batch_idx in range(int(math.ceil(float(len_update) / text_encode_batch)))
        ]

        for prmpt_batch in prmpt_scripts_upd_batch:
            embeddings_upd = get_condition(prmpt_batch, tokenizer, text_encoder)
            embeddings_update_cache.append(embeddings_upd)

        embeddings_update_cache = torch.cat(embeddings_update_cache, dim=0)
                                              
        # Save emb cache
        emb_cache = {
            "embeddings_surrogate_cache": embeddings_surrogate_cache,
            "embeddings_target_cache": embeddings_target_cache,
            "embeddings_anchor_cache": embeddings_anchor_cache,
            "embeddings_update_cache": embeddings_update_cache,
            "embeddings_surrogate_sel_base": embeddings_surrogate_sel_base,
            "embeddings_target_sel_base": embeddings_target_sel_base,
            "embeddings_anchor_sel_base": embeddings_anchor_sel_base,
            "embeddings_update_sel_base": embeddings_update_sel_base,
            "prmpt_scripts_sur": prmpt_scripts_sur,
            "prmpt_scripts_tar": prmpt_scripts_tar,
            "prmpt_scripts_anc": prmpt_scripts_anc,
            "prmpt_scripts_upd": prmpt_scripts_upd,
        }

        Path(emb_cache_path).mkdir(exist_ok=True)
        torch.save(emb_cache, f"{emb_cache_path}/{args.gloce_emb_cache_fn}")

    return emb_cache

def get_module_name_type(find_module_name: str) -> tuple[str, str]:
    match find_module_name:
        case  "unet_ca":
            return "Linear", "attn2"
        case "unet_ca_kv":
            return "Linear", "attn2"
        case "unet_ca_v":
            return "Linear", "attn2"
        case "unet_ca_out":
            return "Linear", "attn2"
        case "unet_sa_out":
            return "Linear", "attn1"
        case "unet_sa":
            return "Linear", "attn1"
        case "unet_conv2d":
            return "Conv2d", "conv2d"
        case "unet_misc":
            return "Linear", "misc"
        case "te_attn":
            return "Linear", "self_attn"
        case _:
            return "Linear", "mlp.fc"

def get_modules_list(
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    find_module_name: str,
    module_name: str,
    module_type: str
) -> tuple[dict[str, nn.Module], list[str]]:
    org_modules = dict()
    module_name_list = []
    return_ok = False

    match find_module_name:
        case "unet_ca_out":
            return_ok = True
            for n, m in unet.named_modules():
                if m.__class__.__name__ == module_type:
                    if f"{module_name}.to_out" in n:
                        module_name_list.append(n)
                        org_modules[n] = m

        case "unet_ca_kv":
            return_ok = True
            for n, m in unet.named_modules():
                if m.__class__.__name__ == module_type:
                    if f"{module_name}.to_k" in n or f"{module_name}.to_v" in n:
                        module_name_list.append(n)
                        org_modules[n] = m

        case "unet_ca_v":
            return_ok = True
            for n, m in unet.named_modules():
                if m.__class__.__name__ == module_type:
                    if f"{module_name}.to_v" in n:
                        module_name_list.append(n)
                        org_modules[n] = m

        case "unet_sa_out":
            return_ok = True
            for n, m in unet.named_modules():
                if m.__class__.__name__ == module_type:
                    if f"{module_name}.to_out" in n:
                        module_name_list.append(n)
                        org_modules[n] = m
        case _:
            pass

    if "unet" in find_module_name and not return_ok:
        for n, m in unet.named_modules():
            if m.__class__.__name__ == module_type:
                if module_name == "misc":
                    if ("attn1" not in n) and ("attn2" not in n):
                        module_name_list.append(n)
                        org_modules[n] = m
                elif module_name in ["attn1", "attn2"]: 
                    if module_name in n:
                        module_name_list.append(n)
                        org_modules[n] = m
                else:
                    module_name_list.append(n)
                    org_modules[n] = m
    else:
        for n, m in text_encoder.named_modules():
            if m.__class__.__name__ == module_type:       
                if module_name in n:
                    module_name_list.append(n)
                    org_modules[n] = m

    return org_modules, module_name_list

def load_model_sv_cache(find_module_name, param_cache_path, device, org_modules: dict[str, nn.Module]):
    if Path(param_cache_path, f"vh_cache_dict_{find_module_name}.pt").is_file:
        param_vh_cache_dict = torch.load(f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt", map_location=torch.device(device)) 
        param_s_cache_dict = torch.load(f"{param_cache_path}/s_cache_dict_{find_module_name}.pt", map_location=torch.device(device))
    else:
        param_vh_cache_dict = dict()
        param_s_cache_dict = dict()

        for k, m in org_modules.items():
            if m.__class__.__name__ == "Linear":
                _, S, Vh = torch.linalg.svd(m.weight, full_matrices=False) 
                param_vh_cache_dict[k] = Vh.detach().cpu()
                param_s_cache_dict[k] = S.detach().cpu()        

            elif m.__class__.__name__ == "Conv2d":
                module_weight_flatten = m.weight.view(m.weight.size(0), -1)
                _, S, Vh = torch.linalg.svd(module_weight_flatten, full_matrices=False) 
                param_vh_cache_dict[k] = Vh.detach().cpu()
                param_s_cache_dict[k] = S.detach().cpu()                

        Path(param_cache_path).mkdir(exist_ok=True)
        torch.save(param_vh_cache_dict, f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt")
        torch.save(param_s_cache_dict, f"{param_cache_path}/s_cache_dict_{find_module_name}.pt")

    return param_vh_cache_dict, param_s_cache_dict

def train(args: Arguments):
    # preprocess before erasing
    args.gloce_method = args.gloce_method.split(",")
    seed_everything(args.seed)
    prompts_target = load_prompts_from_yaml(args.gloce_prompts_file_targets)
    prompts_anchor = load_prompts_from_yaml(args.gloce_prompts_file_anchors)
    prompts_update = load_prompts_from_yaml(args.gloce_prompts_file_updates)
    
    # network config
    network_rank: int = args.gloce_update_rank if args.gloce_update_rank != -1 else 1
    network_alpha: float = args.gloce_alpha
    network_delta: float = args.gloce_delta
    network_init_size: int = args.gloce_gate_rank

    n_target_concepts = args.gloce_n_target_concepts
    tar_concept_idx = args.gloce_tar_concept_idx
    n_anchor_concepts = args.gloce_n_anchor_concepts
    st_timestep = args.gloce_start_timestep
    end_timestep = args.gloce_end_timestep
    n_avail_tokens = args.gloce_n_tokens
    update_rank = args.gloce_update_rank
    gate_rank = args.gloce_gate_rank
    degen_rank = args.gloce_degen_rank

    prompts_target = prompts_target[tar_concept_idx:tar_concept_idx + n_target_concepts]

    targets = [prompt.target for prompt in prompts_target]
    anchors = [prompt.target for prompt in prompts_anchor]
    surrogate = [prompts_target[0].neutral]
    updates = [prompt.target for prompt in prompts_update]

    save_path = f"{args.save_dir}/{targets[0].replace(' ', '_')}"
    emb_cache_path = f"{args.gloce_emb_cache_path}/{targets[0].replace(' ', '_')}"
    register_buffer_path = f"{args.gloce_buffer_path}/{targets[0].replace(' ', '_')}"
        
    model_metadata = {
        "prompts": ",".join([prompt.target for prompt in prompts_target]),
        "rank": str(network_rank),
        "alpha": str(network_alpha),
    }
    
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    device = get_devices(args)[0]

    text_encoder.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    unet.to(device)
    unet.requires_grad_(False)
    unet.eval()

    # register org modules
    module_types = []
    module_names = []
    org_modules_all = []
    module_name_list_all = []
    param_vh_cache_dict_all = []
    param_s_cache_dict_all = []
    
    for find_module_name in args.gloce_method:
        module_name, module_type = get_module_name_type(find_module_name)            
        org_modules, module_name_list = get_modules_list(unet, text_encoder, find_module_name, module_name, module_type)
        param_vh_cache_dict, param_s_cache_dict = load_model_sv_cache(find_module_name, args.gloce_param_cache_path, device, org_modules)

        module_names.append(module_name)
        module_types.append(module_type)
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)
        param_vh_cache_dict_all.append(param_vh_cache_dict)
        param_s_cache_dict_all.append(param_s_cache_dict)

    network = GLoCENetworkOutProp(
        unet,
        multiplier=1.0,
        alpha=network_alpha,
        module=GLoCELayerOutProp,
        delta=network_delta,
        gate_rank=gate_rank,
        update_rank=update_rank,
        degen_rank=degen_rank,
        n_concepts=1,
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names=args.gloce_method,
        last_layer=args.gloce_last_layer,
    ).to(device)

    print(f"gate rank of netowrk: {network_init_size}")

    network.eval()
    network_modules = dict()
    for name, module in network.named_modules():
        if "GLoCELayer" in module.__class__.__name__:
            network_modules[name] = module

    unet_modules = dict()
    for name, module in unet.named_modules():
        name = "_".join(name.split("."))
        name = "lora_unet_" + name

        for network_name in network_modules.keys():
            if name == network_name:
                unet_modules[name] = module   

    # Prepare for text embedding token
    emb_cache = prepare_text_embedding_token(
        args,
        prompts_target,
        prompts_anchor,
        prompts_update,
        tokenizer,
        text_encoder,
        emb_cache_path,
        n_avail_tokens=n_avail_tokens,
        n_anchor_concepts=n_anchor_concepts
    )

    embeddings_surrogate_sel_base = emb_cache["embeddings_surrogate_sel_base"]
    embeddings_target_sel_base = emb_cache["embeddings_target_sel_base"]
    embeddings_anchor_sel_base = emb_cache["embeddings_anchor_sel_base"]
    embeddings_update_sel_base = emb_cache["embeddings_update_sel_base"]

    embeddings_surrogate_cache = emb_cache["embeddings_surrogate_cache"]
    embeddings_target_cache = emb_cache["embeddings_target_cache"]
    embeddings_anchor_cache = emb_cache["embeddings_anchor_cache"]
    embeddings_update_cache = emb_cache["embeddings_update_cache"]

    prmpt_scripts_sur = emb_cache["prmpt_scripts_sur"]
    prmpt_scripts_tar = emb_cache["prmpt_scripts_tar"]
    prmpt_scripts_anc = emb_cache["prmpt_scripts_anc"]
    prmpt_scripts_upd = emb_cache["prmpt_scripts_upd"]    
    
    use_prompt = args.gloce_method in ["unet_ca_v", "unet_ca_out"]
    if args.gloce_replace_word == "artist" and use_prompt:
        embeddings_surrogate_sel_base = embeddings_surrogate_cache
        embeddings_target_sel_base = embeddings_target_cache
        embeddings_anchor_sel_base = embeddings_anchor_cache
        embeddings_update_sel_base = embeddings_update_cache

        surrogate = prmpt_scripts_sur
        targets = prmpt_scripts_tar
        anchors = prmpt_scripts_anc
        updates = prmpt_scripts_upd

    # Compute register buffer for surrogate concept for erasing 
    buffer_sel_basis_surrogate = get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        surrogate,
        embeddings_surrogate_sel_base,
        device,
        register_buffer_path,
        register_buffer_fn="stacked_surrogate.pt",
        register_func="register_sum_buffer_avg_spatial"
    )

    # Compute principal components for surrogate concept
    Vh_sur_dict = dict()
    surrogate_mean_dict = dict()
    for find_name in args.gloce_method:
        Vh_sur_dict[find_name] = dict()
        surrogate_mean_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:        
        n_forward = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_sum_per_forward = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]['n_sum_per_forward']
        n_sum = n_forward*n_sum_per_forward

        stacked_buffer_surrogate = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]['data'] / n_sum
        stacked_buffer_surrogate_mean = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]["data_mean"] / n_sum
        stacked_buffer_surrogate_cov = stacked_buffer_surrogate - stacked_buffer_surrogate_mean.T @ stacked_buffer_surrogate_mean
        
        _, _, Vh_sur = torch.linalg.svd(stacked_buffer_surrogate_cov, full_matrices=False)
        Vh_sur_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_sur
        surrogate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_surrogate_mean

        gloce_module.lora_degen.weight.data = Vh_sur[:degen_rank].T.contiguous()
        gloce_module.bias.weight.data = stacked_buffer_surrogate_mean.unsqueeze(0).clone().contiguous()  

    # Compute registder buffer for target concept for erasing
    buffer_sel_basis_target = get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        targets,
        embeddings_target_sel_base,
        device,
        register_buffer_path,
        register_buffer_fn="stacked_target.pt",
        register_func="register_sum_buffer_avg_spatial"
    )

    # Compute principal components for target concept
    target_mean_dict: dict[str, dict[str, nn.Module]] = dict()
    target_cov_dict = dict()
    Vh_tar_dict = dict()
    for find_name in args.gloce_method:
        target_mean_dict[find_name] = dict()
        Vh_tar_dict[find_name] = dict()
        target_cov_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:        
        n_forward = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_sum_per_forward = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['n_sum_per_forward']
        n_sum = n_forward*n_sum_per_forward

        stacked_buffer_target_mean = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['data_mean'] / n_sum
        stacked_buffer_target = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['data'] / n_sum
        stacked_buffer_target_cov = stacked_buffer_target - stacked_buffer_target_mean.T @ stacked_buffer_target_mean

        _, _, Vh_tar = torch.linalg.svd(stacked_buffer_target_cov, full_matrices=False)
        Vh_tar_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_tar[:update_rank]
        target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_target_mean
        target_cov_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_target_cov
        
    for gloce_module in network.gloce_layers:   
        Vh_upd = Vh_tar_dict[gloce_module.find_name][gloce_module.gloce_org_name][:update_rank]
        target_mean = target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name].squeeze(0)
        dim_emb = Vh_upd.size(1)

        Vh_sur = Vh_sur_dict[gloce_module.find_name][gloce_module.gloce_org_name][:degen_rank] # hxd        
        gloce_module.lora_update.weight.data = (Vh_sur @ (torch.eye(dim_emb).to(device)- Vh_upd.T @ Vh_upd)).T.contiguous()
        gloce_module.debias.weight.data = target_mean.unsqueeze(0).unsqueeze(0).clone().contiguous()  
    
    # Compute register buffer for surrogate for gate
    buffer_sel_basis_gate = get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        updates,
        embeddings_update_sel_base,
        device,
        register_buffer_path,
        register_buffer_fn="stacked_gate.pt",
        register_func="register_sum_buffer_avg_spatial"
    )
    
    # Compute principal components of surrogate for gate
    Vh_gate_dict: dict[str, dict[str, dict[str, nn.Module]]] = dict()
    gate_mean_dict = dict()
    rel_gate_dict = dict()
    for find_name in args.gloce_method:
        Vh_gate_dict[find_name] = dict()
        gate_mean_dict[find_name] = dict()
        rel_gate_dict[find_name] = dict()

    for gloce_module in network.gloce_layers:        
        n_forward = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_sum_per_forward = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]['n_sum_per_forward']
        n_sum = n_forward*n_sum_per_forward

        stacked_buffer_gate_mean = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]["data_mean"] / n_sum        
        stacked_buffer_rel_mean = target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] - stacked_buffer_gate_mean
        stacked_buffer_rel_cov = target_cov_dict[gloce_module.find_name][gloce_module.gloce_org_name] + stacked_buffer_rel_mean.T @ stacked_buffer_rel_mean
                
        _, _, Vh_gate = torch.linalg.svd(stacked_buffer_rel_cov, full_matrices=False)
        rel_gate_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_gate[:gate_rank]
        gate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_gate_mean

    # Compute registder buffer for discriminative basis for erasing
    buffer_norm_basis_target = get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        targets,
        embeddings_target_sel_base,
        device,
        register_buffer_path,
        register_buffer_fn="norm_target.pt",
        register_func="register_norm_buffer_avg_spatial",
        rel_gate_dict=rel_gate_dict,
        target_mean_dict=target_mean_dict,
        gate_mean_dict=gate_mean_dict
    )

    buffer_norm_basis_anchor = get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        anchors,
        embeddings_anchor_sel_base,
        device,
        register_buffer_path,
        register_buffer_fn="norm_anchor.pt",
        register_func="register_norm_buffer_avg_spatial",
        rel_gate_dict=rel_gate_dict,
        target_mean_dict=target_mean_dict,
        gate_mean_dict=gate_mean_dict
    )

    # Compute discriminative basis for erasing
    for gloce_module in network.gloce_layers:
        importance_tgt_stack = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['data_stack']
        importance_anc_stack = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]['data_stack']
        importance_tgt_stack = torch.cat([imp.unsqueeze(0) for imp in importance_tgt_stack], dim=0)
        importance_anc_stack = torch.cat([imp.unsqueeze(0) for imp in importance_anc_stack], dim=0)

        # Determine parameters in logistic function
        tol1 = args.gloce_thresh
        x_center = importance_anc_stack.mean() + tol1 * importance_anc_stack.std()     
        tol2 = 0.001 * tol1

        c_right = torch.tensor([0.99]).to(device)
        C_right = torch.log(1 / (1 / c_right - 1))

        imp_center = x_center
        imp_slope = C_right / tol2

        print(f"{importance_anc_stack.max().item():10.5f}, {imp_center.item():10.5f}, {importance_tgt_stack.min().item():10.5f}, {importance_tgt_stack.max().item():10.5f}")
        
        Vh_gate = rel_gate_dict[gloce_module.find_name][gloce_module.gloce_org_name]
        gate_mean = gate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name]

        # NxD
        gloce_module.selector.select_weight.weight.data = Vh_gate.T.unsqueeze(0).clone().contiguous()
        gloce_module.selector.select_mean_diff.weight.data = gate_mean.clone().contiguous()

        gloce_module.selector.imp_center = imp_center
        gloce_module.selector.imp_slope = imp_slope

    print("saving gloce parameters...")
    save_path = Path(f"{save_path}")            
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(save_path / f"ckpt.safetensors", metadata=model_metadata)
    print("Done.")

def main(args: Arguments):
    # train(args)
    raise NotImplementedError("under construction of train function")
