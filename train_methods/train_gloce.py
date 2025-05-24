# GLoCE: Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation
# https://github.com/Hyun1A/GLoCE/tree/main

import argparse
import gc
import os
import math
import random
from pathlib import Path
from typing import Optional, Literal, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pydantic import BaseModel, model_validator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from safetensors.torch import save_file

from train_methods.train_utils import get_condition

ACTION_TYPES = Literal["erase", "erase_with_la"]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
DEVICE_CUDA = torch.device("cuda")

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
            # applying the default values of PromptSetting
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

class InferenceConfig(BaseModel):
    use_wandb: bool = False
    negative_prompt: str = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seeds: list[int] = None    
    # precision: str = "float32"

class TrainConfig(BaseModel):    
    precision: str = "float32"
    noise_scheduler: Literal["ddim", "ddpm", "lms", "euler_a"] = "ddim"

    iterations: int = 3000
    batch_size: int = 1

    lr: float = 1e-4
    unet_lr: float = 1e-4
    text_encoder_lr: float = 5e-5

    optimizer_type: str = "AdamW8bit"
    optimizer_args: list[str] = None

    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 500
    lr_scheduler_num_cycles: int = 3
    lr_scheduler_power: float = 1.0
    lr_scheduler_args: str = ""

    max_grad_norm: float = 0.0

    max_denoising_steps: int = 30
    importance_path: str="./"
    portion: float=1.0
    push_strength: float=1.0
    norm_strength: float=1.0
    
    pal: float=0.01
    value_weight: float=0.1
    swap_iteration: int = 1500
    erase_scale: float = 1.
    
    #########################################
    ########### For adv memory ##############
    num_stages: int = 10
    iterations_adv: int = 1000
    lr_scheduler_adv: str = "cosine_with_restarts"
    lr_warmup_steps_adv: int = 500
    lr_scheduler_num_cycles_adv: int = 3
    lr_scheduler_power_adv: float = 1.0
    lr_scheduler_args_adv: str = ""    
    lr_adv: float = 1e-4
    adv_coef: float = 0.1
    num_add_prompts: int = 10
    num_multi_concepts: int = 1
    train_seed: int = 0
    factor_init_iter: int = 1
    factor_init_lr: float = 1
    factor_init_lr_cycle: int = 1
    do_adv_learn: bool = False
    ########### For adv memory ##############
    #########################################
    
    st_prompt_idx: int = 0
    end_prompt_idx: int = 100000000
    resume_stage: int = 0
    skip_learned: bool = False
    noise_scale: float = 0.001
    mixup: bool = True
   
class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    per_steps: int = 500
    precision: str = "float32"
    stage_interval: int = 1

class LoggingConfig(BaseModel):
    use_wandb: bool = False
    project_name: str = "proposed_method"
    run_name: str = None
    verbose: bool = False
    
    interval: int = 50
    prompts: list[str] = []
    negative_prompt: str = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    # negative_prompt: str = ""    
    anchor_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = None
    generate_num: int = 1
    eval_num: int = 10
    stage_interval: int = 1
    gen_init_img: bool = False

class PretrainedModelConfig(BaseModel):
    name_or_path: str
    safetensor: Optional[list[str] | str] = None
    v2: bool = False
    v_pred: bool = False
    clip_skip: Optional[int] = None


class NetworkConfig(BaseModel):
    rank: int = 1
    continual_rank: int = 4
    alpha: float = 1.0
    delta: float = 1e-5
    num_embeddings: int = 3
    hidden_size: int = 128
    init_size: int = 16


class RootConfig(BaseModel):
    prompts_file: Optional[str] = None
    scripts_file: Optional[str] = None
    replace_word: Optional[str] = None
    prompts_file_target: Optional[str] = None   
    prompts_file_anchor: Optional[str] = None   
    prompts_file_update: Optional[str] = None

    pretrained_model: PretrainedModelConfig
    network: Optional[NetworkConfig] = None
    train: Optional[TrainConfig] = None
    save: Optional[SaveConfig] = None
    logging: Optional[LoggingConfig] = None
    inference: Optional[InferenceConfig] = None

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
        # text_encoder: CLIPTextModel,
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

        ############### Add: printing modified text encoder module ################
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
    
    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        
        state_dict: dict[str, torch.Tensor] = self.state_dict()
        
        state_dict_save = dict()
        if dtype is not None:
            for key in state_dict.keys():
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict_save[key] = v
                
        if os.path.splitext(file)[1] == ".safetensors":
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
    args,
    module_name_list_all,
    org_modules_all,
    st_timestep,
    end_timestep,
    n_avail_tokens: int,
    prompts: list[str],
    embeddings: torch.Tensor,
    embedding_uncond: torch.Tensor,
    pipe: StableDiffusionPipeline,
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

    os.makedirs(register_buffer_path, exist_ok=True)

    if os.path.isfile(f"{register_buffer_path}/{register_buffer_fn}"):
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

            for seed in range(args.n_generation_per_concept):
                for find_module_name, module_name_list, org_modules in zip(args.find_module_name, module_name_list_all, org_modules_all):
                    for n in module_name_list:
                        if "seed" in registered_buffer[find_module_name][n].keys():
                            registered_buffer[find_module_name][n]["seed"] = seed

                if len(embs.size())==4:
                    B,C,T,D = embs.size()
                    embs = embs.reshape(B*C,T,D)

                if "save_path" in kwargs.keys():
                    save_path = f"{kwargs['save_path']}/seed_{seed}"
                    os.makedirs(f"{save_path}", exist_ok=True)
                    save_path = f"{save_path}/image.png"

                else:
                    save_path = "./test2.png"

                embedding2img(
                    embs,
                    "",
                    pipe,
                    seed=seed,
                    uncond_embeddings=embedding_uncond,
                    end_timestep=end_timestep,
                    save_path=save_path
                )

                for find_module_name, module_name_list, org_modules in zip(args.find_module_name, module_name_list_all, org_modules_all):
                    for n in module_name_list:
                        registered_buffer[find_module_name][n]["t"] = 0

        for find_module_name, module_name_list, org_modules in zip(args.find_module_name, module_name_list_all, org_modules_all):
            for n in module_name_list:
                registered_buffer[find_module_name][n]["t"] = 0

        if register_func != "register_norm_buffer_save_activation_sel":
            torch.save(registered_buffer, f"{register_buffer_path}/{register_buffer_fn}")

    for hook in hooks:
        hook.remove()

    return registered_buffer


@torch.no_grad()
def prepare_text_embedding_token(
    args,
    config,
    prompts_target: list[PromptSettings],
    prompts_surround: list[PromptSettings],
    prompts_update: list[PromptSettings],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    emb_cache_path,
    emb_cache_fn,
    n_avail_tokens=8,
    n_anchor_concepts=5
):
    prompt_scripts_path = config.scripts_file

    prompt_scripts_df = pd.read_csv(prompt_scripts_path)
    prompt_scripts_list: list[str] = prompt_scripts_df['prompt'].to_list()
    replace_word = config.replace_word

    if replace_word == "artist":
        prmpt_temp_sel_base = f"An image in the style of {replace_word}" 
    elif replace_word == "celeb":
        prmpt_temp_sel_base = f"A face of {replace_word}"
    elif replace_word == "explicit":
        prmpt_temp_sel_base = replace_word

    prompt_scripts_list.append(prmpt_temp_sel_base)
    if args.use_emb_cache and os.path.isfile(f"{emb_cache_path}/{emb_cache_fn}"):
        print("load pre-computed text emb cache...")
        emb_cache = torch.load(f"{emb_cache_path}/{emb_cache_fn}", map_location=torch.device(text_encoder.device))
        
    else:
        print("compute text emb cache...")

        ##################### compute concept embeddings ####################
        
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
        val_sorted, ind_sorted = similarity.sort()
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
        
        ##################### compute concept embeddings ####################
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

        os.makedirs(emb_cache_path, exist_ok=True)
        torch.save(emb_cache, f"{emb_cache_path}/{emb_cache_fn}")

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
                    if (module_name + ".to_out" in n):
                        module_name_list.append(n)
                        org_modules[n] = m

        case "unet_ca_kv":
            return_ok = True
            for n, m in unet.named_modules():
                if m.__class__.__name__ == module_type:
                    if (module_name + ".to_k" in n) or (module_name + ".to_v" in n):
                        module_name_list.append(n)
                        org_modules[n] = m

        case "unet_ca_v":
            return_ok = True
            for n, m in unet.named_modules():
                if m.__class__.__name__ == module_type:
                    if (module_name + ".to_v" in n):
                        module_name_list.append(n)
                        org_modules[n] = m
                        

        case "unet_sa_out":
            return_ok = True
            for n, m in unet.named_modules():
                if m.__class__.__name__ == module_type:
                    if (module_name + ".to_out" in n):
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
    
    if os.path.isfile(f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt"):
        print("load precomputed svd for original models ....")

        param_vh_cache_dict = torch.load(f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt", map_location=torch.device(device)) 
        param_s_cache_dict = torch.load(f"{param_cache_path}/s_cache_dict_{find_module_name}.pt", map_location=torch.device(device))

    else:
        print("compute svd for original models ....")

        param_vh_cache_dict = dict()
        param_s_cache_dict = dict()

        for k, m in org_modules.items():
            if m.__class__.__name__ == "Linear":
                U,S,Vh = torch.linalg.svd(m.weight, full_matrices=False) 
                param_vh_cache_dict[k] = Vh.detach().cpu()
                param_s_cache_dict[k] = S.detach().cpu()        

            elif m.__class__.__name__ == "Conv2d":
                module_weight_flatten = m.weight.view(m.weight.size(0), -1)

                U, S, Vh = torch.linalg.svd(module_weight_flatten, full_matrices=False) 
                param_vh_cache_dict[k] = Vh.detach().cpu()
                param_s_cache_dict[k] = S.detach().cpu()                

        os.makedirs(param_cache_path, exist_ok=True)
        torch.save(param_vh_cache_dict, f"{param_cache_path}/vh_cache_dict_{find_module_name}.pt")
        torch.save(param_s_cache_dict, f"{param_cache_path}/s_cache_dict_{find_module_name}.pt")

    return param_vh_cache_dict, param_s_cache_dict

def train(
    config: RootConfig,
    prompts_target: list[PromptSettings],
    prompts_anchor: list[PromptSettings],
    prompts_update: list[PromptSettings],
    args,
):

    ################### Setup for GLoCE #####################

    n_target_concepts = args.n_target_concepts
    tar_concept_idx = args.tar_concept_idx
    n_anchor_concepts = args.n_anchor_concepts
    st_timestep = args.st_timestep
    end_timestep = args.end_timestep
    n_avail_tokens = args.n_tokens
    # eta = args.eta
    # lamb = args.lamb
    update_rank = args.update_rank
    gate_rank = args.gate_rank
    degen_rank = args.degen_rank

    prompts_target = prompts_target[tar_concept_idx:tar_concept_idx+n_target_concepts]

    targets = [prompt.target for prompt in prompts_target]
    anchors = [prompt.target for prompt in prompts_anchor]
    surrogate = [prompts_target[0].neutral]
    updates = [prompt.target for prompt in prompts_update]

    # targets_fn = [prompt.target.replace(" ", "_") for prompt in prompts_target]
    # anchors_fn = [prompt.target.replace(" ", "_") for prompt in prompts_anchor]

    save_path = f"{args.save_path}/{targets[0].replace(' ', '_')}"     
    param_cache_path = args.param_cache_path 
    emb_cache_path = f"{args.emb_cache_path}/{targets[0].replace(' ', '_')}"
    register_buffer_path = f"{args.buffer_path}/{targets[0].replace(' ', '_')}"
    emb_cache_fn = args.emb_cache_fn
        
    model_metadata = {
        "prompts": ",".join([prompt.target for prompt in prompts_target]),
        "rank": str(config.network.rank),
        "alpha": str(config.network.alpha),
    }
    
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(args.sd_version) 

    text_encoder.to(DEVICE_CUDA)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    unet.to(DEVICE_CUDA)
    unet.requires_grad_(False)
    unet.eval()
    
    pipe.safety_checker = None

    ############# register org modules ############
    module_types = []
    module_names = []
    org_modules_all = []
    module_name_list_all = []
    param_vh_cache_dict_all = []
    param_s_cache_dict_all = []
    
    for find_module_name in args.find_module_name:
        module_name, module_type = get_module_name_type(find_module_name)            
        org_modules, module_name_list = get_modules_list(unet, text_encoder, find_module_name, module_name, module_type)
        param_vh_cache_dict, param_s_cache_dict = load_model_sv_cache(find_module_name, param_cache_path, DEVICE_CUDA, org_modules)

        module_names.append(module_name)
        module_types.append(module_type)
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)
        param_vh_cache_dict_all.append(param_vh_cache_dict)
        param_s_cache_dict_all.append(param_s_cache_dict)
    
    ################### Prepare network ####################
    network = GLoCENetworkOutProp(
        unet,
        multiplier=1.0,
        alpha=config.network.alpha,
        module=GLoCELayerOutProp,
        gate_rank=gate_rank,
        update_rank=update_rank,
        degen_rank=degen_rank,
        n_concepts=1,
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names = args.find_module_name,
        last_layer=args.last_layer,
    ).to(DEVICE_CUDA)

    print("gate rank of netowrk:" , config.network.init_size)

    network.eval()    

    embedding_unconditional = get_condition([""], tokenizer, text_encoder)

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

    ############### Prepare for text embedding token ###################    
    emb_cache = prepare_text_embedding_token(
        args,
        config,
        prompts_target,
        prompts_anchor,
        prompts_update,
        tokenizer,
        text_encoder,                                
        emb_cache_path,
        emb_cache_fn,
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
    
    prompt_scripts_path = config.scripts_file
    prompt_scripts_df = pd.read_csv(prompt_scripts_path)
    prompt_scripts_list = prompt_scripts_df['prompt'].to_list()
    len_prmpts_list = len(prompt_scripts_list) + 1

    use_prompt = args.find_module_name in ["unet_ca_v", "unet_ca_out"]
    if config.replace_word == "artist" and use_prompt:
        embeddings_surrogate_sel_base = embeddings_surrogate_cache
        embeddings_target_sel_base = embeddings_target_cache
        embeddings_anchor_sel_base = embeddings_anchor_cache
        embeddings_update_sel_base = embeddings_update_cache

        surrogate = prmpt_scripts_sur
        targets = prmpt_scripts_tar
        anchors = prmpt_scripts_anc
        updates = prmpt_scripts_upd

    ################# Compute register buffer for surrogate concept for erasing #################

    buffer_sel_basis_surrogate = get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        surrogate,
        embeddings_surrogate_sel_base,
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
        register_buffer_path,
        register_buffer_fn="stacked_surrogate.pt",
        register_func="register_sum_buffer_avg_spatial"
    )

    #################### Compute principal components for surrogate concept ######################

    Vh_sur_dict = dict()
    surrogate_mean_dict = dict()
    for find_name in args.find_module_name:
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

    ################# Compute registder buffer for target concept for erasing #################

    buffer_sel_basis_target = get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        targets,
        embeddings_target_sel_base,
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
        register_buffer_path,
        register_buffer_fn="stacked_target.pt",
        register_func="register_sum_buffer_avg_spatial"
    )

    #################### Compute principal components for target concept ######################

    target_mean_dict: dict[str, dict[str, nn.Module]] = dict()
    target_cov_dict = dict()
    Vh_tar_dict = dict()
    for find_name in args.find_module_name:
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
        gloce_module.lora_update.weight.data = (Vh_sur @ (torch.eye(dim_emb).to(DEVICE_CUDA)- Vh_upd.T @ Vh_upd)).T.contiguous()
        gloce_module.debias.weight.data = target_mean.unsqueeze(0).unsqueeze(0).clone().contiguous()  
    
    #################### Compute register buffer for surrogate for gate #######################

    buffer_sel_basis_gate = get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        updates,
        embeddings_update_sel_base,
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
        register_buffer_path,
        register_buffer_fn="stacked_gate.pt",
        register_func="register_sum_buffer_avg_spatial"
    )
    
    #################### Compute principal components of surrogate for gate ######################

    Vh_gate_dict: dict[str, dict[str, dict[str, nn.Module]]] = dict()
    gate_mean_dict = dict()
    rel_gate_dict = dict()
    for find_name in args.find_module_name:
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

    ############## Compute registder buffer for discriminative basis for erasing ##############
    buffer_norm_basis_target = get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        targets,
        embeddings_target_sel_base,
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
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
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
        register_buffer_path,
        register_buffer_fn="norm_anchor.pt",
        register_func="register_norm_buffer_avg_spatial",
        rel_gate_dict=rel_gate_dict,
        target_mean_dict=target_mean_dict,
        gate_mean_dict=gate_mean_dict
    )

    ############## Compute discriminative basis for erasing ##############
 
    for gloce_module in network.gloce_layers:        
        n_forward_tar = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_forward_anc = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]['n_forward']
        n_sum_tar = n_forward_tar
        n_sum_anc = n_forward_anc

        importance_tgt = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['data_max'] / n_sum_tar
        importance_anc = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]['data_max'] / n_sum_anc

        importance_tgt_stack = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]['data_stack']
        importance_anc_stack = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]['data_stack']
        importance_tgt_stack = torch.cat([imp.unsqueeze(0) for imp in importance_tgt_stack], dim=0)
        importance_anc_stack = torch.cat([imp.unsqueeze(0) for imp in importance_anc_stack], dim=0)

        ########### Determine parameters in logistic function ############

        tol1 = args.thresh

        x_center = importance_anc_stack.mean() + tol1 * importance_anc_stack.std()     
        tol2 = 0.001 * tol1

        c_right = torch.tensor([0.99]).to(DEVICE_CUDA)
        C_right = torch.log(1 / (1 / c_right - 1))

        imp_center = x_center
        imp_slope = C_right/tol2

        print(f"{importance_anc_stack.max().item():10.5f}, {imp_center.item():10.5f}, {importance_tgt_stack.min().item():10.5f}, {importance_tgt_stack.max().item():10.5f}")
        
        Vh_gate = rel_gate_dict[gloce_module.find_name][gloce_module.gloce_org_name]
        gate_mean = gate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name]

        # NxD
        gloce_module.selector.select_weight.weight.data = Vh_gate.T.unsqueeze(0).clone().contiguous()
        gloce_module.selector.select_mean_diff.weight.data = gate_mean.clone().contiguous()

        gloce_module.selector.imp_center = imp_center
        gloce_module.selector.imp_slope = imp_slope

    ############## Compute discriminative basis for erasing ##############

    print("saving gloce parameters...")
    save_path = Path(f"{save_path}")            
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(save_path / f"ckpt.safetensors", metadata=model_metadata)
    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_pkg.load_config_from_yaml(config_file)
        
    prompts_target = prompt_pkg.load_prompts_from_yaml(config.prompts_file_target)
    prompts_anchor = prompt_pkg.load_prompts_from_yaml(config.prompts_file_anchor)
    prompts_update = prompt_pkg.load_prompts_from_yaml(config.prompts_file_update)
    
    if args.gate_rank != -1:
        config.network.init_size = args.gate_rank
        config.network.hidden_size = args.gate_rank
        config.network.continual_rank = args.gate_rank
            
    if args.update_rank != -1:
        config.network.rank = args.update_rank     

    base_logging_prompts = config.logging.prompts
    
    for p_idx, p in enumerate(prompts_target):
        config.logging.prompts = [prompt.replace('[target]', p.target) if '[target]' in prompt else prompt for prompt in base_logging_prompts]
    
    args.find_module_name = args.find_module_name.split(",")
    if args.find_module_name.__class__ == str:
        args.find_module_name = [args.find_module_name]

    seed_everything(config.train.train_seed)        
    train(config, prompts_target, prompts_anchor, prompts_update, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", required=True, help="Config file for training.")
    parser.add_argument("--st_prompt_idx", type=int, default=-1)
    parser.add_argument("--end_prompt_idx", type=int, default=-1)
    parser.add_argument("--update_rank", type=int, default=-1)
    parser.add_argument("--degen_rank", type=int, default=-1)
    parser.add_argument("--gate_rank", type=int, default=-1)
    parser.add_argument("--n_tokens", type=int, default=-1)
    parser.add_argument("--eta", type=float, default=-1)
    parser.add_argument("--lamb", type=float, default=-1)
    parser.add_argument("--lamb2", type=float, default=-1)
    parser.add_argument("--p_val", type=float, default=-1)
    parser.add_argument("--find_module_name", type=str, default="unet_ca")

    parser.add_argument('--n_target_concepts', type=int, default=1, help="Number of target concepts")
    parser.add_argument('--n_anchor_concepts', type=int, default=5, help="Number of anchor concepts")
    parser.add_argument('--tar_concept_idx', type=int, default=0, help="Target concept index")
    parser.add_argument('--st_timestep', type=int, default=10, help="Start timestep")
    parser.add_argument('--end_timestep', type=int, default=20, help="End timestep")
    parser.add_argument('--n_generation_per_concept', type=int, default=3, help="End timestep")
    parser.add_argument('--sel_basis_buffer_fn', action='store_true', help="Select basis buffer function")
    parser.add_argument('--param_cache_path', type=str, default="./importance_cache/org_comps/sd_v1.4", help="Path to parameter cache")
    parser.add_argument('--emb_cache_path', type=str, default="./importance_cache/text_embs/sd_v1.4", help="Path to embedding cache")
    parser.add_argument('--emb_cache_fn', type=str, default="text_emb_cache_w_sel_base_chris_evans_anchor5.pt", help="Embedding cache file name")
    parser.add_argument("--buffer_path", type=str, default="./importance_cache/buffers")
    parser.add_argument("--use_emb_cache", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--last_layer", type=str, default="")
    parser.add_argument("--opposite_for_map", type=bool, default=False)
    parser.add_argument("--thresh", type=float, default=1.5)
    args = parser.parse_args()        

    main(args)
