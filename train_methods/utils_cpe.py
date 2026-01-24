import random
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

from train_methods.train_spm import PromptEmbedsPair

class ParamModule(nn.Module):
    def __init__(self, size):
        super(ParamModule, self).__init__()
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x * self.weight

    def __repr__(self):
        return f"ParameterModule(param_shape={tuple(self.weight.shape)})"


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_list: list[torch.Tensor], k_list: list[torch.Tensor], mask: Optional[torch.Tensor]=None):
        for i, (q, k) in enumerate(zip(q_list, k_list)):
            if i == 0:
                attn = torch.matmul(q, k.transpose(3, 4)) / self.temperature
            else:
                attn += torch.matmul(q, k.transpose(3, 4)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.sum(dim=2)
        return F.softmax(attn, dim=-1)


class AttentionModule(nn.Module):
    def __init__(self, init_size, n_head, d_model, d_k, dropout=0.25, task_id=0, n_concepts=1):
        super().__init__()        
        
        self.n_head = n_head
        self.d_model = d_model
        self.init_size = init_size
        self.n_concepts = n_concepts
        self.temperature = 1.0
        
        self.d_k = n_head * d_k
        self.task_id = task_id
        
        self.w_qs = ParamModule((n_concepts, d_model, init_size))
        self.w_ks = ParamModule((n_concepts, d_model, init_size))
        
        self.w_qs_list = [self.w_qs]
        self.w_ks_list = [self.w_ks]
        
        nn.init.kaiming_uniform_(self.w_qs.weight, a=math.sqrt(5))
        self.w_qs.weight.data = self.w_qs.weight.data / (d_model**2)    
        nn.init.kaiming_uniform_(self.w_ks.weight, a=math.sqrt(5))
        self.w_ks.weight.data = self.w_ks.weight.data / (d_model**2)          

        
        self.attention = ScaledDotProductAttention(temperature=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None):
        q, k, v = x, x, x
        
        input = q.unsqueeze(1)
        n_head = self.n_head
        sz_b, len_q, len_k, _ = q.size(0), q.size(1), k.size(1), v.size(1)        
        
        q_list = []
        k_list = []
                            
        for w_qs, w_ks in zip(self.w_qs_list, self.w_ks_list):
            wq = torch.einsum("btd,ndh->bnth", q, w_qs.weight)  # 10x50x77x768, 50x768x4
            wk = torch.einsum("btd,ndh->bnth", k, w_ks.weight)
            
            q_list.append(wq.view(sz_b, self.n_concepts, len_q, n_head, wq.shape[-1]//n_head).transpose(2, 3))
            k_list.append(wq.view(sz_b, self.n_concepts, len_k, n_head, wk.shape[-1]//n_head).transpose(2, 3))
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        attn = self.attention(q_list, k_list, mask=mask)        
        e_aggregated = torch.einsum("bnst,bitd->bnsd", attn, input)

        return e_aggregated, attn


class GateModule(nn.Module):
    def __init__(self, init_size, n_head, d_model, d_k, dropout=0.25, task_id=0, n_concepts=1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.init_size = init_size
        self.n_concepts = n_concepts
        self.temperature = 1.0
        self.d_k = n_head * d_k
        self.task_id = task_id
        self.scaler = ParamModule((n_concepts, d_model, 1))      
        nn.init.zeros_(self.scaler.weight)

    def forward(self, e_aggregated: torch.Tensor):
        scale = torch.sigmoid(self.temperature * torch.einsum("ndi,bntd->bnti", self.scaler.weight, e_aggregated))
        scale_max_val, scale_max_ind = scale.max(dim=1, keepdim=True)
        ind_one_hot = F.one_hot(scale_max_ind.squeeze(-1), num_classes=self.n_concepts).permute(0,3,2,1)
        e_aggregated = (scale_max_val*ind_one_hot*e_aggregated).sum(dim=1)

        return e_aggregated, ind_one_hot, scale_max_val, scale_max_ind


class MultiHeadAttention(nn.Module):
    
    def __init__(self, init_size, n_head, d_model, d_k, dropout=0.25, task_id=0, n_concepts=1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.init_size = init_size
        self.n_concepts = n_concepts
        self.temperature = 1.0
        self.d_k = n_head * d_k
        self.task_id = task_id 
        self.attention_module1 = AttentionModule(init_size, n_head, d_model, d_k, n_concepts=n_concepts)
        self.gate = GateModule(init_size, n_head, d_model, d_k, n_concepts=n_concepts)
        
    def forward(self, x, mask=None):
        e_aggregated, attn = self.attention_module1(x)
        e_aggregated, ind_one_hot, _, _ = self.gate(e_aggregated)
        
        return e_aggregated, attn, ind_one_hot    


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, init_size, d_k, d_v, dropout=0.1, task_id=0, n_concepts=1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(init_size, n_head, d_model, d_k, dropout=dropout, task_id=task_id, n_concepts=n_concepts)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn, ind_one_hot = self.slf_attn(enc_input, mask=slf_attn_mask)
        return enc_output, enc_slf_attn, ind_one_hot


class Attention_Gate(nn.Module):
    def __init__(
        self,
        input_size=768,
        init_size=16,
        hidden_size=16,
        num_embeddings=77,
        n_head=1,
        dropout=0.5,
        task_id=0,
        n_concepts=1
    ):
        super(Attention_Gate, self).__init__()   
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_size = init_size
        self.n_head = n_head
        self.num_embeddings = num_embeddings
        self.inner_size = self.n_head * self.hidden_size
        self.enc_cache = None
        self.last_prompt = None
        self.encoder_layer = EncoderLayer(
            self.input_size, 
            self.inner_size,
            self.n_head,
            self.init_size,
            self.hidden_size,
            self.hidden_size,
            dropout=dropout,
            task_id=task_id,
            n_concepts=n_concepts
        )
        self.use_cache = False
        self.enc_output = None
        self.ind_one_hot = None
        
    def forward(self, x):
        enc_output, _, ind_one_hot  = self.encoder_layer(x, slf_attn_mask=None)
        return enc_output, ind_one_hot  

    def reset_cache(self):
        self.enc_output = None
        self.ind_one_hot = None
    
    def set_cache(self, flag):
        self.reset_cache()
        self.use_cache = flag

class CPELayer_ResAG(nn.Module):
    """replaces forward method of the original Linear, instead of replacing the otrain_prompt_editor_mixup_sampriginal Linear module.
    """

    def __init__(
        self,
        cpe_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
        init_size=1,
        hidden_size=1,
        num_embeddings=77,
        task_id=1,
        attention_gate=None,
        n_concepts=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.cpe_name = cpe_name
        self.dim = dim

        if org_module.__class__.__name__ == "Linear":
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            # dim of lora_down: N x D x H
            # dim of lora_up: N x H x D          
            self.lora_down = ParamModule((n_concepts, in_dim, dim))
            self.lora_up = ParamModule((n_concepts, dim, out_dim))  

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)            
            
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().numpy()
        alpha = dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.attention_gate = attention_gate
        self.use_prompt_tuning = False    
        self.inference_mode = False
        self.att_output = None        
        
    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # x.shape: (B, 77, 768)
                
        if self.use_prompt_tuning:    
            if not self.inference_mode:
                edit_direction, ind_one_hot = self.attention_gate(x)

                # dim of edit_direction: B x T x D
                # dim of ind_one_hot: B x N x T x 1
                # dim of lora_down: N x D x H
                # dim of lora_up: N x H x D

                # selection_down: (lora_down)NDH, (ind_one_hot)BNT1 -> BTDH
                # selection_up  : (lora_up)NHD, (ind_one_hot)BNT1 -> BTHD
                # compute down  : (edit)BTD, (selected_lora_down)BTDH -> BTH 
                # compute up    : (edit)BTH, (selected_lora_down)BTHD -> BTD 
                selection_down = torch.einsum("ndh,bnti->btdh", self.lora_down.weight, ind_one_hot.float())
                selection_up = torch.einsum("nhd,bnti->bthd", self.lora_up.weight, ind_one_hot.float())
                down = torch.einsum("btd,btdh->bth", edit_direction, selection_down)
                up = torch.einsum("bth,bthd->btd", down, selection_up)

                return self.org_forward(x) + up * self.multiplier * self.scale

            elif (self.inference_mode) and (self.att_output is None):
                edit_direction, ind_one_hot = self.attention_gate(x)
                selection_down = torch.einsum("ndh,bnti->btdh", self.lora_down.weight, ind_one_hot.float())
                selection_up = torch.einsum("nhd,bnti->bthd", self.lora_up.weight, ind_one_hot.float())
                down = torch.einsum("btd,btdh->bth", edit_direction, selection_down)
                up = torch.einsum("bth,bthd->btd", down, selection_up)
                self.att_output = up
                    
                return self.org_forward(x) + self.att_output * self.multiplier * self.scale
                    
            else:
                return self.org_forward(x) + self.att_output * self.multiplier * self.scale

        else:
            return self.org_forward(x)  

        
class CPENetwork_ResAG(nn.Module):
    UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
        "Transformer2DModel",
    ]
    UNET_TARGET_REPLACE_MODULE_CONV = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]

    CPE_PREFIX_UNET = "lora_unet"   # aligning with SD webui usage
    DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        module = CPELayer_ResAG,
        module_kwargs = None,
        delta=1e-5,
        num_embeddings=77,
        text_emb_dimension=768,
        hidden_size=4,
        edit_scale=2.5,
        continual=False,
        task_id=None,    
        continual_rank=4,
        init_size=4,
        n_concepts=1,
    ) -> None:
        
        super().__init__()
        
        self.continual=continual
        self.task_id=task_id
        self.continual_rank=continual_rank
        self.n_concepts = n_concepts
        self.multiplier = multiplier
        self.dim = rank
        self.alpha = alpha
        self.delta = delta 
        self.module = module
        self.module_kwargs = module_kwargs or {}
        self.num_embeddings = num_embeddings
        self.text_emb_dimension = text_emb_dimension
        self.hidden_size = hidden_size
        self.init_size = init_size
        self.edit_scale = edit_scale

        self.attention_gate = Attention_Gate(
            input_size=self.text_emb_dimension, 
            init_size=self.init_size,
            hidden_size=self.hidden_size,
            num_embeddings=self.num_embeddings, 
            task_id=self.task_id, 
            n_concepts=self.n_concepts
        )

        self.unet_cpe_layers = self.create_modules(
            CPENetwork_ResAG.CPE_PREFIX_UNET,
            unet,
            CPENetwork_ResAG.DEFAULT_TARGET_REPLACE,
            self.dim,
            self.multiplier,
        )

        print(f"Create CPE for U-Net: {len(self.unet_cpe_layers)} modules.")

        cpe_names = set()
        for cpe_layer in self.unet_cpe_layers:
            assert (
                cpe_layer.cpe_name not in cpe_names
            ), f"duplicated CPE layer name: {cpe_layer.cpe_name}. {cpe_names}"
            cpe_names.add(cpe_layer.cpe_name)

        for cpe_layer in self.unet_cpe_layers:
            cpe_layer.apply_to()
            self.add_module(
                cpe_layer.cpe_name,
                cpe_layer,
            )
        
        del unet

    def reset_cache_attention_gate(self):
        for layer in self.unet_cpe_layers:
            layer.att_output = None
      
    def set_inference_mode(self):
        for layer in self.unet_cpe_layers:
            layer.inference_mode = True    
      
    def set_train_mode(self):
        for layer in self.unet_cpe_layers:
            layer.inference_mode = False        

    def load_cpe_lora_models(self, model_paths):
        for layer in self.unet_cpe_layers:
            self.attention.encoder_layer.add_slf_attn(model_paths, layer.cpe_name)        

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: list[str],
        rank: int,
        multiplier: float,
    ) -> list:
        cpe_layers = []

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d"]:
                        if not ("attn2" in child_name):
                            continue
                        
                        if not(("to_k" in child_name) or ("to_v" in child_name)):
                            continue
                        
                        cpe_name = prefix + "." + name + "." + child_name
                        cpe_name = cpe_name.replace(".", "_")
                        print(f"{cpe_name=}")
                        
                        cpe_layer = self.module(
                            cpe_name, 
                            child_module, 
                            multiplier, 
                            rank, 
                            self.alpha,
                            init_size=self.init_size, 
                            hidden_size=self.hidden_size,
                            num_embeddings=self.num_embeddings,
                            task_id=self.task_id,
                            attention_gate=self.attention_gate,
                            n_concepts=self.n_concepts,
                            **self.module_kwargs
                        )
                        cpe_layers.append(cpe_layer)

        return cpe_layers
    
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):    
        all_params = []

        param_data = {"params": self.parameters()}
        if default_lr is not None:
            param_data["lr"] = default_lr
        all_params.append(param_data)                

        return all_params
    
    def save_weights(self, file: str, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()
        state_dict_save = dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                if ("lora" in key) and ("attention_gate" in key):
                    continue                
                
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict_save[key] = v

        if file.endswith(".safetensors"):
            save_file(state_dict_save, file, metadata)
        else:
            torch.save(state_dict_save, file)
    
    def __enter__(self):
        for cpe_layer in self.unet_cpe_layers:
            cpe_layer.multiplier = 1.0
            cpe_layer.use_prompt_tuning = True

    def __exit__(self, exc_type, exc_value, tb):
        for cpe_layer in self.unet_cpe_layers:
            cpe_layer.multiplier = 0
            cpe_layer.use_prompt_tuning = False


class PromptTuningLayer(nn.Module):
    def __init__(self, num_add_prompts, num_tokens, token_dim, device):
        super(PromptTuningLayer, self).__init__()
        
        self.num_add_prompts = num_add_prompts
        self.num_tokens = num_tokens
        self.token_dim = token_dim        
        self.device = device
        self.weight_dtype = torch.float32
        self.prompts = None # ParamModule(size)  # Zero-initialized learnable tensor
        self.prompts_prev = None
        self.len_prompts = 0
        self.len_prompts_prev = 0
    
    def expand_prompts(self, num_add_prompts=None):
        num_new_prompts = num_add_prompts if num_add_prompts is not None else self.num_add_prompts        
        self.len_prompts_prev += self.len_prompts
        self.len_prompts = num_new_prompts
        
        if self.prompts is not None and self.prompts_prev is not None:
            tmp = self.prompts_prev
            size_prev = tmp.weight.size() 
            self.prompts_prev = ParamModule(size=(size_prev[0] + self.prompts.weight.size(0), size_prev[1], size_prev[2])).to(self.device, self.weight_dtype)
            self.prompts_prev.weight.data[:tmp.weight.size(0)] = tmp.weight.data.detach()
            self.prompts_prev.weight.data[tmp.weight.size(0):] = self.prompts.weight.data.detach()            

        elif self.prompts is not None and self.prompts_prev is None:
            self.prompts_prev = self.prompts

        size = (num_new_prompts, self.num_tokens, self.token_dim)
        self.prompts = ParamModule(size=size).to(self.device, self.weight_dtype)
        nn.init.kaiming_uniform_(self.prompts.weight, a=math.sqrt(5))
        self.prompts.weight.data = self.prompts.weight.data / (self.token_dim**2)    

    def forward(self, x, idx=None):
        return x + self.prompts.weight if idx is None else x + self.prompts.weight[idx]
        
    def forward_eval(self, x, idx=None):
        return x + self.prompts.weight.detach() if idx is None else x + self.prompts.weight[idx].detach()

    def forward_prev(self, x, idx=None):
        return x + self.prompts_prev.weight if idx is None else x + self.prompts_prev.weight[idx]
    
    def forward_prev_eval(self, x, idx=None):
        return x + self.prompts_prev.weight.detach() if idx is None else x + self.prompts_prev.weight[idx].detach()
    
    def save_weights(self, file: str, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        state_dict_save = dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict_save[key] = v
                
        if file.endswith(".safetensors"):
            save_file(state_dict_save, file, metadata)
        else:
            torch.save(state_dict_save, file)


class AnchorSamplerGensim():
    
    def sample_mixup_batch_cache(
        self, 
        prompt_pair: PromptEmbedsPair,
        embeddings_anchor_cache: torch.Tensor,
        scale=0.001,
        mixup=True
    ):
        inds = []
        for _ in range(2 * prompt_pair.sampling_batch_size * prompt_pair.target.shape[0]):
            inds.append(random.randint(0, embeddings_anchor_cache.size(0) - 1))

        embs = embeddings_anchor_cache[inds]
        D,H,W = embs.shape[0], embs.shape[1], embs.shape[2]
        noise = scale * embs.view(D, -1).norm(2, dim=1, keepdim=True).unsqueeze(-1) * torch.randn_like(embs)
        samples = embs + noise
        samples_pair = samples.view(D//2, 2, H, W)

        # MixUp 
        mixRate = torch.tensor(np.random.beta(1.0, 1.0, (prompt_pair.sampling_batch_size * prompt_pair.target.shape[0], 1, 1))).to(samples_pair.device)

        samples = mixRate * samples_pair[:,0,:] + (1 - mixRate) * samples_pair[:,1,:] if mixup else samples_pair[:,0,:]

        if prompt_pair.unconditional.shape[0] == 1:
            samples = [torch.cat([prompt_pair.unconditional, samples[idx].unsqueeze(0)]) for idx in range(samples.shape[0])]
        else:
            samples = [torch.cat([prompt_pair.unconditional[0].unsqueeze(0), samples[idx].unsqueeze(0)]) for idx in range(samples.shape[0])]
        return torch.cat(samples, dim=0).float()
