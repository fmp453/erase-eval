# Precise, Fast, and Low-cost Concept Erasure in Value Space: Orthogonal Complement Matters

from copy import deepcopy
from pathlib import Path

from tqdm import tqdm
from einops import rearrange

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import Attention

from train_methods.templates import template_dict
from train_methods.train_utils import get_condition, tokenize, get_devices, get_models, predict_noise
from utils import Arguments


class VisualAttentionProcess(nn.Module):

    def __init__(
        self,
        module_name=None,
        atten_type='original',
        target_records=None,
        record=False,
        record_type=None,
        sigmoid_setting: tuple[float, ...]=(0.0),
        decomp_timestep=0,
        **kwargs
    ):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records  = target_records
        self.record = record
        self.record_type = record_type
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = decomp_timestep

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states, *args, **kwargs):
        attn._modules.pop("processor")
        attn.processor = AttnProcessor(self.module_name, self.atten_type, self.target_records, self.record, self.record_type, self.sigmoid_setting, self.decomp_timestep)
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)
    

class AttnProcessor():

    def __init__(
        self, 
        module_name=None, 
        atten_type='original', 
        target_records=None, 
        record=False, 
        record_type: str | None=None, 
        sigmoid_setting: tuple[float, ...]=(0.0, 0.0, 0.0),
        decomp_timestep=0
    ) -> None:
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = deepcopy(target_records)
        self.record = record
        self.record_type = record_type.strip().split(',') if record_type is not None else []
        self.records = {key: {} for key in self.record_type} if record_type is not None else {}
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep=decomp_timestep

    def sigmoid(self, x: torch.Tensor, setting) -> torch.Tensor:
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def cal_ortho_decomp(
        self,
        target_value: torch.Tensor | list[torch.Tensor],
        pro_record: torch.Tensor,
        ortho_basis: torch.Tensor | None=None,
        project_matrix=None
    ): 

        if ortho_basis is None and project_matrix is None:
            tar_record_ = target_value[0].permute(1, 0, 2).reshape(77, -1) # [77, 640]
            pro_record_ = pro_record.permute(1, 0, 2).reshape(77, -1) # [77, 640]
            dot1 = (tar_record_ * pro_record_).sum(-1)
            dot2 = (tar_record_ * tar_record_).sum(-1)
            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)
            if self.sigmoid_setting is not None:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)
            weight = torch.nan_to_num(cos_sim * (dot1 / dot2), nan=0.0)
            weight[0].fill_(0)
            era_record = weight.unsqueeze(0).unsqueeze(-1) * tar_record_.view((77, 16, -1)).permute(1, 0, 2)
        else:
            tar_record_ = rearrange(target_value, 'b h l d -> l b (h d)') # [77, num_concepts, 640]
            pro_record_ = rearrange(pro_record, 'h l d -> l (h d)').unsqueeze(1) # [77, 1, 640]
            dot1 = (ortho_basis * pro_record_).sum(-1)
            dot2 = (ortho_basis * ortho_basis).sum(-1)
            weight = torch.nan_to_num((dot1 / dot2).unsqueeze(1), nan=0.0)
            weight[0].fill_(0)
            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1) # [77, 2]
            if self.sigmoid_setting is not None:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)
            projected_basis = torch.bmm(project_matrix, cos_sim.unsqueeze(-1) * tar_record_)
            era_record = torch.bmm(weight, projected_basis).view((77, 16, -1)).permute(1, 0, 2)

        return era_record

    def record_ortho_decomp(self, target_record: dict[str, torch.Tensor], current_record: torch.Tensor):
        current_name = next(k for k in target_record if k.endswith(self.module_name))
        current_timestep, current_block = current_name.split('.', 1)
        (target_value, project_matrix, ortho_basis) = target_record.pop(current_name)

        if int(current_timestep) <=  self.decomp_timestep:
            return current_record, current_record

        if current_block in ORTHO_DECOMP_STORAGE: # if you decompose both key and value, don't use this global variable
            pass
        else:
            target_value = target_value.view((2, int(len(target_value)//16), -1)+ target_value.size()[-2:])
            target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
            current_record = current_record.view((2, int(len(current_record)//16), -1)+ target_value.size()[-2:])
            current_record = current_record.permute(1, 0, 2, 3, 4).contiguous().view((current_record.size()[1], -1) + target_value.size()[-2:])
            erase_record, retain_record = [], []

            for pro_record in current_record:
                era_record = self.cal_ortho_decomp(target_value, pro_record, ortho_basis, project_matrix)
                ret_record = pro_record - era_record
                erase_record.append(era_record.view((2, -1) + era_record.size()[-2:]))
                retain_record.append(ret_record.view((2, -1) + ret_record.size()[-2:]))
            retain_record = rearrange(torch.stack(retain_record, dim=0), 'b n c l d -> (n b c) l d')
            erase_record =  rearrange(torch.stack(erase_record, dim=0), 'b n c l d -> (n b c) l d')
            ORTHO_DECOMP_STORAGE[current_block] = (erase_record, retain_record)

        return ORTHO_DECOMP_STORAGE[current_block]

    def cal_gram_schmidt(self, target_value: torch.Tensor):
        target_value = target_value.view((2, int(len(target_value)//16), -1)+ target_value.size()[-2:])
        target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1)+target_value.size()[-2:])
        target_value_ = rearrange(target_value, 'b h l d -> b l (h d)')
        results = [self.gram_schmidt(target_value_[:, i, :]) for i in range(target_value_.size()[1])]
        project_matrix = torch.stack([result[0] for result in results], dim=0) # [77, 2, 2]
        basis_ortho = torch.stack([result[1] for result in results], dim=0) # [77, 2, 640]
        return project_matrix, basis_ortho
    
    def gram_schmidt(self, V: torch.Tensor): # [n, 1, d] 
        n = len(V)
        project_matrix = torch.zeros((n,n), dtype=V.dtype).to(V.device)+ torch.diag(torch.ones(n, dtype=V.dtype)).to(V.device)
        for i in range(1, n):
            vi = V[i:i+1, :]
            for j in range(i):
                qj = V[j:j+1, :]
                project_matrix[i][j] = -torch.dot(qj.view(-1), vi.view(-1)) / torch.dot(qj.view(-1), qj.view(-1))
        ortho_basis = torch.matmul(project_matrix.to(V.device), V)# n d
        
        return project_matrix.to(V.device), ortho_basis

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value) # [batch, head, len, dim//head]

        if not self.record and encoder_hidden_states.shape[1] == 77:
            if 'queries' in self.target_records:
                erase_query, retain_query = self.record_ortho_decomp(
                    target_record=self.target_records['queries'],
                    current_record=query,
                )
                query = retain_query if self.atten_type == 'retain' else erase_query if self.atten_type == 'erase' else query
            if 'keys' in self.target_records:
                erase_key, retain_key = self.record_ortho_decomp(
                    target_record=self.target_records['keys'],
                    current_record=key,
                )
                key = retain_key if self.atten_type == 'retain' else erase_key if self.atten_type == 'erase' else key

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if not self.record and encoder_hidden_states.shape[1] == 77:
            if 'attn_maps' in self.target_records:
                erase_attention_probs, retain_attention_probs = self.record_ortho_decomp(
                    target_record=self.target_records['attn_maps'],
                    current_record=attention_probs,
                )
                attention_probs = retain_attention_probs if self.atten_type == 'retain' else erase_attention_probs if self.atten_type == 'erase' else attention_probs

        if encoder_hidden_states.shape[1] != 77:
            # self-attention
            hidden_states = torch.bmm(attention_probs, value)
        else:  
            # cross-attention
            if self.record:
                for kk, vv in {'queries': query, 'keys': key, 'values': value, 'attn_maps': attention_probs}.items():
                    if kk in self.record_type:
                        if vv.shape[0] // 16 == 1: # single-concept
                            self.records[kk][self.module_name] = [vv] + [None, None]
                        else: # multi-concept
                            self.records[kk][self.module_name] = [vv] + list(self.cal_gram_schmidt(vv))
            elif 'values' in self.target_records:
                erase_value, retain_value = self.record_ortho_decomp(
                    target_record=self.target_records['values'],
                    current_record=value,
                )
                value = retain_value if self.atten_type == 'retain' else erase_value if self.atten_type == 'erase' else value
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)  # linear proj
        hidden_states = attn.to_out[1](hidden_states)  # # dropout

        if input_ndim == 4: hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection: hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor

def set_attenprocessor(
    unet: UNet2DConditionModel, 
    atten_type='original', 
    target_records=None, 
    record=False, 
    record_type=None, 
    sigmoid_setting=None, 
    decomp_timestep=0
) -> UNet2DConditionModel:
    for name, m in unet.named_modules():
        m: Attention
        if name.endswith('attn2') or name.endswith('attn1'):
            cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
            m.set_processor(VisualAttentionProcess(
                module_name=name, 
                atten_type=atten_type,
                target_records=target_records, 
                record=record,
                record_type=record_type,
                cross_attention_dim=cross_attention_dim, 
                sigmoid_setting=sigmoid_setting,
                decomp_timestep=decomp_timestep
            ))
    return unet

def get_eot_idx(tokens: torch.Tensor):
    return (tokens == 49407).nonzero(as_tuple=True)[1][0].item()

def get_spread_embedding(original_token: torch.Tensor, idx: int):
    spread_token = original_token.clone()
    spread_token[:, 1:, :] = original_token[:, idx-1, :].unsqueeze(1)
    return spread_token

def diffusion(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler | DDPMScheduler | DPMSolverMultistepScheduler,
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

ORTHO_DECOMP_STORAGE = {}

@torch.no_grad()
def main(args: Arguments):

    global ORTHO_DECOMP_STORAGE

    device = get_devices(args)[0]
    mode_list = args.adavd_mode.replace(' ', '').split(',')

    concept_list, concept_list_tmp = [], [item.strip() for item in args.adavd_contents.split(',')]
    if 'retain' in mode_list:
        for concept in concept_list_tmp:
            check_path = Path(args.save_dir, args.concepts.replace(', ', '_'), concept, 'retain')
            check_path.mkdir(exist_ok=True)
            if len(Path(check_path).iterdir()) != len(template_dict[args.adavd_erase_type]) * 10:
                concept_list.append(concept)
    else:
        concept_list = concept_list_tmp
    if len(concept_list) == 0: 
        exit()

    tokenizer, text_encoder, vae, unet, ddim_scheduler, _ = get_models(args)
    scheduler = DPMSolverMultistepScheduler.from_config(ddim_scheduler.config)
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    if 'erase' in mode_list: 
        unet_erase = deepcopy(unet)
    if 'retain' in mode_list: 
        unet_retain = deepcopy(unet)

    target_concepts = [item.strip() for item in args.concepts.split(',')]
    target_concept_encodings_ = [get_condition(prompt=concept, tokenizer=tokenizer, text_encoder=text_encoder) for concept in target_concepts]
    target_eot_idxs = [get_eot_idx(tokenize(prompt=concept, tokenizer=tokenizer).input_ids) for concept in target_concepts]
    target_concept_encoding = [get_spread_embedding(target_concept_encoding_, idx) for (target_concept_encoding_, idx) in zip(target_concept_encodings_, target_eot_idxs)]
    target_concept_encoding = torch.concat(target_concept_encoding)
    uncond_encoding = get_condition(prompt='', tokenizer=tokenizer, text_encoder=text_encoder)

    if 'erase' in mode_list or 'retain' in mode_list:
        unet = set_attenprocessor(unet, atten_type='original', record=True, record_type=args.adavd_record_type)
        _, target_records = diffusion(
            unet=unet, 
            scheduler=scheduler, 
            latents=torch.zeros(len(target_concept_encoding), 4, 64, 64).to(device, dtype=target_concept_encoding.dtype),
            text_embeddings=torch.cat([uncond_encoding] * len(target_concept_encoding) + [target_concept_encoding], dim=0),
            total_timesteps=1, 
            guidance_scale=args.guidance_scale, 
            record=True, 
            record_type=args.adavd_record_type, 
            desc="Calculating target records",
        )
        scheduler.set_timesteps(args.adavd_total_timesteps)
        original_keys = target_records[args.adavd_record_type].keys()
        target_records[args.adavd_record_type].update({
            f"{timestep}.{'.'.join(key.split('.')[1:])}": target_records[args.adavd_record_type][key]
            for timestep in scheduler.timesteps
            for key in original_keys
        })
    del unet
    if "erase" in mode_list:
        unet_erase = set_attenprocessor(
            unet_erase,
            atten_type='erase',
            target_records=deepcopy(target_records),
            sigmoid_setting=(args.adavd_sigmoid_a, args.adavd_sigmoid_b, args.adavd_sigmoid_c),
            decomp_timestep=args.adavd_decomp_timestep,
        )
        unet_erase.save_pretrained(f"{args.save_dir}/adavd/erase")
    elif "retain" in mode_list:
        unet_retain = set_attenprocessor(
            unet_retain, 
            atten_type='retain', 
            target_records=deepcopy(target_records), 
            sigmoid_setting=(args.adavd_sigmoid_a, args.adavd_sigmoid_b, args.adavd_sigmoid_c),
            decomp_timestep=args.adavd_decomp_timestep,
        )
        unet_erase.save_pretrained(f"{args.save_dir}/adavd/retain")
