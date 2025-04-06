# Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers

import random
from typing import Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange

from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer

from train_methods.train_utils import sample_until, apply_model
from utils import Arguments

def get_mask(attn_maps, word_indices, thres):
    """
    attn_maps: {name: attns in shape (bs, heads, h*w, text_len)}
    word_indices: (num_tokens,)
    thres: float, threshold of mask
    """
    name2res = {}
    attns_choosen = []
    for name, attns in attn_maps.items():
        name = diffuser_prefix_name(name)
        attns = attns[..., word_indices].mean(-1).mean(1)  # (bs, hw)
        res = int(np.sqrt(attns.shape[-1]))
        name2res[name] = res
        if res != 16:  # following MasaCtrl, we only use 16 x 16 cross attn maps
            continue
        attns = rearrange(attns, 'b (h w) -> b h w', h=res)  # (bs, h, w)
        attns_choosen.append(attns)
    # prepare mask
    attns_avg = torch.stack(attns_choosen, dim=1).mean(1)  # (bs, h, w)
    attn_min = attns_avg.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    attn_max = attns_avg.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    mask = (attns_avg - attn_min) / (attn_max - attn_min)  # normalize
    mask[mask >= thres] = 1
    mask[mask < thres] = 0
    # rescale mask for all possibility
    cached_masks = {}
    ret_masks = {}
    for name, res in name2res.items():
        if res in cached_masks:
            ret_masks[name] = cached_masks[res]
        else:
            rescaled_mask = F.interpolate(mask.unsqueeze(0), (res, res)).squeeze(0)
            cached_masks[res] = rescaled_mask
            ret_masks[name] = rescaled_mask
    return ret_masks

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

class DisableEraser:
    def __init__(self, unet: UNet2DConditionModel, train=False):
        self.model = unet
        self.train = train
        self.old_states = {}

    def __enter__(self):
        self.old_training = self.model.training
        self.model.train(self.train)
        # disable erasers
        for name, module in self.model.named_modules():
            if isinstance(module, EraserControlMixin):
                self.old_states[name] = module.use_eraser
                module.use_eraser = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.train(self.old_training)
        # enable erasers
        for name, module in self.model.named_modules():
            if isinstance(module, EraserControlMixin):
                module.use_eraser = self.old_states[name]

class AttnMapsCapture:
    def __init__(self, unet: UNet2DConditionModel, attn_maps):
        self.model = unet
        self.attn_maps = attn_maps
        self.handlers = []

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            # attn2 is cross attention layer
            # 最後のisinstance(module, Attention)は必要か？
            if 'transformer_blocks' in module_name and 'attn2' in module_name and isinstance(module, Attention):
                handler = module.register_forward_hook(self.get_attn_maps(module_name))
                self.handlers.append(handler)

    def __exit__(self, exc_type, exc_value, traceback):
        for handler in self.handlers:
            handler.remove()

    def get_attn_maps(self, module_name):
        def hook(model, input, output):
            self.attn_maps[module_name] = model.processor.attn_outs.detach()
        return hook

class EraserOutputsCapture:
    def __init__(self, unet: UNet2DConditionModel, erasers, eraser_outs):
        self.model = unet
        self.eraser_names = list(erasers.keys())
        self.eraser_outs = eraser_outs
        self.handlers = []

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if module_name in self.eraser_names:
                handler = module.register_forward_hook(self.get_eraser_outs(module_name))
                self.handlers.append(handler)

    def __exit__(self, exc_type, exc_value, traceback):
        for handler in self.handlers:
            handler.remove()

    def get_eraser_outs(self, module_name):
        def hook(model, input, output):
            if output.dim() == 2:
                output = torch.unsqueeze(output, 0)
            self.eraser_outs[module_name] = output
        return hook

def diffuser_prefix_name(name):
    block_type = name.split('.')[0]
    if block_type == 'mid_block':
        return '.'.join(name.split('.')[:3])
    return  '.'.join(name.split('.')[:4])

class BasicTransformerBlockWithEraser(BasicTransformerBlock):
    def __init__(self, dim, n_heads, d_head, eraser_rank, dropout=0., context_dim=None, activation_fn="geglu", checkpoint=True, only_cross_attention=False, *args, **kwargs):
        super().__init__(dim, n_heads, d_head, dropout, context_dim, activation_fn, checkpoint, only_cross_attention, *args, **kwargs)

        self.adapter = AdapterEraser(dim, eraser_rank)

    def forward(self, x, encoder_hidden_states=None, attention_mask=None, encoder_attention_mask=None, **kwargs):
        x = self.attn1(self.norm1(x), encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None, attention_mask=attention_mask)[0] + x
        if self.adapter.use_eraser:
            ca_output = self.attn2(self.norm2(x), encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask)[0]
            x = self.adapter(ca_output) + ca_output + x
        else:
            x = self.attn2(self.norm2(x), encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask)[0] + x
        x = self.ff(self.norm3(x)) + x
        return x

    @classmethod
    def from_pretrained_block(cls, block: BasicTransformerBlock, eraser_rank):
        dim = block.norm1.weight.shape[0]
        n_heads = block.attn1.heads
        d_head = round(block.attn1.scale ** -2)
        dropout = block.attn1.to_out[1].p
        context_dim = block.cross_attention_dim
        act_fn = block.activation_fn
        checkpoint = True
        disable_self_attn = block.only_cross_attention
        block_w_adapter = cls(dim, n_heads, d_head, eraser_rank, dropout, context_dim, act_fn, checkpoint, disable_self_attn)
        block_w_adapter.load_state_dict(block.state_dict(), strict=False)
        return block_w_adapter

class CustomAttnProcessor(AttnProcessor):
    def __init__(self):
        super().__init__()
        self.attn_outs = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
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

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        h = attn.heads
        self.attn_outs = rearrange(attention_probs, '(b h) i j -> b h i j', h=h)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def setup_unet_adapter_eraser(unet: UNet2DConditionModel, eraser_rank, device):
    def replace_transformer_block(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, BasicTransformerBlock):
                block_w_adapter = BasicTransformerBlockWithEraser.from_pretrained_block(child, eraser_rank).to(device)
                setattr(module, name, block_w_adapter)
            else:
                replace_transformer_block(child)
    replace_transformer_block(unet)
    erasers = {}
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlockWithEraser):
            eraser_name = f'{name}.adapter'
            print(eraser_name)
            erasers[eraser_name] = module.adapter
    return erasers

@torch.no_grad()
def get_learned_conditioning(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, prompt: list[str]) -> torch.Tensor:
    input_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        truncation=True
    ).input_ids
    device = text_encoder.device
    emb = text_encoder(input_ids.to(device))[0]
    return emb

def train_receler(args: Arguments):
    
    device = torch.device(f'cuda:{args.device.split(",")[0]}')

    # extend specific concept
    concept = args.concepts
    word_print = concept.replace(' ', '')
    original_concept = concept

    concept_mappings = {'i2p': "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"}

    concept = concept_mappings.get(original_concept, concept)

    # seperate concept string into (multiple) concepts
    if args.seperator is not None:
        words = concept.split(args.seperator)
        words = [word.strip() for word in words]
    else:
        words = [concept]

    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")    

    unet.eval()
    vae.eval()
    text_encoder.eval()
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    # setup eraser
    erasers = setup_unet_adapter_eraser(unet, eraser_rank=args.receler_rank, device=device)
    unet.set_attn_processor(CustomAttnProcessor())

    # setup optimizer
    opt = optim.Adam([param for eraser in erasers.values() for param in eraser.parameters()], lr=args.receler_lr)

    # lambda function for only denoising till time step t
    quick_sample_till_t = lambda x, s, code, t: sample_until(
        until=t,
        latents=code,
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=x,
        guidance_scale=s
    )
    
    print('\n'.join(['#'*50, word_print, '#'*50]))

    # dicts to store captured attention maps and eraser outputs
    attn_maps = {}
    eraser_outs = {}

    # create attack prompt embeddings
    if args.receler_advrs_iters:
        advrs_prompt_embs = [nn.Parameter(torch.rand((1, args.num_advrs_prompts, 768), device=device)) for _ in range(len(words))]
        advrs_prompt_opts = [optim.Adam([advrs_prompt_embs[idx]], lr=0.1, weight_decay=0.1) for idx in range(len(words))]

    scheduler.set_timesteps(args.ddim_steps, device)
    # training
    pbar = tqdm(range(args.receler_iterations))
    for it in pbar:
        unet.train()
        
        word_idx, word = random.sample(list(enumerate(words)),1)[0]
        # get text embeddings for unconditional and conditional prompts
        emb_0 = get_learned_conditioning(tokenizer, text_encoder, [''])
        emb_p = get_learned_conditioning(tokenizer, text_encoder, [f'{word}'])
        emb_n = get_learned_conditioning(tokenizer, text_encoder, [f'{word}'])

        # hacking the indices of targeted word and adversarial prompts
        text_len = len(tokenizer(f'{word}', add_special_tokens=False)['input_ids'])
        word_indices = torch.arange(1, 1 + text_len, device=device)
        advrs_indices = torch.arange(1 + text_len, 1 + text_len + args.num_advrs_prompts, device=device)

        # time step from 1000 to 0 (0 being good)
        t_enc = torch.randint(args.ddim_steps, (1,), device=device)
        og_num = round((int(t_enc) / args.ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=device)

        start_code = torch.randn((1, 4, 64, 64)).to(device)

        with torch.no_grad():
            # generate an image with the concept from model
            z = quick_sample_till_t(torch.cat([emb_0, emb_p], dim=0), args.start_guidance, start_code, int(t_enc)) 
            # get conditional and unconditional scores from frozen model at time step t and image z
            with DisableEraser(unet, train=False):
                e_0 = apply_model(unet, z.to(device), t_enc_ddpm.to(device), emb_0.to(device))
                with AttnMapsCapture(unet, attn_maps=attn_maps):
                    e_p = apply_model(unet, z.to(device), t_enc_ddpm.to(device), emb_p.to(device))

        attn_masks = get_mask(attn_maps, word_indices, args.receler_mask_thres)

        for inner_it in range(args.receler_advrs_iters):
            # copy advrs_prompt_emb to input emb_n and make it requires_grad if advrs train
            emb_n = emb_n.detach()
            emb_n[:, advrs_indices, :].data = advrs_prompt_embs[word_idx].data
            emb_n.requires_grad = True

            # get conditional score from model
            with EraserOutputsCapture(unet, erasers, eraser_outs=eraser_outs):
                e_n = apply_model(unet, z.to(device), t_enc_ddpm.to(device), emb_n.to(device))

            # perform advrs attack
            loss_adv = F.mse_loss(e_n, e_p, reduction='mean')
            tmp_grad, = torch.autograd.grad(loss_adv, [emb_n], retain_graph=True)
            advrs_prompt_embs[word_idx].grad = tmp_grad[:, advrs_indices, :]
            advrs_prompt_opts[word_idx].step()
            advrs_prompt_opts[word_idx].zero_grad()

            # perform erase training
            if inner_it == args.receler_advrs_iters - 1:
                loss_total = torch.tensor(0.).to(device)
                e_0.requires_grad = False
                e_p.requires_grad = False
                loss_erase = F.mse_loss(e_n, e_0 - (args.negative_guidance * (e_p - e_0)))
                loss_total += loss_erase
                # compute cross attn regularization loss
                loss_eraser_reg = torch.tensor(0.).to(device)
                reg_count = 0
                for e_name, e_out in eraser_outs.items():
                    prefix_name = diffuser_prefix_name(e_name)
                    if prefix_name not in attn_masks:
                        print(f'Warning: cannot compute regularization loss for {e_name}, because corresponding mask not found.')  # cannot find mask for regularizing
                        continue
                    reg_count += 1
                    mask = attn_masks[prefix_name]
                    flip_mask = (~mask.unsqueeze(1).bool()).float()  # (1, 1, w, h)
                    if e_out.dim() == 3:  # (1, w*h, dim) -> (1, dim, w, h)
                        w = flip_mask.shape[2]
                        e_out = rearrange(e_out, 'b (w h) d -> b d w h', w=w)
                    loss_eraser_reg += ((e_out * flip_mask) ** 2).mean(1).sum() / (flip_mask.sum() + 1e-9)
                loss_eraser_reg /= reg_count
                loss_total += args.receler_concept_reg_weight * loss_eraser_reg

                # update weights to erase the concept
                loss_total.backward()
                opt.step()
                opt.zero_grad()
                pbar.set_postfix({"loss_total": loss_total.item()})
                pbar.set_description_str(f"[{datetime.now().strftime('%H:%M:%S')}] Erase \"{concept}\"")

    unet.eval()
    unet.save_pretrained(args.save_dir)
    
def main(args: Arguments):
    train_receler(args)
