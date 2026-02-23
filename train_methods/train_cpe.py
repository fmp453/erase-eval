# Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate
# https://github.com/Hyun1A/CPE
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from pathlib import Path
import pandas as pd
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import bitsandbytes as bnb
from tqdm import tqdm, trange
from diffusers import PNDMScheduler
from diffusers.models.attention_processor import Attention
from torch.optim.lr_scheduler import LRScheduler
from diffusers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType

from train_methods.train_spm import PromptSettings, PromptEmbedsPair
from train_methods.utils_cpe import CPELayer_ResAG, CPENetwork_ResAG, PromptTuningLayer, AnchorSamplerGensim
from train_methods.train_utils import get_devices, get_models, get_condition, seed_everything
from utils import Arguments


def get_scheduler_fix(optimizer, iterations, lr_scheduler_num_cycles, lr_warmup_steps, num_processes: int = 1):
    num_training_steps = iterations * num_processes  
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[SchedulerType("cosine_with_restarts")]
    return schedule_func(optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=num_training_steps, num_cycles=lr_scheduler_num_cycles)


def train_erase_one_stage(
    args: Arguments,
    stage: int,
    pbar: tqdm[int],
    device_cuda,
    network: CPENetwork_ResAG,
    adv_prompts: PromptTuningLayer,
    network_modules: dict[str, nn.Module],
    unet_modules: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: LRScheduler,
    criteria,
    prompts: list[PromptSettings],
    embedding_unconditional,
    anchor_sampler: AnchorSamplerGensim,
    lipschitz: dict[str, list[torch.Tensor]],
    embeddings_erase_cache: torch.Tensor,
    embeddings_anchor_cache: torch.Tensor,
    trainable_params: list[nn.Module],
):

    for _ in pbar:
        loss: dict[str, torch.Tensor] = {}
        optimizer.zero_grad()
        prompt_one = prompts

        cache: dict[str, torch.Tensor] = {}
        with torch.no_grad():            
            prompt_pairs: list[PromptEmbedsPair] = []

            for settings in prompt_one:
                ind = random.randint(0, embeddings_erase_cache.size(0)-1)
                embeddings = embeddings_erase_cache[ind]

                cache[settings.target] = embeddings[0].unsqueeze(0)
                cache[settings.neutral] = embeddings[1].unsqueeze(0)
                cache['unconditional'] = embedding_unconditional

                prompt_pair = PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.target],
                    cache['unconditional'],
                    cache[settings.neutral],
                    settings,
                )
                assert prompt_pair.sampling_batch_size % prompt_pair.batch_size == 0
                prompt_pairs.append(prompt_pair)

        # Prepare for anchoring prompt
            anchors = anchor_sampler.sample_mixup_batch_cache(
                prompt_pair,
                embeddings_anchor_cache=embeddings_anchor_cache,
                scale=args.cpe_noise_scale,
                mixup=args.cpe_mixup
            )
            
        # Prepare adversairal prompt
        len_emb_adv = 0
        if adv_prompts.len_prompts > 0:
            embeddings_adv = adv_prompts.forward_eval(prompt_pair.target)
            len_emb_adv = embeddings_adv.size(0)
        
        # loss_prompt_erase/anchor
        pal = torch.tensor([args.cpe_pal]).float().to(device=device_cuda)     

        pal_k_coef_log_dict_erase = dict()
        pal_v_coef_log_dict_erase = dict()        
        loss_prompt_erase_to_k = 0
        loss_prompt_erase_to_v = 0

        pal_k_coef_log_dict_anchor = dict()
        pal_v_coef_log_dict_anchor = dict()
        loss_prompt_anchor_to_k = 0
        loss_prompt_anchor_to_v = 0   

        loss_adv_erase_to_k = torch.tensor([0.]).float().to(device=device_cuda)     
        loss_adv_erase_to_v = torch.tensor([0.]).float().to(device=device_cuda)     
        
        idx = 0
    
        for name in network_modules.keys():
            if not "lora_unet" in name or "lora_adaptor" in name:
                continue

            targets = torch.cat([prompt_pair.target, embeddings_adv]) if len_emb_adv > 0 else prompt_pair.target

            with torch.no_grad():
                crsattn_org: torch.Tensor = unet_modules[name](torch.cat([targets, prompt_pair.neutral, prompt_pair.unconditional, anchors[1::2]], dim=0).float())
                crsattn_target_org = crsattn_org[0].unsqueeze(0)
                crsattn_neutral_org = crsattn_org[1 + len_emb_adv].unsqueeze(0)
                crsattn_comp_org = crsattn_org[(2+len_emb_adv):]
                
                if len_emb_adv > 0:
                    crsattn_neutral_adv_org = crsattn_org[1+len_emb_adv].unsqueeze(0).repeat(len_emb_adv,1,1)

            with network:
                crsattn: torch.Tensor = unet_modules[name](torch.cat([targets, prompt_pair.neutral, prompt_pair.unconditional, anchors[1::2]], dim=0).float())
                crsattn_target = crsattn[0].unsqueeze(0)
                crsattn_comp = crsattn[(2+len_emb_adv):]
                
                if len_emb_adv > 0:
                    crsattn_target_adv = crsattn[1:1+len_emb_adv]

            g_scale = prompt_pair.guidance_scale
            if "to_k" in name:
                lipschitz_for_key_target = (lipschitz['lipschitz_ov'][idx]*lipschitz['lipschitz_q'][idx]).unsqueeze(0).unsqueeze(1).unsqueeze(2) / prompt_pair.target.shape[1]
                loss_prompt_erase_to_k += (lipschitz_for_key_target * ((crsattn_neutral_org - crsattn_target) + g_scale *(crsattn_neutral_org-crsattn_target_org)) ** 2).mean()
                if len_emb_adv > 0:
                    loss_adv_erase_to_k += (lipschitz_for_key_target * ((crsattn_neutral_adv_org - crsattn_target_adv) + g_scale *(crsattn_neutral_org-crsattn_target_org)) ** 2).mean()
                pal_k_coef_log_dict_erase[f"pal_k_coef_log_dict_erase/{idx}th-layer"] = lipschitz_for_key_target.mean()

                lipschitz_for_key_comp = (lipschitz['lipschitz_ov'][idx]*lipschitz['lipschitz_q'][idx]).unsqueeze(0).unsqueeze(1).unsqueeze(2) / prompt_pair.target.shape[1]
                loss_prompt_anchor_to_k += (lipschitz_for_key_comp * (crsattn_comp_org-crsattn_comp) ** 2).mean()
                pal_k_coef_log_dict_anchor[f"pal_k_coef_log_dict_anchor/{idx}th-layer"] = lipschitz_for_key_comp.mean()                

            else:
                lipschitz_for_val_target = lipschitz['lipschitz_o'][idx].unsqueeze(0).unsqueeze(1)
                loss_prompt_erase_to_v += (lipschitz_for_val_target * ((crsattn_neutral_org - crsattn_target) + g_scale*(crsattn_neutral_org-crsattn_target_org)) ** 2).mean()
                if len_emb_adv > 0:
                    loss_adv_erase_to_v += (lipschitz_for_val_target * ((crsattn_neutral_adv_org - crsattn_target_adv) + g_scale*(crsattn_neutral_org-crsattn_target_org)) ** 2).mean()
                pal_v_coef_log_dict_erase[f"pal_v_coef_log_dict_erase/{idx}th-layer"] = lipschitz_for_val_target.mean()

                lipschitz_for_val_comp = lipschitz['lipschitz_o'][idx].unsqueeze(0).repeat(crsattn_comp.shape[0], 1).unsqueeze(2)
                loss_prompt_anchor_to_v += (lipschitz_for_val_comp * (crsattn_comp_org-crsattn_comp)**2).mean()
                pal_v_coef_log_dict_anchor[f"pal_v_coef_log_dict_anchor/{idx}th-layer"] = lipschitz_for_val_comp.mean()            

                idx += 1
        
        loss_prompt_erase_to_k = loss_prompt_erase_to_k / len(network_modules)
        loss_prompt_erase_to_v = loss_prompt_erase_to_v / len(network_modules)
        loss_prompt_erase = loss_prompt_erase_to_v + loss_prompt_erase_to_k       

        loss_prompt_anchor_to_k = loss_prompt_anchor_to_k / len(network_modules)
        loss_prompt_anchor_to_v = loss_prompt_anchor_to_v / len(network_modules)
        loss_prompt_anchor = loss_prompt_anchor_to_v + loss_prompt_anchor_to_k        


        loss_adv_erase_to_k = loss_adv_erase_to_k / len(network_modules)
        loss_adv_erase_to_v = loss_adv_erase_to_v / len(network_modules)
        loss_adv_erase = loss_adv_erase_to_v + loss_adv_erase_to_k
        loss[f"loss_erasing_stage{stage}/loss_prompt_erase"] = loss_prompt_erase
        loss[f"loss_erasing_stage{stage}/loss_prompt_erase_to_k"] = loss_prompt_erase_to_k
        loss[f"loss_erasing_stage{stage}/loss_prompt_erase_to_v"] = loss_prompt_erase_to_v

        loss[f"loss_erasing_stage{stage}/loss_prompt_anchor"] = loss_prompt_anchor
        loss[f"loss_erasing_stage{stage}/loss_prompt_anchor_to_k"] = loss_prompt_anchor_to_k
        loss[f"loss_erasing_stage{stage}/loss_prompt_anchor_to_v"] = loss_prompt_anchor_to_v 

        loss[f"loss_erasing_stage{stage}/loss_adv_erase"] = loss_adv_erase 
        
        adv_coef = args.cpe_adv_coef
        loss[f"loss_erasing"] = loss[f"loss_erasing_stage{stage}/loss_prompt_erase"] + adv_coef * loss[f"loss_erasing_stage{stage}/loss_adv_erase"] + pal * loss[f"loss_erasing_stage{stage}/loss_prompt_anchor"]

        loss["pal"] = pal
        loss["guidance"] = torch.tensor([prompt_pair.guidance_scale]).cuda()
        loss["la_strength"] = torch.tensor([prompt_pair.la_strength]).cuda()
        loss["batch_size"] = torch.tensor([prompt_pair.batch_size]).cuda()               

        loss[f"loss_erasing"].backward()

        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm, norm_type=2)
        optimizer.step()
        lr_scheduler.step()

        pbar.set_description(f"Loss: {loss[f'loss_erasing'].item():.4f}")

    return network, adv_prompts, 

def train_adv_one_stage(
    args: Arguments,
    stage,
    pbar_adv: tqdm[int],
    network,
    adv_prompts: nn.Module,
    network_modules: dict[str, nn.Module],
    unet_modules,
    optimizer_adv: optim.Optimizer,
    lr_scheduler_adv: LRScheduler,
    criteria_adv,
    prompts: list[PromptSettings],
    embedding_unconditional,
    lipschitz: dict[str, list[torch.Tensor]],
    embeddings_erase_cache: torch.Tensor,
    trainable_params_adv: list[nn.Module]
):

    for _ in pbar_adv:
        loss_adv: dict[str, torch.Tensor] = dict()
        optimizer_adv.zero_grad()

        # Prepare for erasing prompt
        prompt_one = prompts

        cache: dict[str, torch.Tensor] = {}
        with torch.no_grad():            
            prompt_pairs: list[PromptEmbedsPair] = []

            for settings in prompt_one:
                ind = random.randint(0, embeddings_erase_cache.size(0) - 1)
                embeddings = embeddings_erase_cache[ind]

                cache[settings.target] = embeddings[0].unsqueeze(0)
                cache[settings.neutral] = embeddings[1].unsqueeze(0)
                cache['unconditional'] = embedding_unconditional

                prompt_pair = PromptEmbedsPair(
                    criteria_adv,
                    cache[settings.target],
                    cache[settings.target],
                    cache['unconditional'],
                    cache[settings.neutral],
                    settings,
                )
                assert prompt_pair.sampling_batch_size % prompt_pair.batch_size == 0
                prompt_pairs.append(prompt_pair)

        # Prepare adversairal prompt
        embeddings_adv: torch.Tensor = adv_prompts.forward(prompt_pair.target)
        len_emb_adv = embeddings_adv.size(0)

        loss_prompt_adv_to_k = 0
        loss_prompt_adv_to_v = 0

        idx = 0
        for name in network_modules.keys():
            if not "lora_unet" in name or "lora_adaptor" in name:
                continue

            targets = embeddings_adv

            with torch.no_grad():
                crsattn_target_org: torch.Tensor = unet_modules[name](prompt_pair.target).float().repeat(len_emb_adv,1,1)

            with network:
                crsattn_target_adv: torch.Tensor = unet_modules[name](targets).float()

            if "to_k" in name:
                lipschitz_for_key_target = (lipschitz['lipschitz_ov'][idx] * lipschitz['lipschitz_q'][idx]).unsqueeze(0).unsqueeze(1).unsqueeze(2) / prompt_pair.target.shape[1]
                loss_prompt_adv_to_k += (lipschitz_for_key_target * (crsattn_target_adv - crsattn_target_org) ** 2).mean()

            else:
                lipschitz_for_val_target = lipschitz['lipschitz_o'][idx].unsqueeze(0).unsqueeze(1)
                loss_prompt_adv_to_v += (lipschitz_for_val_target * (crsattn_target_adv - crsattn_target_org) ** 2).mean()

                idx+=1
                
        loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv_to_k"] = loss_prompt_adv_to_k / len(network_modules)
        loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv_to_v"] = loss_prompt_adv_to_v / len(network_modules)   
        loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv"] = loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv_to_k"] + loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv_to_v"]

        # optim 
        loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv"].backward()

        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(trainable_params_adv, args.max_grad_norm, norm_type=2)
        optimizer_adv.step()
        lr_scheduler_adv.step()

        pbar_adv.set_description(f"Loss: {loss_adv[f'loss_adv_stage{stage}/loss_prompt_adv'].item():.4f}")
        
    return network, adv_prompts

def train(args: Arguments, prompts: list[PromptSettings]):
    model_metadata = {
        "prompts": ",".join([prompt.target for prompt in prompts]),
        "rank": str(args.cpe_network_rank),
        "alpha": str(args.cpe_network_alpha),
    }
    save_path = Path(args.save_dir)
    tokenizer, text_encoder, _, unet, _, _ = get_models(args)
    noise_scheduler: PNDMScheduler = PNDMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    device = get_devices(args)[0]
    text_encoder.to(device)
    text_encoder.eval()
    unet.to(device)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    anchor_sampler = AnchorSamplerGensim()
    task_id = len([])
    
    network = CPENetwork_ResAG(
        unet,
        rank=args.cpe_network_rank,
        multiplier=1.0,
        alpha=args.cpe_network_alpha,
        module=CPELayer_ResAG,
        continual=True,
        task_id=task_id,
        continual_rank=args.cpe_network_continual_rank,
        hidden_size=args.cpe_network_hidden_size,
        init_size=args.cpe_network_init_size,
    ).to(device)

    print("Prompts")
    for settings in prompts:
        print(settings)
    print("pal:", args.cpe_pal)

    prompt_scripts_path = args.cpe_prompt_scripts_path
    prompt_scripts_df = pd.read_csv(prompt_scripts_path)
    prompt_scripts_list: list[str] = prompt_scripts_df['prompt'].to_list()
    replace_word = args.cpe_replace_word
    prompt_scripts_list += [replace_word] * (1)

    # Compute lipschitz for weight matrices
    lipschitz_o = []
    lipschitz_q = []
    lipschitz_ov = []

    for name, module in unet.named_modules():
        if ("attn2" in name) and (module.__class__.__name__ == "Attention"):
            assert isinstance(module, Attention)
            mat_o = module.to_out[0].weight.detach()
            mat_q = module.to_q.weight.detach()
            mat_v = module.to_v.weight.detach()
            
            _, S_o, _ = torch.svd(mat_o)
            _, S_q, _ = torch.svd(mat_q)
            _, S_ov, _ = torch.svd(mat_o @ mat_v)
            
            lipschitz_o.append(S_o[0])
            lipschitz_q.append(S_q[0])
            lipschitz_ov.append(S_ov[0])
    lipschitz = {"lipschitz_o": lipschitz_o, "lipschitz_q": lipschitz_q, "lipschitz_ov": lipschitz_ov}
    # Compute lipschitz for weight matrices

    embedding_unconditional = get_condition([""], tokenizer, text_encoder)
    
    network_modules = dict()
    for name, module in network.named_modules():
        if "CPELayer" in module.__class__.__name__:
            network_modules[name] = module

    unet_modules = dict()
    for name, module in unet.named_modules():
        name = "_".join(name.split("."))
        name = f"lora_unet_{name}"

        for network_name in network_modules.keys():
            if name == network_name:
                unet_modules[name] = module
    
    _, num_tokens, token_dim = embedding_unconditional.size()    
    adv_prompts = PromptTuningLayer(
        args.cpe_num_add_prompts,
        num_tokens, token_dim, device
    ).to(device)
    
    # Prepare for erasing token cache
    prompt_one = prompts
    with torch.no_grad():
        prompt_in_scripts_target = []
        prompt_in_scripts_neutral = []
        
        for settings in prompt_one:
            for prompt_script in prompt_scripts_list:
                pr_in_script_tgt = prompt_script.replace(replace_word, settings.target)
                pr_in_script_tgt = pr_in_script_tgt.replace(replace_word.lower(), settings.target)

                pr_in_script_ntl = prompt_script.replace(replace_word, settings.neutral)
                pr_in_script_ntl = pr_in_script_ntl.replace(replace_word.lower(), settings.neutral)                
                

                prompt_in_scripts_target.append(pr_in_script_tgt)
                prompt_in_scripts_neutral.append(pr_in_script_ntl)
            
        embeddings_erase_tgt = get_condition(prompt_in_scripts_target, tokenizer, text_encoder).unsqueeze(1)
        embeddings_erase_ntl = get_condition(prompt_in_scripts_neutral, tokenizer, text_encoder).unsqueeze(1)
        embeddings_erase_cache = torch.cat([embeddings_erase_tgt, embeddings_erase_ntl], dim=1)
            
    simWords = []

    with torch.no_grad():
        if replace_word == "explicit":
            simWords_csv = pd.read_csv("../captions/cpe_surr_prompts_explicit.csv")
            simWords = list(simWords_csv.itertuples(index=False, name=None))
            simWords = [(a,b) for _,a,b in simWords]

            embeddings_anchor_cache = []
            prompt_in_scripts_anchor = [simWord[0] for simWord in simWords]
            embeddings_anchor_cache = get_condition(prompt_in_scripts_anchor, tokenizer, text_encoder)

        else:
            if replace_word == "target":
                simWords_csv = pd.read_csv("../captions/cpe_surr_words_character.csv")
            elif replace_word in ["actor", "artist"]:
                
                # prepare simWords
                if replace_word == "actor":
                    simWords_erase = [pr.target for pr in prompt_one]
                
                    simWords_general_words_csv = pd.read_csv("../captions/cpe_surr_words_actor_general_words.csv")
                    simWords_general_words = list(simWords_general_words_csv.itertuples(index=False, name=None))
                    simWords_general_words = [a for _,a,_ in simWords_general_words]
        
                    simWords_actor_anchor_csv = pd.read_csv("../captions/cpe_surr_words_500celebs.csv")
                    simWords_actor_anchor = list(simWords_actor_anchor_csv.itertuples(index=False, name=None))
                
                    simWords_actor_anchor = [a[0] for a in simWords_actor_anchor]

                elif replace_word == "artist":
                    simWords_erase = [pr.target for pr in prompt_one]
                
                    simWords_general_words_csv = pd.read_csv("../captions/cpe_surr_words_artist_general_words.csv")
                    simWords_general_words = list(simWords_general_words_csv.itertuples(index=False, name=None))
                    simWords_general_words = [a for _, a, _ in simWords_general_words]
        
                    simWords_actor_anchor_csv = pd.read_csv("../captions/cpe_surr_words_1734artists.csv")
                    simWords_actor_anchor = list(simWords_actor_anchor_csv.itertuples(index=False, name=None))
                    simWords_actor_anchor = [a[-1] for a in simWords_actor_anchor]
                    simWords_actor_anchor = list(set(simWords_actor_anchor))
                else:
                    ValueError("invalid replace word. use `target`, `actor`, or `artist`")

                for simW_erase in simWords_erase:            
                    simWords_actor_anchor = [item for item in simWords_actor_anchor if simW_erase not in item.lower()]

                len_actor_anchor = len(simWords_actor_anchor)
                anchor_batch = 100
    
                simWords_act_anc_batch = []
                for batch_idx in range(int(math.ceil(float(len_actor_anchor) / anchor_batch))):
                    if anchor_batch * (batch_idx + 1) <= len_actor_anchor:
                        simWords_act_anc_batch.append(simWords_actor_anchor[anchor_batch * batch_idx:anchor_batch * (batch_idx + 1)])
                    else:
                        simWords_act_anc_batch.append(simWords_actor_anchor[anchor_batch * batch_idx:])
                embeddings_erase = get_condition(simWords_erase, tokenizer, text_encoder)
    
                embeddings_actor_anchor = []
                for simW_batch in simWords_act_anc_batch:
                    emb_act_anc = get_condition(simW_batch, tokenizer, text_encoder)
                    embeddings_actor_anchor.append(emb_act_anc)
                embeddings_actor_anchor = torch.cat(embeddings_actor_anchor, dim=0)
                
                # compute similarity
                emb_erase_flat = embeddings_erase.view(len(simWords_erase), -1)
                emb_anchor_flat = embeddings_actor_anchor.view(len(simWords_actor_anchor), -1)
    
                emb_erase_flat_norm = emb_erase_flat / emb_erase_flat.norm(2, dim=1, keepdim=True)
                emb_anchor_flat_norm = emb_anchor_flat / emb_anchor_flat.norm(2, dim=1, keepdim=True)
    
                similarity = emb_erase_flat_norm @ emb_anchor_flat_norm.T

                # select anchor celebs
                _, ind_sorted = similarity.sort()
                ind_sorted_list = ind_sorted.cpu().numpy().tolist()
                simWords_selected = [simWords_actor_anchor[sim_idx] for sim_idx in ind_sorted_list[0][-50:]]
                simWords = simWords_general_words + simWords_selected
            
            embeddings_anchor_cache = []
            for simWord in simWords:
                prompt_in_scripts_anchor = []

                for prompt_script in prompt_scripts_list:
                    pr_in_script_anc = prompt_script.replace(replace_word, simWord)
                    pr_in_script_anc = pr_in_script_anc.replace(replace_word.lower(), simWord)        
                    prompt_in_scripts_anchor.append(pr_in_script_anc)

                embeddings_erase_anc = get_condition(prompt_in_scripts_anchor, tokenizer, text_encoder)
                embeddings_anchor_cache.append(embeddings_erase_anc)
            
            embeddings_anchor_cache = torch.cat(embeddings_anchor_cache, dim=0)
    
    train_iterations = args.cpe_iterations
    train_lr = args.cpe_lr
    train_lr_scheduler_num_cycles = args.cpe_lr_scheduler_num_cycles

    for stage in range(args.cpe_num_stages):
        print(f"Stage: {stage}\n")

        if stage == 0:
            args.cpe_iterations = args.cpe_factor_init_iter * train_iterations
            args.cpe_lr = args.cpe_factor_init_lr * train_lr
            args.cpe_lr_scheduler_num_cycles = args.cpe_factor_init_lr_cycle * train_lr_scheduler_num_cycles
        else:
            args.cpe_iterations = train_iterations
            args.cpe_lr = train_lr
            args.cpe_lr_scheduler_num_cycles = train_lr_scheduler_num_cycles

        trainable_params = network.prepare_optimizer_params(args.cpe_lr)

        pbar = trange(args.cpe_iterations)

        network.requires_grad_(True)
        network.train()
        adv_prompts.requires_grad_(False)
        adv_prompts.eval()
        
        optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.cpe_lr)
        lr_scheduler = get_scheduler_fix(
            optimizer, 
            iterations=args.cpe_iterations,
            lr_scheduler_num_cycles=args.cpe_lr_scheduler_num_cycles,
            lr_warmup_steps=args.cpe_lr_warmup_steps
        )
        criteria = torch.nn.MSELoss()
                
        network, adv_prompts = train_erase_one_stage(
            args=args,
            stage=stage,
            pbar=pbar,
            device_cuda=device,
            network=network,
            adv_prompts=adv_prompts,
            network_modules=network_modules,
            unet_modules=unet_modules,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criteria=criteria,
            prompts=prompts,
            embedding_unconditional=embedding_unconditional,
            anchor_sampler=anchor_sampler,
            lipschitz=lipschitz,
            embeddings_erase_cache=embeddings_erase_cache,
            embeddings_anchor_cache=embeddings_anchor_cache,
            trainable_params=trainable_params,
        )
        
        if args.cpe_do_adv_learn:
            adv_prompts.expand_prompts()
            
            pbar_adv = trange(args.cpe_adv_iters)
            network.requires_grad_(False)
            network.eval()
            adv_prompts.requires_grad_(True)
            adv_prompts.train()

            adv_parameters = adv_prompts.prompts.parameters()
            trainable_params_adv = [{"params": adv_parameters, "lr": args.cpe_adv_lr}]

            optimizer_adv = bnb.optim.AdamW8bit(
                trainable_params_adv,
                lr=args.cpe_lr
            )
            lr_scheduler_adv = get_scheduler_fix(
                optimizer_adv,
                iterations=args.cpe_iterations,
                lr_scheduler_num_cycles=args.cpe_lr_scheduler_num_cycles,
                lr_warmup_steps=args.cpe_lr_warmup_steps
            )
            criteria_adv = nn.MSELoss()        

            network, adv_prompts = train_adv_one_stage(
                args=args,
                stage=stage,
                pbar_adv=pbar_adv,
                network=network,
                adv_prompts=adv_prompts,
                network_modules=network_modules,
                unet_modules=unet_modules,
                optimizer_adv=optimizer_adv,
                lr_scheduler_adv=lr_scheduler_adv,
                criteria_adv=criteria_adv,
                prompts=prompts,
                embedding_unconditional=embedding_unconditional,
                lipschitz=lipschitz,
                embeddings_erase_cache=embeddings_erase_cache,
                trainable_params_adv=trainable_params_adv,
            )

    args.cpe_iterations = train_iterations
    args.cpe_lr = train_lr
    args.cpe_lr_scheduler_num_cycles = train_lr_scheduler_num_cycles        

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / "model_last.safetensors",
        metadata=model_metadata,
    )

    adv_prompts.save_weights(
        save_path / "model_adv_prompts_last.safetensors",
        metadata=model_metadata,
    )

    del (
        unet,
        noise_scheduler,
        optimizer,
        network,
    )


def main(args: Arguments):
    concepts = args.concepts.split(",")
    prompts=[
        PromptSettings(
            target=concept,
            positive=concept,
            unconditional="",
            action="erase_with_la",
            guidance_scale=1.0,
            resolution=args.image_size,
            batch_size=1,
            dynamic_resolution=True,
            la_strength=1000,
            sampling_batch_size=4
        )
        for concept in concepts
    ]

    base_path = args.save_dir

    for p_idx, p in enumerate(prompts):
        args.save_dir = base_path.replace(args.cpe_replace_word.upper(), p.target.replace(' ', '_'))

        if (p_idx < args.cpe_st_prompt_idx) or (p_idx > args.cpe_end_prompt_idx):
            continue

        Path(args.save_dir).mkdir(exist_ok=True)
        if args.cpe_skip_learned and Path(f"{args.save_dir}/model_last.safetensors").is_file():
            print(f"{p_idx} {p.target} has already been trained")
            continue

        print(p_idx, [p])
        seed_everything(args.seed)
        train(args, [p])
