# Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models (AdvUnlearn)

import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler

from train_methods.train_utils import id2embedding, soft_prompt_attack, get_train_loss_retain, apply_model, sample_until, encode_prompt, get_devices, tokenize
from train_methods.custom_text_encoder import CustomCLIPTextModel

from utils import Arguments

class PromptDataset:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.unseen_indices = list(self.data.index)

    def get_random_prompts(self, num_prompts=1):
        # Ensure that the number of prompts requested is not greater than the number of unseen prompts
        num_prompts = min(num_prompts, len(self.unseen_indices))

        # Randomly select num_prompts indices from the list of unseen indices
        selected_indices = random.sample(self.unseen_indices, num_prompts)
        
        # Remove the selected indices from the list of unseen indices
        for index in selected_indices:
            self.unseen_indices.remove(index)

        # return the prompts corresponding to the selected indices
        return self.data.loc[selected_indices, 'prompt'].tolist()

    def has_unseen_prompts(self):
        # check if there are any unseen prompts
        return len(self.unseen_indices) > 0
    
    def reset(self):
        self.unseen_indices = list(self.data.index)
        
    def check_unseen_prompt_count(self):
        return len(self.unseen_indices)

def retain_prompt(dataset_retain):
    # Prompt Dataset to be retained

    if dataset_retain == 'imagenet243':
        prompt_dataset_file_path = 'captions/imagenet243_retain.csv'
    elif dataset_retain == 'imagenet243_no_filter':
        prompt_dataset_file_path = 'captions/imagenet243_no_filter_retain.csv'
    elif dataset_retain == 'coco_object':
        prompt_dataset_file_path = 'captions/coco_object_retain.csv'
    elif dataset_retain == 'coco_object_no_filter':
        prompt_dataset_file_path = 'captions/coco_object_no_filter_retain.csv'
    else:
        raise ValueError('Invalid dataset for retaining prompts')
    
    return PromptDataset(prompt_dataset_file_path)

def param_choices(train_method: str, text_encoder: CustomCLIPTextModel=None, unet: UNet2DConditionModel=None, component='all', final_layer_norm=False):
    parameters = []
    
    # Text Encoder FUll Weight Tuning
    if train_method == 'text_encoder':
        for name, param in text_encoder.named_parameters():
            if name.startswith('text_model.final_layer_norm'): # Final Layer Norm
                if component == 'all' or final_layer_norm:
                    parameters.append(param)
            elif name.startswith('text_model.encoder'): # Transformer layers 
                if component == 'fc' and 'mlp' in name:
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    parameters.append(param)
                elif component == 'all':
                    parameters.append(param)
                
    # UNet Model Tuning
    else:
        for name, param in unet.named_parameters():
            # train all layers except x-attns and time_embed layers
            if train_method == 'noxattn':
                if not (name.startswith('out.') or 'attn2' in name or 'time_embed' in name):
                    parameters.append(param)
                    
            # train only self attention layers
            if train_method == 'selfattn':
                if 'attn1' in name:
                    parameters.append(param)
                    
            # train only x attention layers
            if train_method == 'xattn':
                if 'attn2' in name:
                    parameters.append(param)
                    
            # train all layers
            if train_method == 'full':
                parameters.append(param)
                
            # train all layers except time embed layers
            if train_method == 'notime':
                if not (name.startswith('out.') or 'time_embed' in name):
                    parameters.append(param)
            if train_method == 'xlayer':
                if 'attn2' in name:
                    if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                        parameters.append(param)
            if train_method == 'selflayer':
                if 'attn1' in name:
                    if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                        parameters.append(param)
    
    return parameters


def train(args: Arguments):
    
    devices = get_devices(args)
    
    # ====== Stage 0: PROMPT CLEANING ======
    prompt = args.concepts
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if args.seperator is not None:
        words = prompt.split(args.seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(f'The Concept Prompt to be unlearned:{words}')
    
    retain_dataset = retain_prompt(args.dataset_retain)
    
    # ======= Stage 1: TRAINING SETUP =======
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    unet_orig: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    text_encoder_orig: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    unet_orig: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")

    vae.eval()
    unet.to(devices[0])
    unet_orig.to(devices[1])
    unet_orig.eval()
    text_encoder_orig.eval()
    custom_text_encoder: CustomCLIPTextModel = CustomCLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder").to(devices[0])
    all_embeddings = custom_text_encoder.text_model.get_all_embedding().unsqueeze(0)
    scheduler.set_timesteps(args.ddim_steps, devices[1])

    quick_sample_till_t = lambda x, s, code, batch, t: sample_until(
        until=t,
        latents=code,
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=x,
        guidance_scale=s,
    )
    
    # Setup tainable model parameters
    if args.adv_method not in ['noxattn', 'selfattn', 'xattn', 'full', 'notime', 'xlayer', 'selflayer']:
        parameters = param_choices(text_encoder=custom_text_encoder, train_method=args.adv_method, component=args.component, final_layer_norm=args.norm_layer)
    else:
        parameters = param_choices(unet=unet, train_method=args.adv_method, component=args.component, final_layer_norm=args.norm_layer)
    
    losses = []
    opt = optim.Adam(parameters, lr=args.adv_lr)
    criteria = nn.MSELoss()
    history = []
    
    # ========== Stage 2: Training ==========
    pbar = trange(args.adv_iterations)
    global_step = 0
    attack_round = 0
    for i in pbar:
        # Change unlearned concept and obtain its corresponding adv embedding
        if i % args.adv_prompt_update_step == 0:
            
            # Reset the dataset if all prompts are seen           
            if retain_dataset.check_unseen_prompt_count() < args.adv_retain_batch:
                retain_dataset.reset()
            
            word = random.sample(words,1)[0]
            text_input = tokenizer(word, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True)
            text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
            
            prompt_embeds = encode_prompt(
                prompt=prompt, 
                removing_prompt=word,
                text_encoder=text_encoder_orig,
                tokenizer=tokenizer,
            )
            emb_0, emb_p, _ = torch.chunk(prompt_embeds, 3, dim=0)
    
            # ===== ** Attack ** : get adversarial prompt
            if i >= args.adv_warmup_iter:
                custom_text_encoder.eval()
                custom_text_encoder.requires_grad_(False)
                unet.eval()
                # args.adv_attack_embd_typeで処理が分岐されていたが呼び出している関数も引数も同じなので統一
                # 返り値の変数名だけ違うのでそこを揃える形に変更    
                if attack_round == 0:
                    attack_init_embd = None
                else:
                    attack_init_embd = adv_word_embd if args.adv_attack_embd_type == 'word_embd' else adv_condition_embd
                adv_word_embd, adv_input_ids = soft_prompt_attack(
                    word,
                    unet,
                    unet_orig,
                    tokenizer,
                    custom_text_encoder,
                    scheduler,
                    emb_0,
                    emb_p,
                    devices=devices,
                    criteria=criteria,
                    all_embeddings=all_embeddings,
                    args=args,
                    attack_init_embd=attack_init_embd,
                )
                if args.adv_attack_embd_type == 'condition_embd':
                    adv_condition_embd = adv_word_embd
                
                global_step += args.adv_attack_step
                attack_round += 1
        
        # Set model/TextEnocder to train or eval mode
        if args.adv_method == 'text_encoder':
            custom_text_encoder.train()
            custom_text_encoder.requires_grad_(True)
            unet.eval()
        else:
            custom_text_encoder.eval()
            custom_text_encoder.requires_grad_(False)
            unet.train()
        opt.zero_grad()
        
        # Retaining prompts for retaining regularized training
        if args.adv_retain_train == 'reg':
            retain_words = retain_dataset.get_random_prompts(args.adv_retain_batch)
            retain_text_input = tokenize(retain_words, tokenizer)
            retain_input_ids = retain_text_input.input_ids.to(devices[0])
            
            with torch.no_grad():
                retain_emb_p = text_encoder_orig(retain_text_input.input_ids.to(text_encoder_orig.device))[0]
            
            retain_text_embeddings = id2embedding(tokenizer, all_embeddings, retain_text_input.input_ids.to(devices[0]), devices[0])
            retain_text_embeddings = retain_text_embeddings.reshape(args.adv_retain_batch, -1, retain_text_embeddings.shape[-1])  # [batch, 77, 768]
            retain_emb_n = custom_text_encoder(input_ids=retain_input_ids, inputs_embeds=retain_text_embeddings)[0]
        else:
            retain_text_input = None
            retain_text_embeddings = None
            retain_emb_p = None
            retain_emb_n = None
        
        if i < args.adv_warmup_iter:
            # Warmup training
            input_ids = text_input.input_ids.to(devices[0])
            emb_n = custom_text_encoder(input_ids = input_ids, inputs_embeds=text_embeddings)[0]
            adv_embd = None
        else:
            if args.adv_attack_embd_type == 'word_embd':
                adv_embd = adv_word_embd
            elif args.adv_attack_embd_type == 'condition_embd':
                adv_embd = adv_condition_embd
            emb_n = None
        loss = get_train_loss_retain(
            args,
            unet,
            unet_orig,
            custom_text_encoder,
            scheduler,
            emb_0,
            emb_p,
            retain_emb_p,
            emb_n,
            retain_emb_n,
            devices,
            criteria,
            adv_input_ids,
            adv_embd=adv_embd
        )

        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        global_step += 1
        
        opt.step()
        
        if args.adv_retain_train == 'iter':
            for r in range(args.adv_retain_step):
                opt.zero_grad()
                if retain_dataset.check_unseen_prompt_count() < args.adv_retain_batch:
                    retain_dataset.reset()
                retain_words = retain_dataset.get_random_prompts(args.adv_retain_batch)
                
                t_enc = torch.randint(args.ddim_steps, (1,), device=devices[0])
                # time step from 1000 to 0 (0 being good)
                og_num = round((int(t_enc) / args.ddim_steps) * 1000)
                og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)
                t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
                retain_start_code = torch.randn((args.adv_retain_batch, 4, 64, 64)).to(devices[0])
                
                with torch.no_grad():
                    retain_text_input = tokenize(retain_words, tokenizer)
                    retain_emb_p = text_encoder_orig(retain_text_input.input_ids.to(text_encoder_orig.device))[0]
            
                retain_z = quick_sample_till_t(
                    torch.cat([emb_0, retain_emb_p], dim=0) if args.start_guidance > 1 else retain_emb_p,
                    args.start_guidance, retain_start_code, args.adv_retain_batch, int(t_enc)) # emb_p seems to work better instead of emb_0
                retain_e_p = apply_model(unet_orig, retain_z, t_enc_ddpm, retain_emb_p)
                
                retain_text_input = tokenize(retain_words, tokenizer)
                retain_input_ids = retain_text_input.input_ids.to(devices[0])
                retain_text_embeddings = id2embedding(tokenizer, all_embeddings, retain_text_input.input_ids.to(devices[0]), devices[0])
                retain_text_embeddings = retain_text_embeddings.reshape(args.adv_retain_batch, -1, retain_text_embeddings.shape[-1])  # [batch, 77, 768]
                retain_emb_n = custom_text_encoder(input_ids=retain_input_ids, inputs_embeds=retain_text_embeddings)[0]
                retain_e_n = apply_model(unet, retain_z, t_enc_ddpm, retain_emb_n)
                
                retain_loss: torch.Tensor = criteria(retain_e_n.to(devices[0]), retain_e_p.to(devices[0]))
                retain_loss.backward()
                opt.step()

    unet.eval()
    custom_text_encoder.eval()
    custom_text_encoder.requires_grad_(False)
    if args.adv_method == 'text_encoder':
        custom_text_encoder.save_pretrained(args.save_dir)
    else:
        unet.save_pretrained(args.save_dir)
        

def main(args: Arguments):
    train(args)

