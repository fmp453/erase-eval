# Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models (AdvUnlearn)

import random
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler

from train_methods.train_utils import id2embedding, soft_prompt_attack, get_train_loss_retain, apply_model, sample_until
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

@torch.no_grad()
def encode_prompt(
    prompt: Union[str, list[str]]=None,
    negative_prompt: Union[str, list[str]]=None,
    removing_prompt: Union[str, list[str]]=None,
    num_images_per_prompt: int=1,
    text_encoder: CLIPTextModel=None,
    tokenizer: CLIPTokenizer=None,
    device: torch.device=None,
):
    """Encode a prompt into a text embedding. Prompt can be None."""
    # Get text embeddings for unconditional and conditional prompts.
    if isinstance(prompt, str):
        prompt = [prompt]
    
    if removing_prompt is not None and isinstance(removing_prompt, str):
        removing_prompt = [removing_prompt]
        assert len(prompt) == len(removing_prompt), f"Safety concept must be the same length as prompt of length {len(prompt)}."
    
    if negative_prompt is not None and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]
        assert len(prompt) == len(negative_prompt), f"Negative prompt must be the same length as prompt of length {len(prompt)}."

    batch_size = len(prompt) if prompt is not None else 1

    use_attention_mask = hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask
    device = device if device is not None else text_encoder.device

    # Tokenization
    uncond_input = tokenizer([""] * batch_size if negative_prompt is None else negative_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    if prompt is not None:
        prompt_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    else:
        prompt_input = None
    
    if removing_prompt is not None:
        removing_input = tokenizer(removing_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    else:
        removing_input = None

    # Encoding
    prompt_embeds = text_encoder(input_ids=uncond_input["input_ids"].to(device), attention_mask=uncond_input["attention_mask"].to(device) if use_attention_mask else None)[0]
    if prompt_input is not None:
        prompt_emb = text_encoder(input_ids=prompt_input["input_ids"].to(device), attention_mask=prompt_input["attention_mask"].to(device) if use_attention_mask else None)[0]
        prompt_embeds = torch.cat([prompt_embeds, prompt_emb], dim=0)
    
    if removing_input is not None:
        removing_emb = text_encoder(input_ids=removing_input["input_ids"].to(device), attention_mask=removing_input["attention_mask"].to(device) if use_attention_mask else None)[0]
        prompt_embeds = torch.cat([prompt_embeds, removing_emb], dim=0)

    # Duplicate the embeddings for each image.
    if num_images_per_prompt > 1:
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.reshape(batch_size * num_images_per_prompt, seq_len, -1)
    
    return prompt_embeds

def train(args: Arguments):
    
    devices = args.device.split(",")
    if len(devices) > 1:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[1]}")]
    else:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[0]}")]
    
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
    ddim_eta = 0
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
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
                if attack_round == 0:
                    if args.adv_attack_embd_type == 'word_embd':
                        adv_word_embd, adv_input_ids = soft_prompt_attack(word, unet, unet_orig, tokenizer, custom_text_encoder, scheduler, emb_0, emb_p, args.start_guidance, devices, args.ddim_steps, criteria, args.adv_prompt_num, all_embeddings, args.adv_attack_type,  args.adv_attack_embd_type, args.adv_attack_step, args.adv_attack_lr, args.adv_attack_init, None, args.adv_attack_method)
                    elif args.adv_attack_embd_type == 'condition_embd':
                        adv_condition_embd, adv_input_ids = soft_prompt_attack(word, unet, unet_orig, tokenizer, custom_text_encoder, scheduler, emb_0, emb_p, args.start_guidance, devices, args.ddim_steps, criteria, args.adv_prompt_num, all_embeddings, args.adv_attack_type, args.adv_attack_embd_type, args.adv_attack_step, args.adv_attack_lr, args.adv_attack_init, None, args.adv_attack_method) 
                else:
                    if args.adv_attack_embd_type == 'word_embd':
                        adv_word_embd, adv_input_ids = soft_prompt_attack(word, unet, unet_orig, tokenizer, custom_text_encoder, scheduler, emb_0, emb_p, args.start_guidance, devices, args.ddim_steps, criteria, args.adv_prompt_num, all_embeddings, args.adv_attack_type,  args.adv_attack_embd_type, args.adv_attack_step, args.adv_attack_lr, args.adv_attack_init, adv_word_embd, args.adv_attack_method)
                    elif args.adv_attack_embd_type == 'condition_embd':
                        adv_condition_embd, adv_input_ids = soft_prompt_attack(word, unet, unet_orig, tokenizer, custom_text_encoder, scheduler, emb_0, emb_p, args.start_guidance, devices, args.ddim_steps, criteria, args.adv_prompt_num, all_embeddings, args.adv_attack_type, args.adv_attack_embd_type, args.adv_attack_step, args.adv_attack_lr, args.adv_attack_init, adv_condition_embd, args.adv_attack_method) 
                
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
            retain_text_input = tokenizer(retain_words, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True)
            retain_input_ids = retain_text_input.input_ids.to(devices[0])
            
            with torch.no_grad():
                retain_emb_p = text_encoder_orig(retain_text_input.input_ids.to(text_encoder_orig.device))[0]
            
            retain_text_embeddings = id2embedding(tokenizer, all_embeddings, retain_text_input.input_ids.to(devices[0]), devices[0])
            retain_text_embeddings = retain_text_embeddings.reshape(args.adv_retain_batch, -1, retain_text_embeddings.shape[-1])  # [batch, 77, 768]
            retain_emb_n = custom_text_encoder(input_ids = retain_input_ids, inputs_embeds=retain_text_embeddings)[0]
        else:
            retain_text_input = None
            retain_text_embeddings = None
            # retain_emb_0 = None
            retain_emb_p = None
            retain_emb_n = None
        
        if i < args.adv_warmup_iter:
            # Warmup training
            input_ids = text_input.input_ids.to(devices[0])
            emb_n = custom_text_encoder(input_ids = input_ids, inputs_embeds=text_embeddings)[0]
            loss = get_train_loss_retain(args.adv_retain_batch, args.adv_retain_train, args.adv_retain_loss_w, unet, unet_orig, custom_text_encoder, scheduler, emb_0, emb_p, retain_emb_p, emb_n, retain_emb_n, args.start_guidance, args.negative_guidance, devices, args.ddim_steps, criteria, input_ids, args.adv_attack_embd_type)
        else:
            if args.adv_attack_embd_type == 'word_embd':
                loss = get_train_loss_retain(args.adv_retain_batch, args.adv_retain_train, args.adv_retain_loss_w, unet, unet_orig, custom_text_encoder, scheduler, emb_0, emb_p, retain_emb_p, None, retain_emb_n, args.start_guidance, args.negative_guidance, devices, args.ddim_steps, criteria, adv_input_ids, args.adv_attack_embd_type, adv_word_embd)
            elif args.adv_attack_embd_type == 'condition_embd':
                loss = get_train_loss_retain(args.adv_retain_batch, args.adv_retain_train, args.adv_retain_loss_w, unet, unet_orig, custom_text_encoder, scheduler, emb_0, emb_p, retain_emb_p, None, retain_emb_n, args.start_guidance, args.negative_guidance, devices, args.ddim_steps, criteria, adv_input_ids, args.adv_attack_embd_type, adv_condition_embd)
        
        # update weights to erase the concept
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
                
                # retain_emb_p = model_orig.get_learned_conditioning(retain_words)
                with torch.no_grad():
                    retain_text_input = tokenizer(retain_words, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", truncation=True)
                    retain_emb_p = text_encoder_orig(retain_text_input.input_ids.to(text_encoder_orig.device))[0]
            
                retain_z = quick_sample_till_t(
                    torch.cat([emb_0, retain_emb_p], dim=0) if args.start_guidance > 1 else retain_emb_p,
                    args.start_guidance, retain_start_code, args.adv_retain_batch, int(t_enc)) # emb_p seems to work better instead of emb_0
                retain_e_p = apply_model(unet_orig, retain_z, t_enc_ddpm, retain_emb_p)
                
                retain_text_input = tokenizer(retain_words, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt",truncation=True)
                retain_input_ids = retain_text_input.input_ids.to(devices[0])
                retain_text_embeddings = id2embedding(tokenizer, all_embeddings, retain_text_input.input_ids.to(devices[0]), devices[0])
                retain_text_embeddings = retain_text_embeddings.reshape(args.adv_retain_batch, -1, retain_text_embeddings.shape[-1])  # [batch, 77, 768]
                retain_emb_n = custom_text_encoder(input_ids = retain_input_ids, inputs_embeds=retain_text_embeddings)[0]
                retain_e_n = apply_model(unet, retain_z, t_enc_ddpm, retain_emb_n)
                
                retain_loss = criteria(retain_e_n.to(devices[0]), retain_e_p.to(devices[0]))
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

