# https://github.com/damilab/CARE/blob/main/utils/recare.py

import random
import torch
import numpy as np
import string

from pathlib import Path

import torch.nn.functional as F
import math
import time
import json
from diffusers.optimization import get_scheduler
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from train_methods.train_utils import get_models, gather_parameters, encode_prompt, predict_noise
from train_methods.data import TextualInversionDataset
from utils import Arguments


def generate_unique_placeholder_token(saved_tokens: dict, iteration: int) -> str:
    placeholder_token = "token_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    
    if placeholder_token in saved_tokens.values():
        return generate_unique_placeholder_token(saved_tokens, iteration)
    
    saved_tokens[f'{iteration}'] = placeholder_token
    return placeholder_token


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

def project(
    v0: torch.Tensor,  
    v1: torch.Tensor,  
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,   
    pred_uncond: torch.Tensor, 
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
) -> torch.Tensor:
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond - (guidance_scale - 1) * normalized_update 
    return pred_guided


def normalized_compositional_guidance(
    pred_conds: list[torch.Tensor],     
    pred_uncond: torch.Tensor,          
    guidance_scales: list[float],        
    momentum_buffers: list[MomentumBuffer] | None = None, 
    eta: float = 1.0,
    norm_threshold: float = 0.0,
) -> torch.Tensor:
    positive_update = torch.zeros_like(pred_uncond)
    negative_update = torch.zeros_like(pred_uncond)
    pos_count, neg_count = 0, 0

    for i, (pred_cond, guidance_scale) in enumerate(zip(pred_conds, guidance_scales)):
        diff = pred_cond - pred_uncond
        if momentum_buffers and momentum_buffers[i] is not None:
            momentum_buffers[i].update(diff)
            diff = momentum_buffers[i].running_average

        if norm_threshold > 0:
            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
            diff = diff * scale_factor

        diff_parallel, diff_orthogonal = project(diff, pred_cond)
        normalized_update = diff_orthogonal + eta * diff_parallel

        if guidance_scale > 0:
            positive_update += (guidance_scale - 1) * normalized_update
            pos_count += 1
        else:
            negative_update += (guidance_scale - 1) * normalized_update
            neg_count += 1

    if pos_count > 0:
        positive_update /= pos_count
    if neg_count > 0:
        negative_update /= neg_count

    compositional_update = positive_update + negative_update

    pred_guided = pred_uncond + compositional_update
    return pred_guided


def train_erasing(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    erase_concept: str,
    erase_from: str,
    train_method,
    iterations: int,
    negative_guidance,
    lr: float,
    save_path: str,
):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)        
    torch.manual_seed(seed) 

    nsteps = 50
    img_size = 512
    batch_size = 1

    _, parameters = gather_parameters(train_method, unet)

    optimizer = torch.optim.AdamW(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = trange(iterations)
    erase_concept = [a.strip() for a in erase_concept.split(',')]
    erase_from = [a.strip() for a in erase_from.split(',')]

    if len(erase_from) != len(erase_concept):
        if len(erase_from) == 1:
            erase_from = [erase_from[0]] * len(erase_concept)
        else:
            raise ValueError("Erase concepts and target concepts must have matching lengths.")

    erase_concept = [[e, f] for e, f in zip(erase_concept, erase_from)]

    torch.cuda.empty_cache()

    for _ in pbar:
        with torch.no_grad():
            index = np.random.choice(len(erase_concept), 1, replace=False)[0]
            erase_concept_sampled = erase_concept[index]

            neutral_text_embeddings = encode_prompt([''], text_encoder=text_encoder, tokenizer=tokenizer)
            positive_text_embeddings = encode_prompt([erase_concept_sampled[0]], text_encoder=text_encoder, tokenizer=tokenizer)
            target_text_embeddings = encode_prompt([erase_concept_sampled[1]], text_encoder=text_encoder, tokenizer=tokenizer)

            scheduler.set_scheduler_timesteps(nsteps, unet.deveice)
            optimizer.zero_grad()
            iteration = torch.randint(1, nsteps - 1, (1,)).item()

            noise = torch.randn(
                (batch_size, unet.in_channels, img_size // 8, img_size // 8)
            ).type(unet.dtype).to(unet.device).repeat(1, 1, 1, 1)
            latents = noise * scheduler.init_noise_sigma

        latents_steps = []
        for i in range(iteration):
            with torch.no_grad():
                noise_pred = predict_noise(
                    unet,
                    scheduler,
                    i,
                    latents,
                    positive_text_embeddings,
                    guidance_scale=3,
                )
                output = scheduler.step(noise_pred, scheduler.timesteps[i], latents)
                latents = output.prev_sample

                if i == iteration - 1:
                    latents_steps.append(output)

        with torch.no_grad():
            scheduler.set_scheduler_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)
            positive_latents = predict_noise(unet, scheduler, iteration, latents_steps[0], positive_text_embeddings)
            neutral_latents = predict_noise(unet, scheduler, iteration, latents_steps[0], neutral_text_embeddings)
            _ = predict_noise(unet, scheduler, iteration, latents_steps[0], target_text_embeddings)

            torch.cuda.empty_cache()

            if erase_concept_sampled[0] == erase_concept_sampled[1]:
                _ = neutral_latents.clone().detach()
        
        negative_latents = predict_noise(unet, scheduler, iteration, latents_steps[0], target_text_embeddings)
     
        pred_neg_guidance = normalized_guidance(positive_latents, neutral_latents, negative_guidance)
        loss: torch.Tensor = criteria(negative_latents, pred_neg_guidance)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pbar.set_description(f"Loss: {loss.item():.4f}")


    unet.save_pretrained(save_path)

    torch.cuda.empty_cache()


def train_concept_inversion(
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    vae: AutoencoderKL,
    scheduler: DDIMScheduler,
    unet: UNet2DConditionModel,
    placeholder_token, 
    initializer_token, 
    train_data_dir, 
    lr, 
    save_path, 
    device, 
    num_vectors=1, 
    max_train_steps=3000,  
    resolution=512, 
    learnable_property="object",
    lr_scheduler="constant", 
    lr_warmup_steps=0, 
    scale_lr=False,  
    iteration=None,
    num_iterations=None,
    center_crop=False
):
    
    seed = 42
    np.random.seed(seed)     
    random.seed(seed)        
    torch.manual_seed(seed) 

    vae.eval()
    unet.eval()

    for param in text_encoder.embeddings.token_embedding.parameters():
        param.requires_grad = True

    placeholder_tokens = [placeholder_token]
    additional_tokens = [f"{placeholder_token}_{i}" for i in range(1, num_vectors)]
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != num_vectors:
        raise ValueError(f"Token '{placeholder_token}' already exists in tokenizer. Use a different token name.")

    initializer_token_id = tokenizer.convert_tokens_to_ids([initializer_token])[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    text_encoder.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        token_embeds = text_encoder.get_input_embeddings().weight.data
        ctr = 0
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
            ctr += 1
        print(f"Initialized {ctr} placeholder token embeddings with '{initializer_token}' token embeddings.")

    org_token_embeds = text_encoder.get_input_embeddings().weight.data.clone()
    

    dataset = TextualInversionDataset(
        data_root=train_data_dir,
        tokenizer=tokenizer,
        size=resolution,
        placeholder_token=" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)),
        repeats=100,
        set="train",
        learnable_property=learnable_property,
        center_crop=center_crop,
        iteration=iteration,
        num_iterations=num_iterations
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    steps_per_epoch = len(dataloader)
    num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)

    if scale_lr:
        effective_batch_size = dataloader.batch_size
        lr *= effective_batch_size 

    optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=lr)
    scheduler = get_scheduler(lr_scheduler, optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=max_train_steps)

    progress_bar = tqdm(total=max_train_steps, desc="Textual Inversion Progress", unit="step")
    global_step = 0

    for _ in range(num_train_epochs):
        text_encoder.train()
        
        for _, batch in enumerate(dataloader):
            if global_step >= max_train_steps:
                break

            optimizer.zero_grad()

            latents: torch.Tensor = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 999, (latents.shape[0],), device=latents.device)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"].to(device)).last_hidden_state
            model_pred: torch.Tensor = unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]

            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            scheduler.step()

            index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool, device=device)
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False  # False indicates trainable embeddings

            with torch.no_grad():
                text_encoder.get_input_embeddings().weight.data[index_no_updates] = org_token_embeds[index_no_updates]
            
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            global_step += 1

    progress_bar.close()
    text_encoder.eval()
    text_encoder.save_pretrained(save_path)


def iterative_textual_inversion(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    vae: AutoencoderKL,
    initial_erase_concept,
    initializer_token,
    train_data_dir,
    train_method,
    lr,
    ti_lr,
    negative_guidance,
    iterations,
    n_iterations: int,
    device,
    ti_max_train_steps,
    learnable_property,
    output_dir: str,
    center_crop=False
) -> dict[str, str]:
    current_concept = initial_erase_concept
    saved_tokens = {}

    Path(output_dir).mkdir(exist_ok=True)
    
    for iteration in trange(n_iterations):
        placeholder_token = generate_unique_placeholder_token(saved_tokens, iteration)
        saved_tokens[f'{iteration}'] = placeholder_token

        erased_weights_path = Path(output_dir) / f"erased_unet_iteration_{iteration}.pt"
        ti_encoder_path = Path(output_dir) / f"ti_text_encoder_iteration_{iteration}.pt"

        print(f"Erasing concept: {current_concept} -> Placeholder token: '{placeholder_token}' (initialized from '{initializer_token}')")

        train_erasing(
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            erase_concept=current_concept,
            erase_from=current_concept,
            train_method=train_method,
            iterations=iterations,
            negative_guidance=negative_guidance,
            lr=lr,
            save_path=erased_weights_path,
            device=device
        )
        print(f"Erased weights saved to {erased_weights_path}")

        unet.from_pretrained(erased_weights_path).to(device)

        train_concept_inversion(
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            placeholder_token=placeholder_token,
            initializer_token=initializer_token,
            train_data_dir=train_data_dir,
            lr=ti_lr,
            save_path=ti_encoder_path,
            device=device,
            max_train_steps=ti_max_train_steps,
            learnable_property=learnable_property,
            scale_lr=True, 
            iteration=iteration,
            num_iterations=n_iterations,
            center_crop=center_crop
        )
        print(f"Text encoder with placeholder '{placeholder_token}' saved to {ti_encoder_path}")

        current_concept = placeholder_token
        torch.cuda.empty_cache()
        text_encoder.from_pretrained(ti_encoder_path).to(device)


    final_model_path = Path(output_dir) / "recare_stage1"
    text_encoder.save_pretrained(final_model_path / "text_encoder")
    unet.save_pretrained(final_model_path / "unet")
    print(f"\nIterative stage complete. Final model saved to {final_model_path}")
    print(f"Placeholder tokens: {saved_tokens}")

    return saved_tokens


def robust_erase_for_care(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    erase_concepts: list[str],
    train_method: str,
    iterations: int,
    compositional_guidance_scale: float,
    lr: float,
    save_path: str,
    anchor_concepts_path: str
):
    nsteps = 50

    with open(anchor_concepts_path, "r", encoding="utf-8") as f:
        data: dict[str, list | str] = json.load(f)

    anchor_keys = []
    retain_keys = []

    for k, v in data.items():
        if not isinstance(v, list):
            continue

        k_low = k.lower()
        if ("photo" in k_low) or ("painting" in k_low):
            retain_keys.append(k)
        else:
            anchor_keys.append(k)

    if len(retain_keys) != 1:
        raise KeyError(f"[CARE JSON] Expected exactly 1 retain key containing 'photo' or 'painting', got: {retain_keys}")
    if len(anchor_keys) != 1:
        raise KeyError(f"[CARE JSON] Expected exactly 1 anchor key (non-photo/painting), got: {anchor_keys}")

    retain_key = retain_keys[0]
    anchor_key = anchor_keys[0]

    all_anchor_concepts = data[anchor_key]
    all_retain_concepts = data[retain_key]


    parameters = gather_parameters(train_method, unet)

    optimizer = torch.optim.AdamW(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    
    torch.cuda.empty_cache()

    # erase loss
    total_sentences = len(all_anchor_concepts)
    appearances_per_sentence = iterations // total_sentences
    balanced_list = all_anchor_concepts * appearances_per_sentence
    np.random.shuffle(balanced_list)
    remainder = iterations - len(balanced_list)
    if remainder > 0:
        balanced_list.extend(np.random.choice(all_anchor_concepts, remainder, replace=False))

    #retain loss
    total_retain_sentences = len(all_retain_concepts)
    appearances_per_sentence = iterations // total_retain_sentences
    balanced_retain_list = all_retain_concepts * appearances_per_sentence
    np.random.shuffle(balanced_retain_list)
    remainder = iterations - len(balanced_retain_list)
    if remainder > 0:
        balanced_retain_list.extend(np.random.choice(all_retain_concepts, remainder, replace=False))

    # erase concepts
    total_concepts = len(erase_concepts)
    appearances_per_concept = iterations // total_concepts
    balanced_erase_list = erase_concepts * appearances_per_concept
    np.random.shuffle(balanced_erase_list)
    remainder = iterations - len(balanced_erase_list)
    if remainder > 0:
        balanced_erase_list.extend(np.random.choice(erase_concepts, remainder, replace=False))

    pbar = tqdm(range(iterations))
    for i in pbar:
        with torch.no_grad():
            erase_concept_sampled = balanced_erase_list[i]
            anchor_concepts = [balanced_list[i]]
            retain_concepts = [balanced_retain_list[i]]

            print(f"Erasing concept: {erase_concept_sampled} from anchor concept: {anchor_concepts} with retain concept {retain_concepts} at iteration {i}")

            neutral_text_embeddings = encode_prompt([''], text_encoder=text_encoder, tokenizer=tokenizer)
            target_text_embeddings = encode_prompt([erase_concept_sampled], text_encoder=text_encoder, tokenizer=tokenizer)
            retain_text_embeddings = encode_prompt(retain_concepts, text_encoder=text_encoder, tokenizer=tokenizer)

            negative_word_embs = []
            for neg_word in erase_concepts:
                negative_word_embs.append(encode_prompt([neg_word], text_encoder=text_encoder, tokenizer=tokenizer))

            anchor_word_embs = []
            for anchor_word in anchor_concepts:
                anchor_word_embs.append(encode_prompt([anchor_word], text_encoder=text_encoder, tokenizer=tokenizer))
            
            retain_word_embs = []
            for retain_word in retain_concepts:
                retain_word_embs.append(encode_prompt([retain_word], text_encoder=text_encoder, tokenizer=tokenizer))

            scheduler.set_scheduler_timesteps(nsteps)
            optimizer.zero_grad()
            iteration = torch.randint(1, nsteps - 1, (1,)).item()
            latents = scheduler.get_initial_latents(1, 512, 1)

            latents_steps = []
            for i in range(iteration):
                noise_pred = predict_noise(
                    unet,
                    scheduler,
                    i,
                    latents,
                    target_text_embeddings,
                    guidance_scale=3,
                )
                output = scheduler.step(noise_pred, scheduler.timesteps[i], latents)
                latents = output.prev_sample

                if i == iteration - 1:
                    latents_steps.append(output)


            scheduler.set_scheduler_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)

            neutral_latents = predict_noise(unet, scheduler, iteration, latents_steps[0], neutral_text_embeddings)
            _ = predict_noise(unet, scheduler, iteration, latents_steps[0], target_text_embeddings)

            e_negatives_latents = []
            for emb_neg in negative_word_embs:
                e_negatives_latents.append(predict_noise(unet, scheduler, iteration, latents_steps[0], emb_neg))

            e_anchor_latents = []
            for emb_anchor in anchor_word_embs:
                e_anchor_latents.append(predict_noise(unet, scheduler, iteration, latents_steps[0], emb_anchor))
            
            e_retain_latents = []
            for emb_retain in retain_word_embs:
                e_retain_latents.append(predict_noise(unet, scheduler, iteration, latents_steps[0], emb_retain))\

            torch.cuda.empty_cache()

        negative_latents = predict_noise(unet, scheduler, iteration, latents_steps[0], target_text_embeddings)
        retain_latents = predict_noise(unet, scheduler, iteration, latents_steps[0], retain_text_embeddings)

        neg_guidance_scales = []
        for _ in range(len(e_negatives_latents)):
            neg_guidance_scales.append(-compositional_guidance_scale)

        pos_guidance_scales = []
        for _ in range(len(e_anchor_latents)):
            pos_guidance_scales.append(compositional_guidance_scale)

        combined_conditions = e_negatives_latents + e_anchor_latents
        combined_guidance_scales = neg_guidance_scales + pos_guidance_scales

        compositional_guidance_estimate = normalized_compositional_guidance(combined_conditions, neutral_latents, combined_guidance_scales)
        erase_loss: torch.Tensor = criteria(negative_latents, compositional_guidance_estimate)
        retain_loss: torch.Tensor = criteria(retain_latents, e_retain_latents[0])
        # total loss
        loss = erase_loss + retain_loss

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pbar.set_description(f"Total Loss: {loss.item():.4f} | Erase: {erase_loss.item():.4f} | Retain: {retain_loss.item():.4f}")

    unet.save_pretrained(save_path)


def recare(args: Arguments):

    tokenizer, text_encoder, vae, unet, scheduler, _ = get_models(args.sd_version)
    copy_tokenizer, copy_text_encoder, _, copy_unet, _, _ = get_models(args.sd_version)

    iti_start_time = time.time()
    print(f"===== Iterative Textual Inversion =====")
    saved_tokens: dict[str, str] = iterative_textual_inversion(
        unet=unet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        initial_erase_concept=args.concepts,
        initializer_token=args.care_initializer_token,
        train_data_dir=args.care_train_data_dir,
        train_method=args.care_method,
        lr=args.care_recare_stage1_lr,
        ti_lr=args.care_ti_lr,
        negative_guidance=args.negative_guidance,
        iterations=args.care_iterations,
        n_iterations=args.care_n_iterations,
        device=args.device,
        ti_max_train_steps=args.care_ti_max_train_steps,
        learnable_property=args.care_learnable_property,
        output_dir=args.save_dir,
        generic_prompt=args.care_generic_prompt,
        center_crop=args.center_crop
    )
    iti_end_time = time.time()
    print(f"iterative_textual_inversion : {iti_end_time - iti_start_time} seconds\n")

    print(f"===== Robust Erase For CARE =====")
    final_unet_path = Path(args.save_dir) / "ReCARE-Diffusers-UNet.pt"

    for token in list(saved_tokens.values()):
        if token not in tokenizer.get_vocab():
            print(f"!!!! Adding placeholder token '{token}' to tokenizer.")
            tokenizer.add_tokens([token])
            copy_tokenizer.add_tokens([token])
            text_encoder.resize_token_embeddings(len(tokenizer))
            copy_text_encoder.resize_token_embeddings(len(tokenizer))
    
    copy_unet.from_pretrained(Path(args.save_dir) / "recare_stage1" / "unet")
    copy_text_encoder.from_pretrained(Path(args.save_dir) / "recare_stage1" / "text_encoder")
    text_encoder.load_state_dict(copy_text_encoder.state_dict())
    tokenizer.save_pretrained(Path(args.save_dir) / "recare_stage1" / "tokenizer")

    final_concepts_to_erase = [args.concepts]
    adv_tokens_from_iti = list(saved_tokens.values())
    if not args.care_num_of_adv_concepts == 0:
        adv_tokens_from_iti = adv_tokens_from_iti[0:args.care_num_of_adv_concepts]
        final_concepts_to_erase.extend(adv_tokens_from_iti)
    
    recare_start_time = time.time()
    robust_erase_for_care(
        unet=unet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        erase_concepts=final_concepts_to_erase,
        train_method=args.care_method,
        iterations=args.care_iterations,
        compositional_guidance_scale=args.care_compositional_guidance_scale,
        lr=args.care_recare_stage2_lr,
        save_path=final_unet_path,
        anchor_concepts_path=args.care_anchor_concept_path
    )
    recare_end_time = time.time()
    print(f"Final ReCARE unet saved to {final_unet_path}")
    print(f"========= Robust Erase For CARE : {recare_end_time - recare_start_time} seconds =========")

def main(args: Arguments):
    recare(args)
