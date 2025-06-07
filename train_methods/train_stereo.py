# https://github.com/koushiksrivats/robust-concept-erasing/blob/main

import argparse
import os
import random
import time
import string
import math
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from diffusers import LMSDiscreteScheduler, DDIMScheduler, AutoencoderKL, UNet2DConditionModel, get_scheduler

from train_methods.data import TextualInversionDataset
from train_methods.train_utils import get_models, get_devices, get_condition, gather_parameters, predict_noise
from utils import Arguments


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

def project(
    v0: torch.Tensor,  # [B, C, H, W]
    v1: torch.Tensor,  # [B, C, H, W]
) -> torch.Tensor:
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def normalized_guidance(
    pred_cond: torch.Tensor,   # [B, C, H, W]
    pred_uncond: torch.Tensor, # [B, C, H, W]
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
    pred_guided = pred_cond - (guidance_scale - 1) * normalized_update # For negative guidance
    return pred_guided


def normalized_compositional_guidance(
    pred_conds: list[torch.Tensor], # List of [B, C, H, W] conditional predictions
    pred_uncond: torch.Tensor,  # [B, C, H, W] unconditional prediction
    guidance_scales: list[float],  # List of guidance scales for each condition (can be + or -)
    momentum_buffers: Optional[list[MomentumBuffer]] = None, # List of MomentumBuffers for each condition
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    positive_update = torch.zeros_like(pred_uncond)
    negative_update = torch.zeros_like(pred_uncond)
    pos_count, neg_count = 0, 0

    # Separate processing for positive and negative guidance scales
    for i, (pred_cond, guidance_scale) in enumerate(zip(pred_conds, guidance_scales)):
        # Calculate difference for each condition
        diff = pred_cond - pred_uncond
        if momentum_buffers and momentum_buffers[i] is not None:
            momentum_buffers[i].update(diff)
            diff = momentum_buffers[i].running_average

        # Apply normalization threshold if specified
        if norm_threshold > 0:
            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
            diff = diff * scale_factor

        # Compute parallel and orthogonal components
        diff_parallel, diff_orthogonal = project(diff, pred_cond)
        normalized_update = diff_orthogonal + eta * diff_parallel

        # Accumulate in either positive or negative update
        if guidance_scale > 0:
            positive_update += (guidance_scale - 1) * normalized_update
            pos_count += 1
        else:
            negative_update += (guidance_scale - 1) * normalized_update
            neg_count += 1

    # Normalize each set if counts are non-zero
    if pos_count > 0:
        positive_update /= pos_count
    if neg_count > 0:
        negative_update /= neg_count

    return pred_uncond + positive_update + negative_update

# ESDのdiffusers実装と同じ
# StableDiffuserの解体が必要
# unetのみ更新するのでunetを返す
def train_erasing(
    args: Arguments,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    noise_scheduler: DDIMScheduler,
    erase_concept: str,
    erase_from: str,
    save_path,
) -> UNet2DConditionModel:
    # Set the random seed for reproducibility
    seed = args.seed
    np.random.seed(seed)      # For numpy
    random.seed(seed)         # For the random module
    torch.manual_seed(seed) 

    nsteps = 50

    # finetuner = FineTunedModel(diffuser, train_method=train_method)
    parameters = gather_parameters(args.stereo_method, unet)

    optimizer = optim.AdamW(parameters, lr=args.stereo_lr)
    criteria = nn.MSELoss()

    pbar = tqdm(range(args.stereo_iterations))
    erase_concept = [a.strip() for a in erase_concept.split(',')]
    erase_from = [a.strip() for a in erase_from.split(',')]

    if len(erase_from) != len(erase_concept):
        if len(erase_from) == 1:
            erase_from = [erase_from[0]] * len(erase_concept)
        else:
            raise ValueError("Erase concepts and target concepts must have matching lengths.")

    erase_concept = [[e, f] for e, f in zip(erase_concept, erase_from)]

    torch.cuda.empty_cache()

    for i in pbar:
        with torch.no_grad():
            index = np.random.choice(len(erase_concept), 1, replace=False)[0]
            erase_concept_sampled = erase_concept[index]

            neutral_text_embeddings = get_condition([""], tokenizer, text_encoder)
            positive_text_embeddings = get_condition([erase_concept_sampled[0]], tokenizer, text_encoder)
            target_text_embeddings = get_condition([erase_concept_sampled[1]], tokenizer, text_encoder)

            noise_scheduler.set_timesteps(nsteps, unet.device)
            optimizer.zero_grad()
            iteration = torch.randint(1, nsteps - 1, (1,)).item()
            latents = torch.randn(1, unet.config.in_channels, args.image_size // 8, args.image_size // 8).to(unet.device).repeat(1, 1, 1, 1) * noise_scheduler.init_noise_sigma

            with finetuner:
                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=3, 
                    show_progress=False
                )

            noise_scheduler.set_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)
            timestep = noise_scheduler.timesteps[iteration]
            positive_latents = predict_noise(unet, noise_scheduler, timestep, latents, positive_text_embeddings, guidance_scale=1)
            neutral_latents = predict_noise(unet, noise_scheduler, timestep, latents, neutral_text_embeddings, guidance_scale=1)
            target_latents = predict_noise(unet, noise_scheduler, timestep, latents, target_text_embeddings, guidance_scale=1)

            torch.cuda.empty_cache()

            if erase_concept_sampled[0] == erase_concept_sampled[1]:
                target_latents = neutral_latents.clone().detach()
        
        with finetuner:
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
     
        # ----------- NG + APG ------------
        # Using the negative guidance GT with the APG (https://arxiv.org/pdf/2410.02416) formulation.
        pred_neg_guidance = normalized_guidance(positive_latents, neutral_latents, args.negative_guidance)
        loss: torch.Tensor = criteria(negative_latents, pred_neg_guidance)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pbar.set_description(f"Loss: {loss.item():.4f}")

    unet.save_pretrained(save_path)
    torch.cuda.empty_cache()
    return unet


# text encoderのみを更新するのでtext encoderを返す
def train_concept_inversion(
    args: Arguments,
    placeholder_token, 
    initializer_token, 
    train_data_dir, 
    lr, 
    save_path, 
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    noise_scheduler: DDIMScheduler,
    num_vectors=1, 
    max_train_steps=3000,  # Total training steps across all epochs
    learnable_property="object",
    lr_scheduler="constant", 
    lr_warmup_steps=0, 
    scale_lr=False,  # Option to scale learning rate
    iteration=None,
    num_iterations=None,
    center_crop=False
) -> CLIPTextModel:
    
    # Set the random seed for reproducibility
    seed = args.seed
    np.random.seed(seed) 
    random.seed(seed)
    torch.manual_seed(seed) 

    for param in text_encoder.get_input_embeddings().parameters():
        param.requires_grad = True

    # Add placeholder tokens to tokenizer
    placeholder_tokens = [placeholder_token]
    additional_tokens = [f"{placeholder_token}_{i}" for i in range(1, num_vectors)]
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != num_vectors:
        raise ValueError(f"Token '{placeholder_token}' already exists in tokenizer. Use a different token name.")

    # Convert initializer and placeholder tokens to IDs
    initializer_token_id = tokenizer.convert_tokens_to_ids([initializer_token])[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)


    # Resize text encoder embeddings to accommodate new tokens
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize placeholder token embeddings using initializer token
    with torch.no_grad():
        token_embeds = text_encoder.get_input_embeddings().weight.data
        ctr = 0
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
            ctr += 1
        print(f"Initialized {ctr} placeholder token embeddings with '{initializer_token}' token embeddings.")

    # Save the original token embeddings
    org_token_embeds = text_encoder.get_input_embeddings().weight.data.clone()
    

    dataset = TextualInversionDataset(
        data_root=train_data_dir,
        tokenizer=tokenizer,
        size=args.image_size,
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

    # Scale learning rate if specified
    if scale_lr:
        effective_batch_size = dataloader.batch_size
        lr *= effective_batch_size  # Adjust learning rate based on batch size

    optimizer = optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=lr)
    scheduler = get_scheduler(lr_scheduler, optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=max_train_steps)

    # Initialize a single progress bar for the entire training process
    progress_bar = tqdm(total=max_train_steps, desc="Concept Inversion Attack Progress", unit="step")
    global_step = 0

    # Training loop following the epoch and step structure
    for epoch in range(num_train_epochs):
        text_encoder.train()
        
        for step, batch in enumerate(dataloader):
            if global_step >= max_train_steps:
                break

            optimizer.zero_grad()
            batch: dict[str, torch.Tensor]

            latents: torch.Tensor = vae.encode(batch["pixel_values"].to(vae.device)).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 999, (latents.shape[0],), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device)).last_hidden_state
            model_pred: torch.Tensor = unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]

            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            scheduler.step()

            # Freeze all embeddings except for the placeholder tokens
            index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool, device=text_encoder.device)
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False  # False indicates trainable embeddings

            # Restore the frozen embeddings
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight.data[index_no_updates] = org_token_embeds[index_no_updates]
            
            # Update progress bar and global step
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            global_step += 1
            if global_step >= max_train_steps:
                break

    progress_bar.close()

    if save_path is not None:
        text_encoder.save_pretrained(save_path)
    else:
        print("Not saving the text encoder state dict as save_path is None.")

    torch.cuda.empty_cache()
    return text_encoder

def generate_placeholder_token():
    return "token_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

def generate_unique_placeholder_token(saved_tokens: dict[str, torch.Tensor], iteration: int):
    # Generate a new placeholder token
    placeholder_token = generate_placeholder_token()
    
    # Check if the token is already in saved_tokens
    if placeholder_token in saved_tokens.values():
        # If not unique, call the function recursively
        return generate_unique_placeholder_token(saved_tokens, iteration)
    
    # If unique, save it in saved_tokens and return
    saved_tokens[f'{iteration}'] = placeholder_token
    return placeholder_token

def search_thoroughly_enough(
    # diffuser,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    ddim_scheduler: DDIMScheduler,
    initial_erase_concept,
    initializer_token,
    train_data_dir,
    train_method,
    ti_lr,
    n_iterations,
    device,
    ti_max_train_steps,
    learnable_property,
    output_dir, 
    generic_prompt,
    center_crop=False
):
    # Initialize variables for loop
    current_concept = initial_erase_concept
    saved_tokens = {}

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Begin iterative erasure and attack
    for iteration in range(n_iterations):
        # Generate a unique placeholder token for the current attack
        placeholder_token = generate_unique_placeholder_token(saved_tokens, iteration)
        saved_tokens[f'{iteration}'] = placeholder_token

        # Set save paths for intermediate models
        erased_weights_path = os.path.join(output_dir, f"erased_unet_iteration_{iteration}.pt")
        attack_model_path = os.path.join(output_dir, f"ci_attack_text_encoder_iteration_{iteration}.pt")
        
        print(f"\n=========== Iteration {iteration + 1}/{n_iterations} ===========")
        print(f"Erasing concept: {current_concept} -> Placeholder token: '{placeholder_token}' (initialized from '{initializer_token}')")

        # 1. Erase the current concept
        saved_unet = train_erasing(
            args=args,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            noise_scheduler=ddim_scheduler,
            erase_concept=current_concept,
            erase_from=current_concept,
            train_method=train_method,
            save_path=erased_weights_path,
        )
        print(f"Erased weights saved to {erased_weights_path}")

        # 2. Perform textual inversion with the erased model to attack
        unet.load_state_dict(torch.load(erased_weights_path))
        torch.cuda.empty_cache()

        text_encoder = train_concept_inversion(
            args=args,
            placeholder_token=placeholder_token,
            initializer_token=initializer_token,
            train_data_dir=train_data_dir,
            lr=ti_lr,
            save_path=attack_model_path,
            max_train_steps=ti_max_train_steps,
            learnable_property=learnable_property,
            scale_lr=True, 
            iteration=iteration,
            num_iterations=n_iterations,
            center_crop=center_crop
        )
        print(f"Attacked model with placeholder '{placeholder_token}' saved to {attack_model_path}")

        # 3. Perform inference and save images
        print(f"Generating images for current_concept: {current_concept} and placeholder_token: {placeholder_token} using generic prompt: {generic_prompt}")
        inference_and_save(generic_prompt=generic_prompt, prompt=current_concept, placeholder_token=placeholder_token, saved_tokens=saved_tokens, iteration=iteration, output_dir=output_dir, device=device)

        # Update the concept to the current placeholder token for the next iteration
        current_concept = placeholder_token

        print(f"========== Iteration {iteration + 1}/{n_iterations} complete.===========\n\n")
        torch.cuda.empty_cache()


    # Final model and token saving after all iterations
    final_model_path = os.path.join(output_dir, "ste_stage_model.pt")
    torch.save({
        'model_state_dict': diffuser.state_dict(),
        'saved_tokens': saved_tokens
    }, final_model_path)
    print(f"\nIterative erasure and attack complete. Final model saved to {final_model_path}")
    print(f"Placeholder tokens used for attack: {saved_tokens}")

    return final_model_path, saved_tokens, diffuser


def robustly_erase_once(
    args: Arguments,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDIMScheduler,
    unet: UNet2DConditionModel,
    erase_concepts,
    train_method,
    iterations,
    compositional_guidance_scale,
    save_path,
    anchor_concepts_path
):
    nsteps = 50

    with open(anchor_concepts_path, 'r') as f:
        all_anchor_concepts = json.load(f)[erase_concepts[0]]

    finetuner = FineTunedModel(diffuser, train_method=train_method)

    parameters = gather_parameters(train_method, unet)

    optimizer = optim.AdamW(parameters, lr=args.stereo_lr)
    criteria = nn.MSELoss()
    
    torch.cuda.empty_cache()

    # ------- Stratified sampling of the anchor concepts ------
    total_sentences = len(all_anchor_concepts)
    appearances_per_sentence = iterations // total_sentences
    # Repeat and shuffle sentences to get a balanced distribution
    balanced_list = all_anchor_concepts * appearances_per_sentence
    np.random.shuffle(balanced_list)
    # If we still need a few more to reach n iterations
    remainder = iterations - len(balanced_list)
    if remainder > 0:
        balanced_list.extend(np.random.choice(all_anchor_concepts, remainder, replace=False))

    # ------ Stratified sampling of the erase concepts -------
    total_concepts = len(erase_concepts)
    appearances_per_concept = iterations // total_concepts
    # Repeat each concept and shuffle the list
    balanced_erase_list = erase_concepts * appearances_per_concept
    np.random.shuffle(balanced_erase_list)
    # Handle the remainder if `iterations` isn't an exact multiple of `total_concepts`
    remainder = iterations - len(balanced_erase_list)
    if remainder > 0:
        balanced_erase_list.extend(np.random.choice(erase_concepts, remainder, replace=False))

    pbar = tqdm(range(iterations))
    for i in pbar:
        with torch.no_grad():
            erase_concept_sampled = balanced_erase_list[i]
            anchor_concepts = [balanced_list[i]]

            print(f"Erasing concept: {erase_concept_sampled} from anchor concept: {anchor_concepts} at iteration {i}")

            neutral_text_embeddings = get_condition([''], tokenizer, text_encoder)
            target_text_embeddings = get_condition([erase_concept_sampled], tokenizer, text_encoder)

            # Get embeddings for the erase concepts (normal and attack) and anchor concepts
            negative_word_embs = []
            for neg_word in erase_concepts:
                negative_word_embs.append(get_condition([neg_word], tokenizer, text_encoder))

            anchor_word_embs = []
            for anchor_word in anchor_concepts:
                anchor_word_embs.append(get_condition([anchor_word], tokenizer, text_encoder))

            noise_scheduler.set_timesteps(nsteps)
            optimizer.zero_grad()
            iteration = torch.randint(1, nsteps - 1, (1,)).item()
            latents = torch.randn(1, unet.config.in_channels, args.image_size // 8, args.image_size // 8).to(unet.device).repeat(1, 1, 1, 1) * noise_scheduler.init_noise_sigma

            with finetuner:
                latents_steps, _ = diffuser.diffusion(
                    latents,
                    target_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=3, 
                    show_progress=False
                )

            noise_scheduler.set_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)
            timestep = noise_scheduler.timesteps[iteration]
            neutral_latents = predict_noise(unet, noise_scheduler, timestep, latents_steps[0], neutral_text_embeddings)
            target_latents = predict_noise(unet, noise_scheduler, timestep, latents_steps[0], target_text_embeddings)

            # get noise estimate for the negative concepts
            e_negatives_latents = []
            for emb_neg in negative_word_embs:
                e_negatives_latents.append(predict_noise(unet, noise_scheduler, timestep, latents_steps[0], emb_neg))

            # get noise estimate for anchor words
            e_anchor_latents = []
            for emb_anchor in anchor_word_embs:
                e_anchor_latents.append(predict_noise(unet, noise_scheduler, timestep, latents_steps[0], emb_anchor))

            torch.cuda.empty_cache()
        
        with finetuner:
            negative_latents = predict_noise(unet, noise_scheduler, timestep, latents_steps[0], target_text_embeddings)

        # ----- Compositional guidance  + APG 
        # Negative Concept
        neg_guidance_scales = []
        for _ in range(len(e_negatives_latents)):
            neg_guidance_scales.append(-(float(compositional_guidance_scale)))

        # Anchor concepts
        pos_guidance_scales = []
        for _ in range(len(e_anchor_latents)):
            pos_guidance_scales.append(float(compositional_guidance_scale))

        print(f"Using compositional guidance with APG : pos_guidance_scales {pos_guidance_scales} and neg_guidance_scales {neg_guidance_scales}")

        combined_conditions = e_negatives_latents + e_anchor_latents
        combined_guidance_scales = neg_guidance_scales + pos_guidance_scales

        # Using the negative guidance GT with the APG (https://arxiv.org/pdf/2410.02416) formulation with our modified compositional guidance
        compositional_guidance_estimate = normalized_compositional_guidance(combined_conditions, neutral_latents, combined_guidance_scales)
        # Compute loss
        loss: torch.Tensor = criteria(negative_latents, compositional_guidance_estimate)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pbar.set_description(f"Loss: {loss.item():.4f}")

    unet.save_pretrained(save_path)
    torch.cuda.empty_cache()

def stereo(
    args: Arguments,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    ddim_scheduler: DDIMScheduler,
):
    # Initialize the diffuser model on the specified device
    # diffuser = StableDiffuser(scheduler='DDIM')

    ste_start_time = time.time()
    print(f"---------------------------------- Starting Search Thoroughly Enough ----------------------------------")
    # Stage 1: STE (Search Thoroughly Enough)
    final_model_path, saved_tokens, diffuser = search_thoroughly_enough(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        noise_scheduler=ddim_scheduler,
        initial_erase_concept=args.concepts,
        initializer_token=args.initializer_token,
        train_data_dir=args.train_data_dir,
        train_method=args.train_method,
        lr=args.ste_lr,
        ti_lr=args.ci_lr,
        negative_guidance=args.negative_guidance,
        iterations=args.iterations,
        n_iterations=args.n_iterations,
        device=args.device,
        ti_max_train_steps=args.ti_max_train_steps,
        learnable_property=args.learnable_property,
        output_dir=args.output_dir,
        generic_prompt=args.generic_prompt,
        center_crop=args.center_crop
    )
    ste_end_time = time.time()
    print(f"------- Search Thoroughly Enough complete. Time taken: {ste_end_time - ste_start_time} seconds -------")

    print(f"STE model path: {final_model_path}")
    print(f"Saved placeholder tokens: {saved_tokens}")
    torch.cuda.empty_cache()

    # Stage 2: REO (Robustly Erase Once)
    print(f"------- Starting Robustly Erase Once -------")
    diffuser = StableDiffuser(scheduler='DDIM').to(args.device)
    diffuser_copy = StableDiffuser(scheduler='DDIM').to(args.device)
    final_unet_path = os.path.join(args.output_dir, "final_reo_unet.pt")
    final_model_path = os.path.join(args.output_dir, "ste_stage_model.pt")

    ckpt = torch.load(final_model_path)
    saved_tokens = ckpt['saved_tokens']
    # Add the saved tokens to the tokenizer
    for token in list(saved_tokens.values()):
        if token not in diffuser.tokenizer.get_vocab():
            print(f"!!!! Adding placeholder token '{token}' to tokenizer.")
            diffuser.tokenizer.add_tokens([token])
            diffuser_copy.tokenizer.add_tokens([token])
            diffuser.text_encoder.resize_token_embeddings(len(diffuser.tokenizer))
            diffuser_copy.text_encoder.resize_token_embeddings(len(diffuser.tokenizer))
    
    diffuser_copy.load_state_dict(ckpt['model_state_dict'])
    diffuser.text_encoder.load_state_dict(diffuser_copy.text_encoder.state_dict())
    del ckpt, diffuser_copy
    torch.cuda.empty_cache()

    final_concepts_to_erase = [args.erase_concept]
    adv_tokens_from_ste = list(saved_tokens.values())
    print(f"erasing concepts found from STE stage: {adv_tokens_from_ste}")
    if not args.num_of_adv_concepts == 0:
        adv_tokens_from_ste = adv_tokens_from_ste[0:args.num_of_adv_concepts]
        final_concepts_to_erase.extend(adv_tokens_from_ste)
    
    reo_start_time = time.time()
    robustly_erase_once(
        erase_concepts=final_concepts_to_erase,
        train_method=args.train_method,
        iterations=args.iterations,
        compositional_guidance_scale=args.compositional_guidance_scale,
        lr=args.reo_lr,
        save_path=final_unet_path,
        diffuser=diffuser,
        anchor_concepts_path=args.anchor_concept_path
    )
    reo_end_time = time.time()
    del diffuser, saved_tokens
    torch.cuda.empty_cache()
    print(f"---------------------- Robustly Erase Once complete. Final model saved to {final_unet_path} ----------------------")
    print(f"REO time: {reo_end_time - reo_start_time} seconds")


def train(args: Arguments):
    device = get_devices(args)[0]
    tokenizer, text_encoder, vae, unet, ddim_scheduler, ddpm_scheduler = get_models(args)

    lms_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    text_encoder.eval()
    vae.eval()
    unet.eval()
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept Erasing and Textual Inversion")
    parser.add_argument("--erase_concept", required=True, help="Concept to erase")
    parser.add_argument("--train_method", required=True, help="Method for training (OPTIONS: noxattn/xattn)")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations for the erasing objectives")
    parser.add_argument("--negative_guidance", type=float, default=2.0, help="Negative guidance value")
    parser.add_argument("--ste_lr", type=float, default=0.5e-5, help="Learning rate for erasing in search throughly enough stage")
    parser.add_argument("--reo_lr", type=float, default=2e-5, help="Learning rate for erasing in robustly erase once stage")
    parser.add_argument("--ci_lr", type=float, default=5e-3, help="Learning rate for textual inversion")
    parser.add_argument("--ti_max_train_steps", type=int, default=3000, help="Maximum training steps for textual inversion")
    parser.add_argument("--train_data_dir", type=str, required=False, help="Gallery images to be used during training")
    parser.add_argument("--learnable_property", type=str, required=False, help="object/style", default="object")
    parser.add_argument("--initializer_token", type=str, required=True, help="Initializer token (OPTIONS: person/object/art)")
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda')
    parser.add_argument("--n_iterations", type=int, required=False, help="Total number of erasure-attack iterations", default=4)   
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saving models")
    parser.add_argument("--generic_prompt", type=str, required=False, help="Generic prompt for textual inversion visualization", default="a photo of a")
    parser.add_argument("--anchor_concept_path", type=str, required=False, help="Path to anchor concept json used in REO stage", default='utils/anchor_prompts.json')
    parser.add_argument("--compositional_guidance_scale", type=float, required=False, help="Compositional guidance scale. The value has to be +1 of the scale you would like to set. If the intended scale is 1.0, then the value has to be 2.0", default=2.0)
    parser.add_argument("--mode", type=str, required=False, help="Mode of operation (OPTIONS: stereo/attack/both)", default="stereo")
    parser.add_argument("--unet_ckpt_to_attack", type=str, required=False, help="Path to the unet ckpt that has to be attacked to test its robustness", default="final_reo_unet.pt")
    parser.add_argument("--attack_eval_images", type=str, required=True, help="Gallery images to be used for attacking the model for evaluation")
    parser.add_argument("--center_crop", type=bool, required=False, help="Center crop the images during training", default=False)
    parser.add_argument("--num_of_adv_concepts", type=int, required=False, help="Number of adversarial concepts to use in REO", default=4)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set the random seed for reproducibility
    seed = 42
    np.random.seed(seed)      # For numpy
    random.seed(seed)         # For the random module
    torch.manual_seed(seed)   # For PyTorch

    # For CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you have multiple GPUs

    # Ensure PyTorch operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.mode == "stereo":
        stereo(args)
    elif args.mode == "attack":
        diffuser = StableDiffuser(scheduler='DDIM').to(args.device)
        attack_stereo(args, diffuser)
    elif args.mode == 'both':
        stereo(args)
        diffuser = StableDiffuser(scheduler='DDIM').to(args.device)
        attack_stereo(args, diffuser)