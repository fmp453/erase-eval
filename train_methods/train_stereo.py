# https://github.com/koushiksrivats/robust-concept-erasing/blob/main

import random
import time
import string
import math
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel, get_scheduler

from train_methods.data import TextualInversionDataset
from train_methods.train_utils import get_models, get_devices, get_condition, gather_parameters, predict_noise, sample_until
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
    momentum_buffers: list[MomentumBuffer] | None = None, # List of MomentumBuffers for each condition
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


def train_erasing(
    args: Arguments,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    noise_scheduler: DDIMScheduler,
    erase_concept: str,
    erase_from: str,
    save_dir,
) -> UNet2DConditionModel:
    # Set the random seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    nsteps = 50

    parameters = gather_parameters(args.stereo_method, unet)

    optimizer = optim.AdamW(parameters, lr=args.stereo_ste_lr)
    criteria = nn.MSELoss()

    pbar = tqdm(range(args.stereo_iteration))
    erase_concept = [a.strip() for a in erase_concept.split(',')]
    erase_from = [a.strip() for a in erase_from.split(',')]

    if len(erase_from) != len(erase_concept):
        if len(erase_from) == 1:
            erase_from = [erase_from[0]] * len(erase_concept)
        else:
            raise ValueError("Erase concepts and target concepts must have matching lengths.")

    erase_concept = [[e, f] for e, f in zip(erase_concept, erase_from)]

    for _ in pbar:
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

            latents_steps = sample_until(
                until=iteration,
                latents=latents,
                unet=unet,
                scheduler=noise_scheduler,
                prompt_embeds=positive_text_embeddings,
                guidance_scale=3, 
            )

            noise_scheduler.set_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)
            timestep = noise_scheduler.timesteps[iteration]
            positive_latents = predict_noise(unet, noise_scheduler, timestep, latents_steps, positive_text_embeddings)
            neutral_latents = predict_noise(unet, noise_scheduler, timestep, latents_steps, neutral_text_embeddings)

            torch.cuda.empty_cache()

        negative_latents = predict_noise(unet, noise_scheduler, timestep, latents_steps, target_text_embeddings)

        # ----------- NG + APG ------------
        # Using the negative guidance GT with the APG (https://arxiv.org/pdf/2410.02416) formulation.
        pred_neg_guidance = normalized_guidance(positive_latents, neutral_latents, args.negative_guidance)
        loss: torch.Tensor = criteria(negative_latents, pred_neg_guidance)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pbar.set_description(f"Loss: {loss.item():.4f}")

    unet.eval()
    unet.save_pretrained(save_dir)
    torch.cuda.empty_cache()
    return unet


def train_concept_inversion(
    args: Arguments,
    placeholder_token, 
    lr, 
    train_data_dir: str,
    text_encoder_save_path,
    tokenizer_save_path,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    noise_scheduler: DDIMScheduler,
    scale_lr=False,  # Option to scale learning rate
    iteration=None,
    num_iterations=None,
) -> tuple[CLIPTokenizer, CLIPTextModel]:
    
    # Set the random seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for param in text_encoder.get_input_embeddings().parameters():
        param.requires_grad = True

    # Add placeholder tokens to tokenizer
    placeholder_tokens = [placeholder_token]
    additional_tokens = [f"{placeholder_token}_{i}" for i in range(1, 1)]
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != 1:
        raise ValueError(f"Token '{placeholder_token}' already exists in tokenizer. Use a different token name.")

    # Convert initializer and placeholder tokens to IDs
    initializer_token_id = tokenizer.convert_tokens_to_ids([args.stereo_initializer_token])[0]
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
        print(f"Initialized {ctr} placeholder token embeddings with '{args.stereo_initializer_token}' token embeddings.")

    # Save the original token embeddings
    org_token_embeds = text_encoder.get_input_embeddings().weight.data.clone()

    dataset = TextualInversionDataset(
        data_root=train_data_dir,
        tokenizer=tokenizer,
        size=args.image_size,
        placeholder_token=" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)),
        repeats=100,
        set="train",
        learnable_property=args.stereo_learnable_property,
        iteration=iteration,
        num_iterations=num_iterations
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    steps_per_epoch = len(dataloader)
    num_train_epochs = math.ceil(args.stereo_ti_max_iters / steps_per_epoch)

    # Scale learning rate if specified
    if scale_lr:
        effective_batch_size = dataloader.batch_size
        lr *= effective_batch_size  # Adjust learning rate based on batch size

    optimizer = optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=lr)
    scheduler = get_scheduler("constant", optimizer, num_warmup_steps=0, num_training_steps=args.stereo_ti_max_iters)

    # Initialize a single progress bar for the entire training process
    progress_bar = tqdm(total=args.stereo_ti_max_iters, desc="Concept Inversion Attack Progress", unit="step")
    global_step = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        
        for step, batch in enumerate(dataloader):

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
            if global_step >= args.stereo_ti_max_iters:
                break

    progress_bar.close()
    text_encoder.eval()
    text_encoder.save_pretrained(text_encoder_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    torch.cuda.empty_cache()
    return tokenizer, text_encoder

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


def inference_and_save(
    args: Arguments,
    prompt,
    saved_tokens: dict[str, torch.Tensor],
    iteration,
    unet_dir,
    text_encoder_dir,
    tokenizer_dir,
):

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(unet_dir)
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(text_encoder_dir)
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(args.sd_version)

    pipe.unet = unet
    pipe.tokenizer = tokenizer
    pipe.text_encoder = text_encoder
    device = get_devices(args)[0]
    pipe.to(device)
    pipe.eval()

    generator = torch.Generator().manual_seed(args.seed)

    iteration_dir = Path(args.data_dir, f"iteration_{iteration}")
    iteration_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        erased_images = pipe(
            f"{args.stereo_generic_prompt} {prompt}",
            width=args.image_size,
            height=args.image_size,
            num_inference_steps=50,
            num_images_per_prompt=args.num_images_per_prompt,
            generator=generator,
            guidance_scale=args.guidance_scale
        ).images

    for i, img in enumerate(erased_images):
        img.save(Path(iteration_dir, f"erased_image_{i}.png"))

    with torch.no_grad():
        for token in list(saved_tokens.values()):
            attack_images = pipe(
                f"{args.stereo_generic_prompt} {token}",
                width=args.image_size,
                height=args.image_size,
                num_inference_steps=50,
                num_images_per_prompt=args.num_images_per_prompt,
                generator=generator,
                guidance_scale=args.guidance_scale
            )
            for i, img in enumerate(attack_images):
                img.save(Path(iteration_dir, f"attack_image_placeholder_{token}_{i}.png"))
            torch.cuda.empty_cache()

    print(f"Generated and saved images for iteration {iteration}.")


def search_thoroughly_enough(
    args: Arguments,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    ddim_scheduler: DDIMScheduler,
) -> tuple[str, dict[str, torch.Tensor]]:
    # Initialize variables for loop
    current_concept = args.concepts
    saved_tokens = {}
    save_dir = Path(args.save_dir)

    save_dir.mkdir(exist_ok=True)
    
    # Begin iterative erasure and attack
    for iteration in range(args.stereo_n_iters):
        # Generate a unique placeholder token for the current attack
        placeholder_token = generate_unique_placeholder_token(saved_tokens, iteration)
        saved_tokens[f'{iteration}'] = placeholder_token

        # Set save paths for intermediate models
        erased_unet_dir = save_dir / f"{iteration}" / "erased_unet"
        attack_model_dir = save_dir / f"{iteration}" / "ci_attack_text_encoder"
        attack_tokenizer_dir = save_dir / f"{iteration}" / "ci_attack_tokenizer"

        print(f"Erasing concept: {current_concept} -> Placeholder token: '{placeholder_token}' (initialized from '{args.stereo_initializer_token}')")

        # 1. Erase the current concept
        # saved_unet is saved at save_dir
        saved_unet = train_erasing(
            args=args,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            noise_scheduler=ddim_scheduler,
            erase_concept=current_concept,
            erase_from=current_concept,
            save_dir=erased_unet_dir,
        )

        # 2. Perform textual inversion with the erased model to attack
        # text_encoder is saved at attack_model_dir
        # tokenizer is saved at attack_tokenizer_dir
        tokenizer, text_encoder = train_concept_inversion(
            args=args,
            placeholder_token=placeholder_token,
            lr=args.stereo_ci_lr,
            train_data_dir=args.data_dir,
            text_encoder_save_path=attack_model_dir,
            tokenizer_save_path=attack_tokenizer_dir,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=saved_unet,
            noise_scheduler=ddim_scheduler,
            scale_lr=True, 
            iteration=iteration,
        )
        print(f"Attacked model with placeholder '{placeholder_token}' saved to {attack_model_dir}")

        # 3. Perform inference and save images
        print(f"Generating images for current_concept: {current_concept} and placeholder_token: {placeholder_token} using generic prompt: {args.stereo_generic_prompt}")
        inference_and_save(
            args=args,
            prompt=current_concept,
            saved_tokens=saved_tokens,
            iteration=iteration,
            unet_dir=erased_unet_dir,
            text_encoder_dir=attack_model_dir,
            tokenizer_dir=attack_tokenizer_dir,
        )

        # Update the concept to the current placeholder token for the next iteration
        current_concept = placeholder_token

        print(f"Iteration {iteration + 1}/{args.stereo_n_iters} complete.")

    # Final model and token saving after all iterations
    final_model_path = save_dir / "ste_stage_model.pt"
    torch.save({
        'saved_tokens': saved_tokens
    }, final_model_path)
    print(f"\nIterative erasure and attack complete. Final model saved to {final_model_path}")
    print(f"Placeholder tokens used for attack: {saved_tokens}")

    return final_model_path, saved_tokens


def robustly_erase_once(
    args: Arguments,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    noise_scheduler: DDIMScheduler,
    unet: UNet2DConditionModel,
    erase_concepts: list,
    save_path,
):
    nsteps = 50

    with open(args.stereo_anchor_concept_path, 'r') as f:
        all_anchor_concepts: list = json.load(f)[erase_concepts[0]]

    _, parameters = gather_parameters(args.stereo_method, unet)

    optimizer = optim.AdamW(parameters, lr=args.stereo_reo_lr)
    criteria = nn.MSELoss()

    torch.cuda.empty_cache()

    # ------- Stratified sampling of the anchor concepts ------
    total_sentences = len(all_anchor_concepts)
    appearances_per_sentence = args.stereo_iteration // total_sentences
    # Repeat and shuffle sentences to get a balanced distribution
    balanced_list = all_anchor_concepts * appearances_per_sentence
    np.random.shuffle(balanced_list)
    # If we still need a few more to reach n iterations
    remainder = args.stereo_iteration - len(balanced_list)
    if remainder > 0:
        balanced_list.extend(np.random.choice(all_anchor_concepts, remainder, replace=False))

    # ------ Stratified sampling of the erase concepts -------
    total_concepts = len(erase_concepts)
    appearances_per_concept = args.stereo_iteration // total_concepts
    # Repeat each concept and shuffle the list
    balanced_erase_list = erase_concepts * appearances_per_concept
    np.random.shuffle(balanced_erase_list)
    # Handle the remainder if `iterations` isn't an exact multiple of `total_concepts`
    remainder = args.stereo_iteration - len(balanced_erase_list)
    if remainder > 0:
        balanced_erase_list.extend(np.random.choice(erase_concepts, remainder, replace=False))

    pbar = tqdm(range(args.stereo_iteration))
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

            latents_steps, _ = sample_until(
                until=iteration,
                latents=latents,
                unet=unet,
                scheduler=noise_scheduler,
                prompt_embeds=target_text_embeddings,
                guidance_scale=3, 
            )

            noise_scheduler.set_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)
            timestep = noise_scheduler.timesteps[iteration]
            neutral_latents = predict_noise(unet, noise_scheduler, timestep, latents_steps[0], neutral_text_embeddings)

            # get noise estimate for the negative concepts
            e_negatives_latents = []
            for emb_neg in negative_word_embs:
                e_negatives_latents.append(predict_noise(unet, noise_scheduler, timestep, latents_steps[0], emb_neg))

            # get noise estimate for anchor words
            e_anchor_latents = []
            for emb_anchor in anchor_word_embs:
                e_anchor_latents.append(predict_noise(unet, noise_scheduler, timestep, latents_steps[0], emb_anchor))

            torch.cuda.empty_cache()

        negative_latents = predict_noise(unet, noise_scheduler, timestep, latents_steps[0], target_text_embeddings)

        # ----- Compositional guidance  + APG 
        # Negative Concept
        neg_guidance_scales = []
        for _ in range(len(e_negatives_latents)):
            neg_guidance_scales.append(-(float(args.stereo_compositional_guidance_scale)))

        # Anchor concepts
        pos_guidance_scales = []
        for _ in range(len(e_anchor_latents)):
            pos_guidance_scales.append(float(args.stereo_compositional_guidance_scale))

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

    ste_start_time = time.time()
    print("-- Starting Search Thoroughly Enough --")
    # Stage 1: STE (Search Thoroughly Enough)
    final_model_path, saved_tokens = search_thoroughly_enough(
        args=args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        noise_scheduler=ddim_scheduler,
        center_crop=args.center_crop
    )
    print(f"------- Search Thoroughly Enough complete. Time taken: {time.time() - ste_start_time} seconds -------")

    print(f"STE model path: {final_model_path}")
    print(f"Saved placeholder tokens: {saved_tokens}")
    torch.cuda.empty_cache()

    # Stage 2: REO (Robustly Erase Once)
    print(f"------- Starting Robustly Erase Once -------")
    final_unet_path = Path(args.save_dir / "final_reo_unet")
    final_model_path = Path(args.save_dir / "ste_stage")

    # copy models
    copy_tokenizer = deepcopy(tokenizer)
    copy_text_encoder = deepcopy(text_encoder)

    ckpt = torch.load(final_model_path)
    saved_tokens: dict[str, torch.Tensor] = ckpt['saved_tokens']
    # Add the saved tokens to the tokenizer
    for token in list(saved_tokens.values()):
        if token not in tokenizer.get_vocab():
            print(f"!!!! Adding placeholder token '{token}' to tokenizer.")
            tokenizer.add_tokens([token])
            copy_tokenizer.add_tokens([token])
            text_encoder.resize_token_embeddings(len(tokenizer))
            copy_text_encoder.resize_token_embeddings(len(tokenizer))

    text_encoder.load_state_dict(copy_text_encoder.state_dict())
    torch.cuda.empty_cache()

    final_concepts_to_erase = [args.concepts]
    adv_tokens_from_ste = list(saved_tokens.values())
    print(f"erasing concepts found from STE stage: {adv_tokens_from_ste}")
    if not args.stereo_num_of_adv_concepts == 0:
        adv_tokens_from_ste = adv_tokens_from_ste[0:args.stereo_num_of_adv_concepts]
        final_concepts_to_erase.extend(adv_tokens_from_ste)
    
    reo_start_time = time.time()
    robustly_erase_once(
        args=args,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        noise_scheduler=ddim_scheduler,
        unet=unet,
        erase_concepts=final_concepts_to_erase,
        save_path=final_unet_path,
    )
    torch.cuda.empty_cache()
    print(f"-- Robustly Erase Once complete. Final model saved to {final_unet_path} --")
    print(f"REO time: {time.time() - reo_start_time} seconds")

def inference_attack(
    args: Arguments,
    saved_tokens,
    unet: UNet2DConditionModel, 
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel
):
    device = text_encoder.device
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_version)
    pipe.text_encoder = text_encoder
    pipe.tokenizer = tokenizer
    pipe.unet = unet
    pipe.eval()
    pipe.to(device)

    generator = torch.Generator().manual_seed(args.seed)

    iteration_dir = Path(args.save_dir / "eval_ci_iteration")
    iteration_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for token in saved_tokens:
            attack_images = pipe(
                f"{args.stereo_generic_prompt} {token}",
                width=args.image_size,
                height=args.image_size,
                num_inference_steps=50,
                num_images_per_prompt=10,
                generator=generator,
                guidance_scale=7.5
            )
            for i, img in enumerate(attack_images):
                img.save(Path(iteration_dir / f"eval_ci_attack_image_placeholder_{token}_{i}.png"))

def attack_stereo(
    args: Arguments,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    noise_scheduler: DDIMScheduler,
):

    placeholder_token = generate_placeholder_token()

    tokenizer, text_encoder = train_concept_inversion(
        args=args,
        placeholder_token=placeholder_token,
        train_data_dir=args.stereo_attack_eval_images,
        lr=args.stereo_ci_lr,
        train_data_dir=args.data_dir,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        noise_scheduler=noise_scheduler,
        scale_lr=True, 
        iteration=0,
        num_iterations=1,
    )
    inference_attack(
        args=args,
        saved_tokens=[placeholder_token], 
        unet=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder
    )


def train(args: Arguments):
    # Ensure PyTorch operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = get_devices(args)[0]
    tokenizer, text_encoder, vae, unet, ddim_scheduler, _ = get_models(args)

    text_encoder.eval()
    vae.eval()
    unet.eval()
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)

    match args.stereo_mode:
        case "stereo":
            stereo(
                args,
                tokenizer,
                text_encoder,
                vae,
                unet,
                ddim_scheduler
            )
        case "attack":
            attack_stereo(
                args,
                tokenizer,
                text_encoder,
                vae,
                unet,
            )
        case "both":
            stereo(
                args,
                tokenizer,
                text_encoder,
                vae,
                unet,
                ddim_scheduler
            )
            attack_stereo(args)
        case _:
            ValueError("mode must be stereo, attack, or both.")
