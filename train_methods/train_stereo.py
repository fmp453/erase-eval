# https://github.com/koushiksrivats/robust-concept-erasing/blob/main

import argparse
import os
import random
import time
import math

import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, AutoencoderKL, UNet2DConditionModel, get_scheduler
# from utils.stereo import stereo, attack_stereo
# from utils.utils import StableDiffuser

from train_methods.train_utils import get_models, get_devices
from utils import Arguments


# ESDのdiffusers実装と同じ
def train_erasing(
    args: Arguments,
    erase_concept,
    erase_from,
    train_method,
    iterations,
    lr,
    save_path,
):
    # Set the random seed for reproducibility
    seed = args.seed
    np.random.seed(seed)      # For numpy
    random.seed(seed)         # For the random module
    torch.manual_seed(seed) 

    nsteps = 50

    diffuser.requires_grad = True
    diffuser.train()

    finetuner = FineTunedModel(diffuser, train_method=train_method)

    optimizer = torch.optim.AdamW(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))
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

            neutral_text_embeddings = diffuser.get_text_embeddings([''], n_imgs=1)
            positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]], n_imgs=1)
            target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]], n_imgs=1)

            diffuser.set_scheduler_timesteps(nsteps)
            optimizer.zero_grad()
            iteration = torch.randint(1, nsteps - 1, (1,)).item()
            latents = diffuser.get_initial_latents(1, 512, 1)

            with finetuner:
                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=3, 
                    show_progress=False
                )

            diffuser.set_scheduler_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)
            positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
            neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
            target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)

            torch.cuda.empty_cache()

            if erase_concept_sampled[0] == erase_concept_sampled[1]:
                target_latents = neutral_latents.clone().detach()
        
        with finetuner:
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
     
        # ------------------------------------------ NG + APG --------------------------------------------------
        # Using the negative guidance GT with the APG (https://arxiv.org/pdf/2410.02416) formulation.
        pred_neg_guidance = normalized_guidance(positive_latents, neutral_latents, args.negative_guidance)
        loss = criteria(negative_latents, pred_neg_guidance)
        # ------------------------------------------------------------------------------------------------------

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pbar.set_description(f"Loss: {loss.item():.4f}")

    with finetuner:
        torch.save(diffuser.unet.state_dict(), save_path)

    del neutral_text_embeddings, positive_text_embeddings, target_text_embeddings
    del latents, latents_steps, positive_latents, neutral_latents, target_latents, negative_latents
    del loss, optimizer, finetuner
    del erase_concept, erase_from, criteria, pbar, index, erase_concept_sampled, iteration, nsteps

    torch.cuda.empty_cache()
    diffuser.eval()
    return diffuser


def train_concept_inversion(
    args: Arguments,
    diffuser,
    placeholder_token, 
    initializer_token, 
    train_data_dir, 
    lr, 
    save_path, 
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    num_vectors=1, 
    max_train_steps=3000,  # Total training steps across all epochs
    learnable_property="object",
    lr_scheduler="constant", 
    lr_warmup_steps=0, 
    scale_lr=False,  # Option to scale learning rate
    iteration=None,
    num_iterations=None,
    center_crop=False
):
    
    # Set the random seed for reproducibility
    seed = args.seed
    np.random.seed(seed) 
    random.seed(seed)
    torch.manual_seed(seed) 

    for param in text_encoder.get_input_embeddings.parameters():
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
    

    # Set up dataset and dataloader with specified resolution
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

    # Calculate steps per epoch and number of epochs needed to reach max_train_steps
    steps_per_epoch = len(dataloader)
    num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)

    # Scale learning rate if specified
    if scale_lr:
        effective_batch_size = dataloader.batch_size
        lr *= effective_batch_size  # Adjust learning rate based on batch size

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=lr)
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

            # Zero gradients for each batch
            optimizer.zero_grad()

            # Encode images to latent space
            latents = vae.encode(batch["pixel_values"].to(vae.device)).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 999, (latents.shape[0],), device=latents.device)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Forward pass through U-Net within finetuner context
            encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device)).last_hidden_state
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]

            # Calculate loss and backpropagate
            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            # Optimizer step and scheduler update
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

    # Save the text encoder state dict
    if not (save_path == None):
        torch.save(diffuser.text_encoder.state_dict(), save_path)
    else:
        print("Not saving the text encoder state dict as save_path is None.")

    del optimizer, scheduler, dataset, dataloader, progress_bar, global_step, steps_per_epoch, num_train_epochs, effective_batch_size, token_embeds, index_no_updates, model_pred, target, loss, batch, latents, noise, timesteps, noisy_latents, encoder_hidden_states, placeholder_tokens, additional_tokens, initializer_token_id, placeholder_token_ids, tokenizer, org_token_embeds
    torch.cuda.empty_cache()
    diffuser.eval()
    return diffuser

def search_thoroughly_enough(
    diffuser,
    initial_erase_concept,
    initializer_token,
    train_data_dir,
    train_method,
    lr,
    ti_lr,
    negative_guidance,
    iterations,
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
        diffuser = train_erasing(
            erase_concept=current_concept,
            erase_from=current_concept,
            train_method=train_method,
            iterations=iterations,
            negative_guidance=negative_guidance,
            lr=lr,
            save_path=erased_weights_path,
            diffuser=diffuser,
            device=device
        )
        print(f"Erased weights saved to {erased_weights_path}")

        # 2. Perform textual inversion with the erased model to attack
        diffuser.unet.load_state_dict(torch.load(erased_weights_path))
        torch.cuda.empty_cache()

        diffuser = train_concept_inversion(
            diffuser=diffuser,
            placeholder_token=placeholder_token,
            initializer_token=initializer_token,
            train_data_dir=train_data_dir,
            lr=ti_lr,
            save_path=attack_model_path,
            device=device,
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

        print(f"===================================== Iteration {iteration + 1}/{n_iterations} complete. =====================================\n\n")
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
        # diffuser=diffuser,
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
        # self.feature_extractor = CLIPFeatureExtractor.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="feature_extractor")
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")

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