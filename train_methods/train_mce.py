# Minimalist Concept Erasure in Generative Models (ICML 2025)
# original repo: https://github.com/YaNgZhAnG-V5/minimalist_concept_erasure


from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from diffusers import EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.testing_utils import enable_full_determinism
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import Arguments
from train_methods.train_utils import get_devices
from train_methods.data import MCEDataset
from train_methods.templates import VALIDATION_PROMPT
from train_methods.utils_mce import dataset_filter, init_hooker, save_image_seed
from train_methods.mce_models import (
    SD2PipelineForCheckpointing,
    SD3PipelineForCheckpointing,
    SDXLPipelineForCheckpointing,
    DiTPipelineForCheckpointing,
    FluxPipelineForCheckpointing,
    ReverseDPMSolverMultistepScheduler,
    PipelineOutput,
    Pipeline
)
from train_methods.mce_models.hooks import (
    CrossAttentionExtractionHook,
    LinearLayerHooker,
    FeedForwardHooker,
    NormHooker,
)


enable_full_determinism()


def load_pipeline(model_str: str) -> Pipeline:
    if model_str == "sd1":
        pipe = SD2PipelineForCheckpointing.from_pretrained("CompVis/stable-diffusion-v1-4", include_entities=False)
    elif model_str == "sd2":
        # Use the Euler scheduler here instead
        scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="scheduler")
        pipe = SD2PipelineForCheckpointing.from_pretrained("stabilityai/stable-diffusion-2-base", scheduler=scheduler)
    elif model_str == "sdxl":
        pipe = SDXLPipelineForCheckpointing.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
    elif model_str == "sdxl_turbo":
        pipe = SDXLPipelineForCheckpointing.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True)
        pipe.set_distilled()
    elif model_str == "sd3":
        pipe = SD3PipelineForCheckpointing.from_pretrained("stabilityai/stable-diffusion-3.5-large")
    elif model_str == "dit":
        pipe = DiTPipelineForCheckpointing.from_pretrained("facebook/DiT-XL-2-256")
        pipe.scheduler = ReverseDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_str == "flux":
        pipe = FluxPipelineForCheckpointing.from_pretrained("black-forest-labs/FLUX.1-schnell")
    elif model_str == "flux_dev":
        pipe = FluxPipelineForCheckpointing.from_pretrained("black-forest-labs/FLUX.1-dev")
    else:
        raise ValueError(f"Model {model_str} not supported")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def calculate_mask_sparsity(
    hooker: CrossAttentionExtractionHook | FeedForwardHooker | LinearLayerHooker | NormHooker,
    threshold: float | None = None
):
    total_num_lambs = 0
    num_activate_lambs = 0
    binary = getattr(hooker, "binary", None) # if binary is not present, it will return None for ff_hooks
    for lamb in hooker.lambs:
        total_num_lambs += lamb.size(0)
        if binary:
            assert threshold is None, "threshold should be None for binary mask"
            num_activate_lambs += lamb.sum().item()
        else:
            assert threshold is not None, "threshold must be provided for non-binary mask"
            num_activate_lambs += (lamb >= threshold).sum().item()
    return total_num_lambs, num_activate_lambs, num_activate_lambs / total_num_lambs


def load_validation_prompts(args: Arguments):
    validation_prompt = []
    prompts = VALIDATION_PROMPT[args.mce_style]
    for t in prompts:
        if callable(t):
            validation_prompt.append(t(args.concepts))
        else:
            validation_prompt.append(t)
    return validation_prompt


@torch.no_grad()
def forward_checkpointing(
    pipe: Pipeline,
    prompt,
    num_inference_steps,
    generator=None,
    output_type="latent",
    keep_last_latent=False,
    width=None,
    height=None,
) -> tuple[PipelineOutput, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    preparation_phase_output = pipe.inference_preparation_phase(
        prompt,
        generator=generator,
        num_inference_steps=num_inference_steps,
        output_type=output_type,
        width=width,
        height=height,
    )
    assert isinstance(preparation_phase_output, PipelineOutput)
    intermediate_latents = [preparation_phase_output.latents]
    timesteps = preparation_phase_output.timesteps
    for timesteps_idx, t in enumerate(timesteps):
        latents = pipe.inference_denoising_step(timesteps_idx, t, preparation_phase_output)
        # update latents in output class
        preparation_phase_output.latents = latents
        intermediate_latents.append(latents)
    # pop the last latents for backprop, only keep it for unlearn concept
    if not keep_last_latent:
        intermediate_latents.pop()
    latents.requires_grad = True
    return preparation_phase_output, intermediate_latents, timesteps, latents


def pruning_loss(
    reconstruction_loss_func: nn.L1Loss | nn.MSELoss,
    image_pt: torch.Tensor,
    image: dict[str, torch.Tensor],
    beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(image, dict):
        image = image["images"]
        loss_reconstruct: torch.Tensor = reconstruction_loss_func(image_pt, image)
    else:
        raise ValueError("image should be a dict")

    loss = loss_reconstruct * beta
    return loss, loss_reconstruct


def train(args: Arguments):

    device = torch.device(get_devices(args)[0])
    project_folder = Path("mce-results")

    # load validation prompts
    validation_prompts = load_validation_prompts(args)
    print(f"Validation prompts: {validation_prompts}")

    pipe = load_pipeline(args.mce_model)
    pipe.to(device)

    # set required_grad to False for all parameters
    # unet, vae, transformer (for sd3)
    pipe.vae.requires_grad_(False)
    if args.mce_model in ["sd3", "dit", "flux", "flux_dev"]:
        pipe.transformer.requires_grad_(False)
    else:
        pipe.unet.requires_grad_(False)

    # prepare for the datasets and dataloader
    train_dataset = MCEDataset(
        metadata=args.mce_metadata,
        deconceptmeta=args.mce_deconceptmeta,
        pipe=pipe,
        num_inference_steps=args.mce_num_intervention_steps,
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
        size=args.mce_size,
        concept=args.concepts,
        neutral_concept=None,
        only_deconcept_latent=False,
        style=(args.mce_style == "style"),
        img_size=args.image_size,
        with_synonyms=args.mce_with_synonyms,
    )

    print(f"starting dataset filtering, current dataset size: {len(train_dataset)}")
    train_dataset, ds_size = dataset_filter(train_dataset, args, device)
    args.mce_size = ds_size
    print(f"filtered dataset, dataset size : {len(train_dataset)}")

    batch_size = args.mce_batch_size
    print(f"Batch size: {batch_size}")
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # save the original image
    train_path = Path("images", "train", "initial_image")
    val_path = Path("images", "validation", "initial_image")
    prompts = [validation_prompts, train_dataset[0]["prompt"]]
    all_imgs = []
    for path, prompt in zip([val_path, train_path], prompts):
        img = save_image_seed(pipe, prompt, args.mce_num_intervention_steps, device, args.seed, save_dir=path)
        if img is not None:
            all_imgs += img

    # define loss for reconstruction
    if args.mce_reconstruct == 1:
        reconstruction_loss_func = nn.L1Loss(reduction="mean")
    elif args.mce_reconstruct == 2:
        reconstruction_loss_func = nn.MSELoss()
    else:
        raise ValueError(f"Reconstruction loss {args.mce_reconstruct} not supported")

    # initialize hooks
    hookers, lr_list = init_hooker(args, pipe, project_folder)

    # dummy generation to initialize the lambda
    print(f"Initializing lambda to be {args.mce_init_lambda}")
    _ = pipe(validation_prompts, generator=None, num_inference_steps=1)
    trainable_lambs: list[torch.Tensor] = []
    for hooker in hookers:
        trainable_lambs += hooker.lambs

    # optimizer and scheduler
    params = []
    for hooker, lr in zip(hookers, lr_list):
        params.append({"params": hooker.lambs, "lr": lr})

    optimizer = optim.AdamW(params)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.mce_lr_warmup_steps,
        num_training_steps=args.mce_epochs * args.mce_size,
        num_cycles=args.mce_lr_num_cycles,
        power=args.mce_lr_power,
    )

    print("Start Training ...")

    mean_loss_reconstruct, mean_intermediate_loss = 0, 0
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    total_step = args.mce_epochs * args.mce_size // args.mce_accumulate_grad_batches
    with tqdm(total=total_step) as pbar:
        for i in range(args.mce_epochs):
            for idx, data in enumerate(dataloader):
                # image_pt contains all latents, the denoise latent is the last one
                image_pt = data["image"]
                deconcept_image_pt = data["deconcept_image"]
                prompt = data["prompt"]
                value = data["value"].item()  # 1 for unlearn, 0 for the rest

                # use grad checkpointing to save memory
                (preparation_phase_output, intermediate_latents, timesteps, latents) = forward_checkpointing(
                    pipe,
                    prompt,
                    num_inference_steps=args.mce_num_intervention_steps,
                    height=args.image_size,
                    width=args.image_size,
                )

                # backprop from loss to the last latents
                with torch.set_grad_enabled(True):
                    prompt_embeds = preparation_phase_output.prompt_embeds

                    image = pipe.inference_aft_denoising(latents, prompt_embeds, None, "latent", True, device)
                    # calculate loss
                    loss, loss_reconstruct = pruning_loss(
                        reconstruction_loss_func,
                        image_pt[:, -1, ...],  # last denoise latent
                        image,
                        args.mce_beta,
                    )

                    if value:  # w unlearn concept
                        # calculate the grad w.r.t. lambda in intermediate range (not z_0)
                        intermediate_loss: torch.Tensor = reconstruction_loss_func(
                            deconcept_image_pt[:, -1, ...], image[0]  # last denoise latent
                        )
                        mean_intermediate_loss += intermediate_loss.item() / args.mce_accumulate_grad_batches
                        loss = intermediate_loss

                    (loss / args.mce_accumulate_grad_batches).backward()
                    grad = latents.grad.detach()

                # backprop from the last latents to the first latents
                timesteps = preparation_phase_output.timesteps

                # update intermediate
                for timesteps_idx, t in enumerate(reversed(timesteps)):
                    current_latents = intermediate_latents[-(timesteps_idx + 1)].detach()
                    current_latents.requires_grad = True
                    timesteps_idx = len(timesteps) - timesteps_idx - 1
                    with torch.set_grad_enabled(True):
                        preparation_phase_output.latents = current_latents
                        # denoised latents t-1
                        latents = pipe.inference_denoising_step(
                            timesteps_idx,
                            t,
                            preparation_phase_output,
                            step_index=timesteps_idx,
                        )
                        # calculate grad w.r.t. lambda
                        lamb_grads = torch.autograd.grad(
                            latents,
                            trainable_lambs,
                            grad_outputs=grad,
                            retain_graph=True,
                        )

                        for lamb, lamb_grad in zip(trainable_lambs, lamb_grads):
                            if lamb.grad is None:
                                lamb.grad = lamb_grad
                            else:
                                lamb.grad += lamb_grad

                        grad = torch.autograd.grad(latents, current_latents, grad_outputs=grad)

                if (idx * batch_size) % args.mce_accumulate_grad_batches == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    mean_loss_reconstruct, mean_intermediate_loss = 0, 0
                    pbar.update()
                else:
                    mean_loss_reconstruct += loss_reconstruct.item() / args.mce_accumulate_grad_batches

            print(f"epoch {i+1}/{args.mce_epochs}")

    # save final pipe
    pipe.save_pretrained(args.save_dir)


def main(args: Arguments):
    train(args)
