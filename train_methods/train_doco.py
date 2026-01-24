import hashlib
import itertools
import math
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import Attention
from diffusers.optimization import get_scheduler

from train_methods.utils_doco import get_anchor_prompts, retrieve, adjust_gradient
from train_methods.utils_doco import CustomDiffusionAttnProcessor, PatchGANDiscriminator, CustomDiffusionPipeline
from train_methods.train_utils import collate_fn, get_devices, get_models, seed_everything
from train_methods.data import DocoDataset, DocoPromptDataset
from utils import Arguments


def init_discriminator(lr=0.0001, b1=0.5, b2=0.999) -> tuple[PatchGANDiscriminator, nn.BCEWithLogitsLoss, optim.Optimizer]:
    discriminator = PatchGANDiscriminator()
    criterion = nn.BCEWithLogitsLoss()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    return discriminator, criterion, optimizer_D


def set_use_memory_efficient_attention(self, use_memory_efficient_attention_xformers: bool):
    processor = CustomDiffusionAttnProcessor()
    self.set_processor(processor)


def create_custom_diffusion(unet: UNet2DConditionModel, parameter_group):
    for name, params in unet.named_parameters():
        if parameter_group == "cross-attn":
            if "attn2.to_k" in name or "attn2.to_v" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif parameter_group == "full-weight":
            params.requires_grad = True
        elif parameter_group == "embedding":
            params.requires_grad = False
        else:
            raise ValueError("parameter_group argument only cross-attn, full-weight, embedding")

    # change attn class
    def change_attn(unet: UNet2DConditionModel):
        for layer in unet.children():
            # check wether the layer is cross attention
            # CrossAttention was renamed to Attention in commit hash e828232
            # https://github.com/huggingface/diffusers/issues/4969
            if isinstance(layer, Attention):
                bound_method = set_use_memory_efficient_attention.__get__(layer, layer.__class__)
                setattr(layer, "set_use_memory_efficient_attention", bound_method)
            else:
                change_attn(layer)

    change_attn(unet)
    unet.set_attn_processor(CustomDiffusionAttnProcessor())
    return unet

def freeze_params(params: nn.Parameter) -> None:
    for param in params:
        param.requires_grad = False


def main(args: Arguments):
    
    seed_everything(args.seed)
    device = get_devices(args)[0]
    
    concepts_list = [
        {
            "instance_prompt": args.concepts,
            "class_prompt": args.concepts,
            "instance_data_dir": args.instance_data_dir,
            "class_data_dir": "for-doco/",
            "caption_target": " ",
        }
    ]
    
    # Generate class images if prior preservation is enabled.
    for i, concept in enumerate(concepts_list):
        # directly path to ablation images and its corresponding prompts is provided.
        if (concept["instance_prompt"] is not None and concept["instance_data_dir"] is not None):
            break

        class_images_dir = Path(concept["class_data_dir"])
        class_images_dir.mkdir(parents=True, exist_ok=True)
        Path(f"{class_images_dir}/images").mkdir(exist_ok=True)

        # we need to generate training images
        if len(list(Path(class_images_dir, "images").iterdir())) < args.doco_num_class_images:

            pipeline: StableDiffusionPipeline = DiffusionPipeline.from_pretrained(args.sd_version)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline.safety_checker = None
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to(device)

            # need to create prompts using class_prompt.
            if not Path(concept["class_prompt"]).is_file():
                # style based prompts are retrieved from laion dataset
                if args.doco_concept_type in ["style", "nudity", "violence"]:
                    name = "images"
                    if (
                        not Path(class_images_dir, name).exists()
                        or len(list(Path(class_images_dir, name).iterdir())) < args.doco_num_class_images
                    ):
                        retrieve(
                            concept["class_prompt"],
                            class_images_dir,
                            args.doco_num_class_prompts,
                            save_images=False,
                        )
                    with open(Path(class_images_dir, "caption.txt")) as f:
                        class_prompt_collection = [x.strip() for x in f.readlines()]

                # LLM based prompt collection.
                else:
                    class_prompt = concept["class_prompt"]
                    # in case of object query chatGPT to generate captions containing the anchor category
                    if args.doco_concept_type == "object":
                        class_prompt_collection, _ = get_anchor_prompts(
                            class_prompt,
                            args.doco_concept_type,
                            args.doco_num_class_prompts,
                        )
                        with open(class_images_dir / "caption_anchor.txt", "w") as f:
                            for prompt in class_prompt_collection:
                                f.write(prompt + "\n")
            # class_prompt is filepath to prompts.
            else:
                with open(concept["class_prompt"]) as f:
                    class_prompt_collection = [x.strip() for x in f.readlines()]

            num_new_images = args.doco_num_class_images

            sample_dataset = DocoPromptDataset(class_prompt_collection, num_new_images)
            sample_dataloader = DataLoader(sample_dataset, batch_size=4)

            if Path(f"{class_images_dir}/caption.txt").exists():
                Path(f"{class_images_dir}/caption.txt").unlink()
            if Path(f"{class_images_dir}/images.txt").exists():
                Path(f"{class_images_dir}/images.txt").unlink()


            for example in tqdm(sample_dataloader, desc="Generating class images"):
                with open(f"{class_images_dir}/caption.txt", "a") as f1, open(
                    f"{class_images_dir}/images.txt", "a"
                ) as f2:
                    images: list[Image.Image] = pipeline(
                        example["prompt"],
                        num_inference_steps=25,
                        guidance_scale=6.0,
                        eta=1.0,
                    ).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = Path(class_images_dir / f"images/{example['index'][i]}-{hash_image}.jpg")
                        image.save(str(image_filename))
                        f2.write(str(image_filename) + "\n")
                    f1.write("\n".join(example["prompt"]) + "\n")

            del pipeline

        concept["class_prompt"] = Path(class_images_dir, "caption.txt")
        concept["class_data_dir"] = Path(class_images_dir, "images.txt")
        concept["instance_prompt"] = Path(class_images_dir, "caption.txt")
        concept["instance_data_dir"] = Path(class_images_dir, "images.txt")

        torch.cuda.empty_cache()

    Path(args.save_dir).mkdir(exist_ok=True)

    shadow_unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    tokenizer, text_encoder, vae, unet, _, noise_scheduler = get_models(args)

    # set shadow_unet requires_grad False
    for param in shadow_unet.parameters():
        param.requires_grad = False
    
    vae.requires_grad_(False)
    if args.doco_parameter_group != "embedding":
        text_encoder.requires_grad_(False)

    unet = create_custom_diffusion(unet, args.doco_parameter_group)
    vae.to(device)

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    if args.doco_parameter_group == "embedding":
        assert (args.doco_concept_type != "memorization"), "embedding finetuning is not supported for memorization"

        for concept in concepts_list:
            token_ids = tokenizer.encode([concept["caption_target"]], add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        modifier_token_id += token_ids

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
        params_to_optimize = itertools.chain(text_encoder.get_input_embeddings().parameters())
    elif args.doco_parameter_group == "cross-attn":
        params_to_optimize = itertools.chain([x[1] for x in unet.named_parameters() if ("attn2.to_k" in x[0] or "attn2.to_v" in x[0])])
    elif args.doco_parameter_group == "full-weight":
        params_to_optimize = itertools.chain(unet.parameters())

    optimizer = optim.AdamW(
        params_to_optimize,
        lr=args.doco_lr
    )

    discriminator, criterion, optimizer_D = init_discriminator(lr=args.doco_dlr)

    # Dataset and DataLoaders creation:
    train_dataset = DocoDataset(
        concepts_list=concepts_list,
        concept_type=args.doco_concept_type,
        tokenizer=tokenizer,
        center_crop=args.doco_center_crop,
        hflip=args.doco_hflip,
        aug=not args.doco_noaug,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.doco_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=2,
    )

    num_update_steps_per_epoch = len(train_dataloader)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.doco_lr_warmup_steps,
        num_training_steps=args.doco_max_train_steps,
    )

    # discriminator lr scheduler
    lr_scheduler_D = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_D,
        num_warmup_steps=args.doco_dlr_warmup_steps,
        num_training_steps=args.doco_max_train_steps,
    )

    num_update_steps_per_epoch = len(train_dataloader)
    args.doco_num_train_epochs = math.ceil(args.doco_max_train_steps / num_update_steps_per_epoch)

    # Train
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.doco_max_train_steps))
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.doco_num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):

            # Convert images to latent space
            latents: torch.Tensor = vae.encode(batch["pixel_values"].to(vae.device)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents).to(latents.device)
            bsz = latents.shape[0]
            timesteps = torch.randint(0,noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            text_encoder.to(device)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            encoder_anchor_hidden_states: torch.Tensor = text_encoder(batch["input_anchor_ids"])[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            with torch.no_grad():
                model_pred_anchor = shadow_unet(
                    noisy_latents[: encoder_anchor_hidden_states.size(0)],
                    timesteps[: encoder_anchor_hidden_states.size(0)],
                    encoder_anchor_hidden_states,
                ).sample

            # Get the target for loss depending on the prediction type
            if args.doco_loss_type_reverse == "model-based":
                target = model_pred_anchor
            else:
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # maybe need to be fixed
            target: torch.Tensor = noise_scheduler.step_batch(target, timesteps, noisy_latents).pred_original_sample   # anchor
            target_prior = torch.chunk(noise, 2, dim=0)[1]
            model_pred: torch.Tensor = noise_scheduler.step_batch(model_pred, timesteps, noisy_latents).pred_original_sample    # concept to erasing
            _, model_pred_prior = torch.chunk(model_pred, 2, dim=0)

            def norm_grad():
                params_to_clip = (
                    itertools.chain(text_encoder.parameters())
                    if args.doco_parameter_group == "embedding"
                    else itertools.chain([x[1] for x in unet.named_parameters() if ("attn2" in x[0])])
                    if args.doco_parameter_group == "cross-attn"
                    else itertools.chain(unet.parameters())
                )
                nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            loss_G: Variable = Variable(torch.zeros(1))
            if global_step > args.doco_dlr_warmup_steps:
                out = discriminator(model_pred).squeeze(1)
                loss = criterion(out, torch.zeros_like(out))

                loss_G = loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                # 公式実装では、prior_lossが未定義で実行される可能性がある不具合を抱えているので実験結果に疑義あり?
                adjust_gradient(unet, optimizer, norm_grad, loss, prior_loss, lambda_=args.doco_lambda_)
                
            real_out = discriminator(target.detach()).squeeze(1)
            loss_real_D = criterion(real_out, torch.zeros_like(real_out))

            fake_out = discriminator(model_pred.detach()).squeeze(1)
            loss_fake_D = criterion(fake_out, torch.ones_like(fake_out)) 

            loss_D: torch.Tensor = loss_fake_D + loss_real_D

            loss_D.backward()

            # Zero out the gradients for all token embeddings except the newly added
            # embeddings for the concept, as we only want to optimize the concept embeddings
            if args.doco_parameter_group == "embedding":
                grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = (torch.arange(len(tokenizer)) != modifier_token_id[0])
                for i in range(len(modifier_token_id[1:])):
                    index_grads_to_zero = index_grads_to_zero & (torch.arange(len(tokenizer)) != modifier_token_id[i])
                grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)
            
            optimizer_D.step()
            lr_scheduler_D.step()
            optimizer_D.zero_grad()
            torch.cuda.empty_cache()

            progress_bar.update(1)
            global_step += 1

            logs = {"loss_D": loss_D.detach().item(), "lr_D": lr_scheduler_D.get_last_lr()[0], "loss_G": loss_G.detach().item(), "lr_G": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.doco_max_train_steps:
                break

    pipeline = CustomDiffusionPipeline.from_pretrained(
        args.sd_version,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        modifier_token_id=modifier_token_id,
    )
    save_path = Path(args.save_dir, "delta.bin")
    pipeline.save_pretrained(save_path, parameter_group=args.doco_parameter_group)
