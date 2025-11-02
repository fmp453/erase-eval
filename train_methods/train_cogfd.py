# official repo: https://github.com/Sirius11311/CoGFD-ICLR25

"""
usage of official repo

1. generate training images 

python img_prepare.py --concept_combination "underage_and_alcohol"

2. unlearning

python concept_combination_erasing.py \
    --combine_concept_x "underage_and_alcohol" \
    --combine_theme_y "normal_life" \
    --p1 -1 \
    --p2 1 \
    --lr 2.5e-5 \
    --max-steps 130 \
    --iterate_n 2 
"""


import itertools
import math
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from diffusers.models.attention_processor import Attention
from transformers import AutoTokenizer, PretrainedConfig
from transformers import CLIPTextModel

from train_methods.data import COGFDDataset
from train_methods.utils_cogfd import RobertaSeriesModelWithTransformation, generate_and_save_iterative_graphs, extract_concept_from_graph
from train_methods.train_utils import get_devices
from utils import Arguments

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str) -> CLIPTextModel | RobertaSeriesModelWithTransformation:
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples, with_prior_preservation=False) -> dict:
    pixel_values = [example["instance_images"] for example in examples]
    source_prompts = [example["concept"] for example in examples]
    source_ids = [example["prompt_ids"] for example in examples]
    source_labels = [example["label"] for example in examples]
    source_mask = [example["attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    source_labels = torch.Tensor(source_labels).float()
    source_ids = torch.cat(source_ids, dim=0)
    source_mask = torch.cat(source_mask, dim=0)

    batch = {
        "source_prompts": source_prompts,
        "source_labels": source_labels,
        "source_ids": source_ids,
        "source_mask": source_mask,
        "pixel_values": pixel_values,
    }
    return batch

class HiddenStatesController:  
    def __init__(self) -> None:
        self.encoder_attn_mask = []

    def set_encoder_attn_mask(self, attn_mask):
        self.encoder_attn_mask = attn_mask

    def zero_attn_probs(self):
        self.encoder_attn_mask = []


class MyCrossAttnProcessor:

    def __init__(self, hiddenstates_controller: "HiddenStatesController", module_name) -> None:
        self.hiddenstates_controller = hiddenstates_controller
        self.module_name = module_name

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):

        encoder_attention_mask = self.hiddenstates_controller.encoder_attn_mask
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        if encoder_attention_mask is not None and encoder_hidden_states is not None:
            # B x 77 -> B x 4096 x 77
            attention_mask = encoder_attention_mask.unsqueeze(1).repeat(1, hidden_states.size(1), 1)
            attention_mask = attention_mask.repeat_interleave(attn.heads, dim=0).type_as(hidden_states)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def train(
    args: Arguments,
    task_info=["child drinking wine", "underage drinking"],
    concept_combination=[],
    labels=[],
):
    train_batch_size = min(len(concept_combination), args.cogfd_train_batch_size)

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.sd_version,
        subfolder="tokenizer",
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.sd_version)

    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(args.sd_version, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    unet_1: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")

    # unet_1 on device 1
    devices = get_devices(args)[0]

    attn_controller = HiddenStatesController()
    module_count = 0
    for name, module in unet.named_modules():
        if name.endswith('attn2'):
            module.set_processor(MyCrossAttnProcessor(attn_controller, name))
            module_count += 1
    print(f"cross attention module count: {module_count}")

    attn_controller_1 = HiddenStatesController()
    module_count = 0
    for name, module in unet_1.named_modules():
        if name.endswith('attn2'):
            module.set_processor(MyCrossAttnProcessor(attn_controller_1, name))
            module_count += 1
    print(f"cross attention module count: {module_count}")

    vae.requires_grad_(False)
    if not args.cogfd_train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.cogfd_scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.cogfd_use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    if args.cogfd_only_optimize_ca:
        params_to_optimize = (itertools.chain(unet.parameters(), text_encoder.parameters()) if args.cogfd_train_text_encoder else [p for n, p in unet.named_parameters() if 'attn2' in n and 'to_v' not in n])
    else:
        params_to_optimize = (itertools.chain(unet.parameters(), text_encoder.parameters()) if args.cogfd_train_text_encoder else unet.parameters())

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.cogfd_lr,
        betas=(args.cogfd_adam_beta_1, args.cogfd_adam_beta_2),
        weight_decay=args.cogfd_adam_weight_decay,
        eps=args.cogfd_adam_epsilon,
    )

    train_dataset = COGFDDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        size=args.image_size,
        center_crop=args.cogfd_center_crop,
        use_pooler=args.cogfd_use_pooler,
        task_info=task_info,
        concept_combination=concept_combination,
        labels=labels,
    )

    if len(train_dataset) == 0:
        raise ValueError("Dataset is empty. Please check your dataset configuration.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.cogfd_dataloader_num_workers,
        drop_last=True
    )

    if len(train_dataloader) == 0:
        raise ValueError("No batches in the dataloader. Please check your batch_size.")


    gradient_accumulation_steps = args.cogfd_gradient_accumulation_steps

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Ensure we have at least one training step
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.cogfd_lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=args.cogfd_lr_num_cycles,
        power=args.cogfd_lr_power,
    )
    
    vae.to(devices[0])
    unet.to(devices[0])
    unet_1.to(devices[1])
    text_encoder.to(devices[0])

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    total_batch_size = train_batch_size * gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        if args.cogfd_train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):

            with torch.no_grad():
                latents: torch.Tensor = vae.encode(batch["pixel_values"].to(vae.device)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps: torch.Tensor = torch.randint(args.cogfd_start, args.cogfd_end, (bsz, ), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states_source: torch.Tensor = text_encoder(batch["source_ids"].to(text_encoder.device), attention_mask=batch["source_mask"])[0]

            # set concept_positions for this batch
            attn_controller.set_encoder_attn_mask(batch["source_mask"])
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states_source,
            ).sample

            # Predict the noise residual
            with torch.no_grad():
                attn_controller_1.set_encoder_attn_mask(batch["source_mask"])
                noisy_latents_1 = noisy_latents.to(unet_1.device)
                timesteps_1 = timesteps.to(unet_1.device)
                encoder_hidden_states_1 = encoder_hidden_states_source.to(unet_1.device)
                
                model_pred_1: torch.Tensor = unet_1(noisy_latents_1, timesteps_1, encoder_hidden_states_1).sample
                model_pred_1 = model_pred_1.to(unet.device)

            unlearn_select = batch["source_labels"] == args.cogfd_p1
            retain_select = batch["source_labels"] == args.cogfd_p2

            # Ensure all tensors are on the same device for loss computation
            loss_1 = F.mse_loss(model_pred[unlearn_select], model_pred_1[unlearn_select])
            loss_2 = F.mse_loss(model_pred[retain_select], model_pred_1[retain_select])

            # Compute final loss on the same device
            final_loss = 0.1 * torch.exp(-loss_1) + torch.exp(loss_2)
            final_loss.backward()

            params_to_clip = params_to_optimize
            nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.cogfd_set_grads_to_none)
            attn_controller.zero_attn_probs()
            attn_controller_1.zero_attn_probs()

            logs = {
                "loss_1": loss_1.detach().item(),
                "loss_2": loss_2.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    pipeline = DiffusionPipeline.from_pretrained(
        args.sd_version,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer
    )
    pipeline.save_pretrained(args.save_dir)


def main(args: Arguments):
    # first, generate concept logic graph
    graph_path = args.cogfd_graph_path
    if Path(graph_path).exists():
        with open(graph_path, 'r') as f:
            parsed_graph = json.load(f)
    else:
        combine_concept_x = args.cogfd_combine_concept_x.replace("_", " ")
        combine_theme_y = args.cogfd_combine_theme_y.replace("_", " ")
        parsed_graph = generate_and_save_iterative_graphs(combine_concept_x, combine_theme_y, graph_path, iterate_n=args.cogfd_iterate_n)
    
    # second, erasing
    # extract concepts from graph
    concept_combination, sub_concept = extract_concept_from_graph(parsed_graph)

    task_info = [args.cogfd_combine_concept_x, args.cogfd_combine_theme_y]
    train(
        task_info=task_info,
        concept_combination=concept_combination,
        labels=[args.cogfd_p1 for _ in concept_combination] + [args.cogfd_p2 for _ in sub_concept]
    )
