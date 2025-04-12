import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import clip
import random
import cv2


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import Attention
from diffusers.optimization import get_scheduler

from train_methods.utils_doco import get_anchor_prompts, collate_fn, retrieve, adjust_gradient
from train_methods.utils_doco import CustomDiffusionAttnProcessor, PatchGANDiscriminator, CustomDiffusionDataset, PromptDataset, CustomDiffusionPipeline
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
            # check  wether the layer is cross attention
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


def save_model_card(
    repo_id: str, images=None, base_model=str, prompt=str, repo_folder=None
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"./image_{i}.png\n"

    yaml = f"""
        ---
        license: creativeml-openrail-m
        base_model: {base_model}
        instance_prompt: {prompt}
        tags:
        - stable-diffusion
        - stable-diffusion-diffusers
        - text-to-image
        - diffusers
        - custom diffusion
        inference: true
        ---
            """
    model_card = f"""
        # Custom Diffusion - {repo_id}

        These are Custom Diffusion adaption weights for {base_model}. The weights were trained on {prompt} using [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion). You can find some example images in the following. \n
        {img_str[0]}
        """
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def freeze_params(params: nn.Parameter) -> None:
    for param in params:
        param.requires_grad = False

def seed_everything(seed: int=42) -> None:
    """
    Seed everything to make results reproducible.
    :param seed: An integer to use as the random seed.
    """
    random.seed(seed)        # Python random module
    np.random.seed(seed)     # Numpy module
    torch.manual_seed(seed)  # PyTorch
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--concept_type",
        type=str,
        required=True,
        choices=["style", "object", "memorization", "nudity", "violence"],
        help="the type of removed concepts",
    )
    parser.add_argument(
        "--caption_target",
        type=str,
        required=True,
        help="target style to remove, used when kldiv loss",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--mem_impath",
        type=str,
        default="",
        help="the path to saved memorized image. Required when concept_type is memorization",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=1000,
        help="the number of generated images used for ablating the concept",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--loss_type_reverse",
        type=str,
        default="model-based",
        help="loss type for reverse fine-tuning",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--hflip", action="store_true", help="Apply horizontal flip data augmentation."
    )
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )
    parser.add_argument(
        "--warm_up",
        type=int,
        default=500,
        help="Number of steps for the warmup training of the discriminator.",
    )
    parser.add_argument(
        "--gradient_clip",
        action="store_true",
        help="Apply gradient clip.",
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=1.0,
        help="The gradient scale for the discriminator.",
    )
    parser.add_argument(
        "--multi_ckpt_path",
        type=str,
        default="",
        help="Multi concept remove checkpoint path.",
    )
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.checkpointing_steps = args.max_train_steps / 10

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation." )
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def main(args: Arguments):
    
    seed_everything(args.seed)
    device = args.device.split(",")[0]
    device = f"cuda:{device}"
    
    # need to fix
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir,
                "caption_target": " ".join(args.caption_target.split("-")),
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Generate class images if prior preservation is enabled.
    for i, concept in enumerate(args.concepts_list):
        # directly path to ablation images and its corresponding prompts is provided.
        if (
            concept["instance_prompt"] is not None
            and concept["instance_data_dir"] is not None
        ):
            break

        class_images_dir = Path(concept["class_data_dir"])
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(f"{class_images_dir}/images", exist_ok=True)

        # we need to generate training images
        if (len(list(Path(os.path.join(class_images_dir, "images")).iterdir()))< args.doco_num_class_images):

            pipeline = DiffusionPipeline.from_pretrained(args.sd_version)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline.safety_checker = None
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to(device)

            # need to create prompts using class_prompt.
            if not os.path.isfile(concept["class_prompt"]):
                # style based prompts are retrieved from laion dataset
                if args.doco_concept_type in ["style", "nudity", "violence"]:
                    name = "images"
                    if (
                        not Path(os.path.join(class_images_dir, name)).exists()
                        or len(list(Path(os.path.join(class_images_dir, name)).iterdir())) < args.doco_num_class_images
                    ):
                        retrieve(
                            concept["class_prompt"],
                            class_images_dir,
                            args.doco_num_class_prompts,
                            save_images=False,
                        )
                    with open(os.path.join(class_images_dir, "caption.txt")) as f:
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
            print(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(class_prompt_collection, num_new_images)
            sample_dataloader = DataLoader(sample_dataset, batch_size=4)

            if Path(f"{class_images_dir}/caption.txt").exists():
                os.remove(f"{class_images_dir}/caption.txt")
            if Path(f"{class_images_dir}/images.txt").exists():
                os.remove(f"{class_images_dir}/images.txt")


            for example in tqdm(sample_dataloader, desc="Generating class images"):
                with open(f"{class_images_dir}/caption.txt", "a") as f1, open(
                    f"{class_images_dir}/images.txt", "a"
                ) as f2:
                    images: list[np.ndarray] = pipeline(
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

        concept["class_prompt"] = os.path.join(class_images_dir, "caption.txt")
        concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
        concept["instance_prompt"] = os.path.join(class_images_dir, "caption.txt")
        concept["instance_data_dir"] = os.path.join(class_images_dir, "images.txt")

        torch.cuda.empty_cache()


    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    shadow_unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")

    # set shadow_unet requires_grad False
    for param in shadow_unet.parameters():
        param.requires_grad = False
    
    vae.requires_grad_(False)
    if args.doco_parameter_group != "embedding":
        text_encoder.requires_grad_(False)

    unet = create_custom_diffusion(unet, args.doco_parameter_group)
    weight_dtype = torch.float32
    vae.to(device, dtype=weight_dtype)

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    if args.doco_parameter_group == "embedding":
        assert (args.doco_concept_type != "memorization"), "embedding finetuning is not supported for memorization"

        for concept in args.concept_list:
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
        lr=args.doco_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    discriminator, criterion, optimizer_D = init_discriminator(lr=args.doco_dlr)

    # Dataset and DataLoaders creation:
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        concept_type=args.doco_concept_type,
        tokenizer=tokenizer,
        size=512,
        center_crop=args.center_crop,
        hflip=args.hflip,
        aug=not args.noaug,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.doco_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # discriminator lr scheduler
    lr_scheduler_D = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_D,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    num_update_steps_per_epoch = len(train_dataloader)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.doco_batch_size

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.doco_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):

            # Convert images to latent space
            latents: torch.Tensor = vae.encode(
                batch["pixel_values"].to(dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents).to(latents.device)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
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
            if args.loss_type_reverse == "model-based":
                target = model_pred_anchor
            else:
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            target: torch.Tensor = noise_scheduler.step_batch(target, timesteps, noisy_latents).pred_original_sample   # anchor
            model_pred: torch.Tensor = noise_scheduler.step_batch(model_pred, timesteps, noisy_latents).pred_original_sample    # concept to erasing

            def norm_grad():
                params_to_clip = (
                    itertools.chain(text_encoder.parameters())
                    if args.doco_parameter_group == "embedding"
                    else itertools.chain(
                        [x[1] for x in unet.named_parameters() if ("attn2" in x[0])]
                    )
                    if args.doco_parameter_group == "cross-attn"
                    else itertools.chain(unet.parameters())
                )
                nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            loss_G: Variable = Variable(torch.zeros(1))
            if global_step > args.warm_up:
                out = discriminator(model_pred).squeeze(1)
                loss = criterion(out, torch.zeros_like(out))

                loss_G = loss
                # prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                if args.gradient_clip:
                    adjust_gradient(unet, optimizer, norm_grad, loss, prior_loss, lambda_=args.lambda_)
                else:
                    loss_G.backward()
                    norm_grad()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

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

            if global_step >= args.max_train_steps:
                break


    pipeline = CustomDiffusionPipeline.from_pretrained(
        args.sd_version,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        modifier_token_id=modifier_token_id,
    )
    save_path = os.path.join(args.output_dir, "delta.bin")
    pipeline.save_pretrained(save_path, parameter_group=args.doco_parameter_group)                


if __name__ == "__main__":
    args = parse_args()
    main(args)
