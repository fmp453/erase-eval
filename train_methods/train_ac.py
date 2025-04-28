# Ablating Concepts in Text-to-Image Diffusion Models (AC)

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict
from itertools import product

from PIL import Image
from tqdm import trange
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL, StableDiffusionPipeline

from train_methods.train_utils import collate_fn
from utils import Arguments

warnings.filterwarnings("ignore")

# model-based concept ablation
# ref: https://huggingface.co/spaces/nupurkmr9/concept-ablation/blob/main/concept-ablation-diffusers/train.py

class CustomDataset(Dataset):
    # ref: https://huggingface.co/spaces/nupurkmr9/concept-ablation/blob/main/concept-ablation-diffusers/utils.py
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, concept_type, image_dir, prompt_path, tokenizer, concept, anchor_concept=None, size=512, hflip=False, aug=True):
        
        self.size = size
        self.tokenizer = tokenizer
        self.interpolation = Image.Resampling.BILINEAR
        self.aug = aug
        self.concept_type = concept_type
        self.concept = concept
        self.anchor_concept = anchor_concept

        self.instance_images_path = []
        self.class_images_path = []
        inst_images_path = []
        for i, j in product(range(200), range(5)):
            inst_images_path.append(f"{image_dir}/{i:03}-{j}.png")
        inst_prompt = pd.read_csv(prompt_path)["prompt"].to_list()
        inst_prompt = [x.lower() for x in inst_prompt]
        
        # caption_target : prompt
        # class_prompt or instance prompt: anchor prompt
        for i in range(200):
            for j in range(5):
                self.instance_images_path.append((inst_images_path[i * 5 + j], inst_prompt[i], self.concept))

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose([
            self.flip,
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // 8, self.size // 8))
        if scale > self.size:
            instance_image = image[top: top + inner, left: left + inner, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            instance_image[top: top + inner, left: left + inner, :] = image
            mask[top // 8 + 1: (top + scale) // 8 - 1, left // 8 + 1: (left + scale) // 8 - 1] = 1.
        return instance_image, mask

    def __getprompt__(self, instance_prompt: str, instance_target):
        if self.concept_type == 'style':
            r = np.random.choice([0, 1, 2])
            instance_prompt = f'{instance_prompt}, in the style of {instance_target}' if r == 0 else f'in {instance_target}\'s style, {instance_prompt}' if r == 1 else f'in {instance_target}\'s style, {instance_prompt}'
        elif self.concept_type == 'object':
            # cat+grumpy cat
            # anchor, target = instance_target.split('+')
            instance_prompt = instance_prompt.replace(self.anchor_concept, self.concept)
        return instance_prompt

    def __getitem__(self, index):
        instance_image, instance_prompt, instance_target = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)
        # modify instance prompt according to the concept_type to include target concept
        # multiple style/object fine-tuning
        if ';' in instance_target:
            instance_target = instance_target.split(';')
            instance_target = instance_target[index % len(instance_target)]

        instance_anchor_prompt = instance_prompt
        instance_prompt = self.__getprompt__(instance_prompt, instance_target)
        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = np.random.randint(self.size // 3, self.size + 1) if np.random.uniform() < 0.66 else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example = {}
        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["instance_anchor_prompt_ids"] = self.tokenizer(
            instance_anchor_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example

def train(args: Arguments):
    
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    scheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    
    device = args.device.split(",")[0]
    device = f"cuda:{device}"

    text_encoder.eval()
    vae.eval()
    unet.to(device)
    vae.to(device)
    text_encoder.to(device)
    
    for name, params in unet.named_parameters():
        if args.ac_method == "xattn":
            if "attn2" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif args.ac_method == "full":
            params.requires_grad = True
        else:
            raise ValueError(f"Unknown finetuning method: {args.ac_method}")

    cnt = 0
    tot = 0
    for param in unet.parameters():
        tot += param.numel()
        if param.requires_grad:
            cnt += param.numel()
    
    print(f"{cnt / tot * 100}% parameters are updated.")

    optimizer = optim.Adam(unet.parameters(), lr=args.ac_lr, weight_decay=1e-2)
    dataset = CustomDataset(
        concept_type=args.ac_concept_type,
        image_dir=args.ac_img_dir,
        prompt_path=args.ac_prompt_path,
        tokenizer=tokenizer,
        concept=args.concepts,
        anchor_concept=args.anchor_concept,
        size=512,
        hflip=True,
        aug=(args.ac_concept_type != "style")
    )

    dataloader = DataLoader(dataset, batch_size=args.ac_batch_size, num_workers=2, shuffle=True, collate_fn=lambda examples: collate_fn(examples))

    print(f"{len(dataloader)=}")

    pbar = trange(0, 1 * len(dataloader), desc="step")
    unet.train()

    for _ in pbar:
        optimizer.zero_grad()
        batch = next(iter(dataloader))

        with torch.no_grad():
            latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
            text_embedding = text_encoder(batch["input_ids"].to(device))[0]
            anchor_embedding = text_encoder(batch["input_anchor_ids"].to(device))[0]
            latents = latents * vae.config.scaling_factor
        
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()

        noisy_latens = scheduler.add_noise(latents, noise, timesteps)
        noise_pred = unet(noisy_latens, timesteps, text_embedding).sample

        with torch.no_grad():
            anchor_pred = unet(noisy_latens[:anchor_embedding.size(0)], timesteps[:anchor_embedding.size(0)], anchor_embedding).sample
        
        mask = batch["mask"].to(device)

        loss: torch.Tensor = F.mse_loss(noise_pred, anchor_pred, reduction="none")
        loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
    
        loss.backward()
        optimizer.step()
        pbar.set_postfix(OrderedDict(loss=loss.detach().item()))
    
    unet.save_pretrained(args.save_dir)

def generation(args: Arguments):
    print("generate images for Ablating Concepts")

    device = args.device.split(",")[0]
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_version).to(device=f"cuda:{device}")
    pipe.safety_checker = None
    df = pd.read_csv(args.ac_prompt_path)
    prompts = df["prompt"].tolist()
    os.makedirs(args.ac_img_dir, exist_ok=True)
    
    for i in range(200):
        images = pipe(prompts[i], num_images_per_prompt=5).images
        for j in range(5):
            images[j].save(f"{args.ac_img_dir}/{i:03}-{j}.png")

    del pipe
    torch.cuda.empty_cache()

def main(args: Arguments):
    # generate images for Ablating Concepts
    generation(args)
    # main part of Ablating Concepts
    train(args)

