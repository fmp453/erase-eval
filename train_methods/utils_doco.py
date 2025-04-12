import os
import random
import regex as re
import requests
from io import BytesIO
from pathlib import Path

import openai
import numpy as np
import torch
import torch.nn as nn
from clip_retrieval.clip_client import ClipClient
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
from diffusers.models.attention_processor import Attention
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose(
    [
        transforms.Resize(288),
        transforms.ToTensor(),
        normalize,
    ]
)

class CustomDiffusionAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.cross_attention_norm:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

class CustomDiffusionPipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker","feature_extractor", "modifier_token_id"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        feature_extractor: CLIPFeatureExtractor,
        modifier_token_id: list = [],
    ):
        super().__init__(vae,
                         text_encoder,
                         tokenizer,
                         unet,
                         scheduler,
                         None,
                         feature_extractor,
                         None)

        self.modifier_token_id = modifier_token_id

    def save_pretrained(self, save_path, parameter_group="cross-attn", all=False):
        if all:
            super().save_pretrained(save_path)
        else:
            delta_dict = {'unet': {}}
            if parameter_group == 'embedding':
                delta_dict['text_encoder'] = self.text_encoder.state_dict()
            for name, params in self.unet.named_parameters():
                if parameter_group == "cross-attn":
                    if 'attn2.to_k' in name or 'attn2.to_v' in name:
                        delta_dict['unet'][name] = params.cpu().clone()
                elif parameter_group == "full-weight":
                    delta_dict['unet'][name] = params.cpu().clone()
                else:
                    raise ValueError(
                        "parameter_group argument only supports one of [cross-attn, full-weight, embedding]"
                    )
            torch.save(delta_dict, save_path)

    def load_model(self, save_path):
        st = torch.load(save_path)
        print(st.keys())
        if 'text_encoder' in st:
            self.text_encoder.load_state_dict(st['text_encoder'])
        for name, params in self.unet.named_parameters():
            if name in st['unet']:
                params.data.copy_(st['unet'][f'{name}'])

class PatchGANDiscriminator(nn.Module):
    # input channel = 4 for latent noise, 3 for real img
    def __init__(self, input_channel=4):
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channel, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img) -> torch.Tensor:
        return self.model(img)

def retrieve(class_prompt, class_images_dir, num_class_images, save_images=False):
    factor = 1.5
    num_images = int(factor * num_class_images)
    client = ClipClient(
        url="https://knn.laion.ai/knn-service",
        indice_name="laion_400m",
        num_images=num_images,
        aesthetic_weight=0.1,
    )

    os.makedirs(f"{class_images_dir}/images", exist_ok=True)
    if len(list(Path(f"{class_images_dir}/images").iterdir())) >= num_class_images:
        return

    while True:
        class_images = client.query(text=class_prompt)
        if len(class_images) >= factor * num_class_images or num_images > 1e4:
            break
        else:
            num_images = int(factor * num_images)
            client = ClipClient(
                url="https://knn.laion.ai/knn-service",
                indice_name="laion_400m",
                num_images=num_images,
                aesthetic_weight=0.1,
            )

    count = 0
    total = 0
    pbar = tqdm(desc="downloading real regularization images", total=num_class_images)

    if save_images:
        with open(f"{class_images_dir}/caption.txt", "w") as f1, open(
            f"{class_images_dir}/urls.txt", "w"
        ) as f2, open(f"{class_images_dir}/images.txt", "w") as f3:
            while total < num_class_images:
                images = class_images[count]
                count += 1
                try:
                    img = requests.get(images["url"])
                    if img.status_code == 200:
                        _ = Image.open(BytesIO(img.content))
                        with open(f"{class_images_dir}/images/{total}.jpg", "wb") as f:
                            f.write(img.content)
                        f1.write(images["caption"] + "\n")
                        f2.write(images["url"] + "\n")
                        f3.write(f"{class_images_dir}/images/{total}.jpg" + "\n")
                        total += 1
                        pbar.update(1)
                    else:
                        continue
                except:
                    continue
    else:
        with open(f"{class_images_dir}/caption.txt", "w") as f1:
            while count < num_class_images:
                images = class_images[count]
                count += 1
                f1.write(images["caption"] + "\n")
                pbar.update(1)
    return


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    input_anchor_ids = [example["instance_anchor_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    
    input_ids = torch.cat(input_ids, dim=0)
    input_anchor_ids = torch.cat(input_anchor_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "input_anchor_ids": input_anchor_ids,
        "pixel_values": pixel_values,
        "mask": mask.unsqueeze(1),
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt[index % len(self.prompt)]
        example["index"] = index
        return example


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        concept_type,
        tokenizer: CLIPTokenizer,
        size=512,
        center_crop=False,
        hflip=False,
        aug=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.Resampling.BILINEAR
        self.aug = aug
        self.concept_type = concept_type

        self.instance_images_path = []
        self.class_images_path = []
        for concept in concepts_list:
            with open(concept["instance_data_dir"], "r") as f:
                inst_images_path = f.read().splitlines()
            with open(concept["instance_prompt"], "r") as f:
                inst_prompt = f.read().splitlines()
            inst_img_path = [
                (x, y, concept["caption_target"])
                for (x, y) in zip(inst_images_path, inst_prompt)
            ]
            self.instance_images_path.extend(inst_img_path)

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image: np.ndarray, scale, resample):
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
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // 8 + 1 : (top + scale) // 8 - 1,
                left // 8 + 1 : (left + scale) // 8 - 1,
            ] = 1.0
        return instance_image, mask

    def __getprompt__(self, instance_prompt, instance_target):
        if self.concept_type == "style":
            r = np.random.choice([0, 1, 2])
            instance_prompt = (
                f"{instance_prompt}, in the style of {instance_target}"
                if r == 0
                else f"in {instance_target}'s style, {instance_prompt}"
                if r == 1
                else f"in {instance_target}'s style, {instance_prompt}"
            )
        elif self.concept_type in ["nudity", "violence"]:
            r = np.random.choice([0, 1, 2])
            instance_prompt = (
                f"{instance_target}, {instance_prompt}"
                if r == 0
                else f"in {instance_target}'s style, {instance_prompt}"
                if r == 1
                else f"in {instance_target}'s style, {instance_prompt}"
            )
        elif self.concept_type == "object":
            anchor, target = instance_target.split("+")
            instance_prompt = instance_prompt.replace(anchor, target)
        elif self.concept_type == "memorization":
            instance_prompt = instance_target.split("+")[1]
        return instance_prompt

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        example = {}
        instance_image, instance_prompt, instance_target = self.instance_images_path[
            index % self.num_instance_images
        ]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)
        # modify instance prompt according to the concept_type to include target concept
        # multiple style/object fine-tuning
        if ";" in instance_target:
            instance_target = instance_target.split(";")
            instance_target = instance_target[index % len(instance_target)]

        instance_anchor_prompt = instance_prompt
        instance_prompt = self.__getprompt__(instance_prompt, instance_target)
        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(
            instance_image, random_scale, self.interpolation
        )

        if random_scale < 0.6 * self.size:
            instance_prompt = (
                np.random.choice(["a far away ", "very small "]) + instance_prompt
            )
        elif random_scale > self.size:
            instance_prompt = (
                np.random.choice(["zoomed in ", "close up "]) + instance_prompt
            )

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

def get_anchor_prompts(
    class_prompt,
    concept_type,
    num_class_images=200,
):
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.Client(api_key=api_key)
    class_prompt_collection = []
    caption_target = []
    if concept_type == "object":
        messages = [
            {
                "role": "system",
                "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate.",
            }
        ]
        messages = [
            {
                "role": "user",
                "content": f'Generate {num_class_images} captions for images containing a {class_prompt}. The caption should also contain the word "{class_prompt}" ',
            }
        ]
        while True:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06", 
                messages=messages
            )
            class_prompt_collection += [
                x
                for x in completion.choices[0].message.content.lower().split("\n")
                if class_prompt in x
            ]
            messages.append(
                {"role": "assistant", "content": completion.choices[0].message.content}
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Generate {num_class_images-len(class_prompt_collection)} more captions",
                }
            )
            if len(class_prompt_collection) >= num_class_images:
                break
        class_prompt_collection = clean_prompt(class_prompt_collection)[:num_class_images]

    return class_prompt_collection, ";*+".join(caption_target)


def clean_prompt(class_prompt_collection):
    class_prompt_collection = [
        re.sub(r"[0-9]+", lambda num: "" * len(num.group(0)), prompt)
        for prompt in class_prompt_collection
    ]
    class_prompt_collection = [
        re.sub(r"^\.+", lambda dots: "" * len(dots.group(0)), prompt)
        for prompt in class_prompt_collection
    ]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    class_prompt_collection = [x.replace('"', "") for x in class_prompt_collection]
    return class_prompt_collection

def safe_dir(dir: Path):
    if not dir.exists():
        dir.mkdir()
    return dir

import torch

def adjust_gradient(model: nn.Module, optim: torch.optim.Optimizer, norm_grad, loss_a: torch.Tensor, loss_b: torch.Tensor, lambda_=1):
    # Clear gradients
    optim.zero_grad()

    # Calculate gradients for loss_b
    loss_b.backward(retain_graph=True)
    norm_grad()
    b_grads = [p[1].grad.clone() for p in model.named_parameters() if ("attn2" in p[0] and p[1].grad != None)]
    # Clear gradients
    optim.zero_grad()

    # Calculate gradients for loss_a
    loss_a.backward()
    norm_grad()

    # Gradient adjustment
    # Iterate through model parameters and adjust gradients
    for (p, b_grad) in zip([p[1] for p in model.named_parameters() if ("attn2" in p[0] and p[1].grad != None)], b_grads):
        if p.grad is not None and b_grad is not None:
            # Normalize gradients
            b_grad_norm = b_grad / (torch.linalg.norm(b_grad) + 1e-8)
            a_grad_norm = p.grad / (torch.linalg.norm(p.grad) + 1e-8)
            # Calculate dot product between gradients
            dot_product = torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten())
            # If gradients are in opposite directions, adjust gradient
            if dot_product < 0:
                adjustment = lambda_ * dot_product * b_grad_norm
                p.grad -= adjustment

    # Apply gradient updates
    optim.step()
    optim.zero_grad()