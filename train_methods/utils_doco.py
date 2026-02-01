import os
import regex as re
import requests
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from clip_retrieval.clip_client import ClipClient
from openai import OpenAI
from PIL import Image
from torchvision import transforms
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

        key: torch.Tensor = attn.to_k(encoder_hidden_states)
        value: torch.Tensor = attn.to_v(encoder_hidden_states)
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
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, None, feature_extractor, None)
        self.modifier_token_id = modifier_token_id

    def save_pretrained(self, save_path, parameter_group="cross-attn", all=False):
        if all:
            super().save_pretrained(save_path)
        else:
            delta_dict = {'unet': {}}
            if parameter_group == 'embedding':
                assert isinstance(self.text_encoder, CLIPTextModel)
                delta_dict['text_encoder'] = self.text_encoder.state_dict()
            assert isinstance(self.unet, UNet2DConditionModel)
            for name, params in self.unet.named_parameters():
                assert isinstance(params, torch.Tensor)
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

    Path(f"{class_images_dir}/images").mkdir(exist_ok=True)
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
                except:
                    continue
    else:
        with open(f"{class_images_dir}/caption.txt", "w") as f1:
            while count < num_class_images:
                images = class_images[count]
                count += 1
                f1.write(images["caption"] + "\n")
                pbar.update(1)


def get_anchor_prompts(
    class_prompt,
    concept_type,
    num_class_images=200,
):
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
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
                model="gpt-4.1-2025-04-14",
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


def adjust_gradient(model: nn.Module, optim: torch.optim.Optimizer, norm_grad, loss_a: torch.Tensor, loss_b: torch.Tensor, lambda_=1):
    optim.zero_grad()

    loss_b.backward(retain_graph=True)
    norm_grad()
    b_grads = [p[1].grad.clone() for p in model.named_parameters() if ("attn2" in p[0] and p[1].grad != None)]
    optim.zero_grad()
    loss_a.backward()
    norm_grad()

    # Gradient adjustment
    # Iterate through model parameters and adjust gradients
    for (p, b_grad) in zip([p[1] for p in model.named_parameters() if ("attn2" in p[0] and p[1].grad != None)], b_grads):
        if p.grad is not None and b_grad is not None:
            # Normalize gradients
            b_grad_norm: torch.Tensor = b_grad / (torch.linalg.norm(b_grad) + 1e-8)
            a_grad_norm: torch.Tensor = p.grad / (torch.linalg.norm(p.grad) + 1e-8)
            # Calculate dot product between gradients
            dot_product = torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten())
            # If gradients are in opposite directions, adjust gradient
            if dot_product < 0:
                adjustment = lambda_ * dot_product * b_grad_norm
                p.grad -= adjustment

    optim.step()
    optim.zero_grad()
