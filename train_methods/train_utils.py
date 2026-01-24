import gc
import inspect
import os
import random
from typing import Any
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL, DDPMScheduler, SchedulerMixin
from diffusers.models.lora import LoRALinearLayer
from diffusers.models.attention_processor import Attention

from custom_text_encoder import CustomCLIPTextModel
from utils import Arguments
from train_methods.consts import LEN_EN_3K_VOCAB, LEN_TOKENIZER_VOCAB

def get_models(args: Arguments) -> tuple[CLIPTokenizer, CLIPTextModel, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler]:
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    ddim_scheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    ddpm_scheduler = DDPMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")

    return tokenizer, text_encoder, vae, unet, ddim_scheduler, ddpm_scheduler


def get_devices(args: Arguments) -> list[torch.device]:
    devices = args.device.split(",")
    if len(devices) > 1:
        return [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[1]}")]
    return [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[0]}")]

def tokenize(prompt: list[str], tokenizer: CLIPTokenizer) -> dict[str, torch.Tensor]:
    return tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def prompt_augmentation(content, sampled_indices=None, concept_type='object') -> list[str]:
    if concept_type == 'object':
        prompts = [
            ("{} in a photo".format(content), content),
            ("{} in a snapshot".format(content), content),
            ("A snapshot of {}".format(content), content),
            ("A photograph showcasing {}".format(content), content),
            ("An illustration of {}".format(content), content),
            ("A digital rendering of {}".format(content), content),
            ("A visual representation of {}".format(content), content),
            ("A graphic of {}".format(content), content),
            ("A shot of {}".format(content), content),
            ("A photo of {}".format(content), content),
            ("A black and white image of {}".format(content), content),
            ("A depiction in portrait form of {}".format(content), content),
            ("A scene depicting {} during a public gathering".format(content), content),
            ("{} captured in an image".format(content), content),
            ("A depiction created with oil paints capturing {}".format(content), content),
            ("An image of {}".format(content), content),
            ("A drawing capturing the essence of {}".format(content), content),
            ("An official photograph featuring {}".format(content), content),
            ("A detailed sketch of {}".format(content), content),
            ("{} during sunset/sunrise".format(content), content),
            ("{} in a detailed portrait".format(content), content),
            ("An official photo of {}".format(content), content),
            ("Historic photo of {}".format(content), content),
            ("Detailed portrait of {}".format(content), content),
            ("A painting of {}".format(content), content),
            ("HD picture of {}".format(content), content),
            ("Magazine cover capturing {}".format(content), content),
            ("Painting-like image of {}".format(content), content),
            ("Hand-drawn art of {}".format(content), content),
            ("An oil portrait of {}".format(content), content),
            ("{} in a sketch painting".format(content), content),
        ]
    elif concept_type == 'style':
        prompts = [
            ("An artwork by {}".format(content), content),
            ("Art piece by {}".format(content), content),
            ("A recent creation by {}".format(content), content),
            ("{}'s renowned art".format(content), content),
            ("Latest masterpiece by {}".format(content), content),
            ("A stunning image by {}".format(content), content),
            ("An art in {}'s style".format(content), content),
            ("Exhibition artwork of {}".format(content), content),
            ("Art display by {}".format(content), content),
            ("a beautiful painting by {}".format(content), content),
            ("An image inspired by {}'s style".format(content), content),
            ("A sketch by {}".format(content), content),
            ("Art piece representing {}".format(content), content),
            ("A drawing by {}".format(content), content),
            ("Artistry showcasing {}".format(content), content),
            ("An illustration by {}".format(content), content),
            ("A digital art by {}".format(content), content),
            ("A visual art by {}".format(content), content),
            ("A reproduction inspired by {}'s colorful, expressive style".format(content), content),
            ("Famous painting of {}".format(content), content),
            ("A famous art by {}".format(content), content),
            ("Artistic style of {}".format(content), content),
            ("{}'s famous piece".format(content), content),
            ("Abstract work of {}".format(content), content),
            ("{}'s famous drawing".format(content), content),
            ("Art from {}'s early period".format(content), content),
            ("A portrait by {}".format(content), content),
            ("An imitation reflecting the style of {}".format(content), content),
            ("An painting from {}'s collection".format(content), content),
            ("Vibrant reproduction of artwork by {}".format(content), content),
            ("Artistic image influenced by {}".format(content), content),
        ] 
    else:
        raise ValueError("unknown concept type.")
    
    if sampled_indices is not None:
        return [prompts[i] for i in sampled_indices if i < len(prompts)]
    return prompts
   
@torch.no_grad()
def encode_prompt(
    prompt: str | list[str]=None,
    negative_prompt: str | list[str]=None,
    removing_prompt: str | list[str]=None,
    num_images_per_prompt: int=1,
    text_encoder: CLIPTextModel=None,
    tokenizer: CLIPTokenizer=None,
    device: torch.device=None,
):
    """Encode a prompt into a text embedding. Prompt can be None."""
    # Get text embeddings for unconditional and conditional prompts.
    if isinstance(prompt, str):
        prompt = [prompt]
    
    if removing_prompt is not None and isinstance(removing_prompt, str):
        removing_prompt = [removing_prompt]
        assert len(prompt) == len(removing_prompt), f"Safety concept must be the same length as prompt of length {len(prompt)}."
    
    if negative_prompt is not None and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]
        assert len(prompt) == len(negative_prompt), f"Negative prompt must be the same length as prompt of length {len(prompt)}."

    batch_size = len(prompt) if prompt is not None else 1

    use_attention_mask = hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask
    device = device if device is not None else text_encoder.device

    # Tokenization
    uncond_input = tokenize([""] * batch_size if negative_prompt is None else negative_prompt, tokenizer)
    prompt_input = tokenize(prompt, tokenizer) if prompt is not None else None    
    removing_input = tokenize(removing_prompt, tokenizer) if removing_prompt is not None else None
    
    # Encoding
    prompt_embeds = text_encoder(input_ids=uncond_input["input_ids"].to(device), attention_mask=uncond_input["attention_mask"].to(device) if use_attention_mask else None)[0]
    if prompt_input is not None:
        prompt_emb = text_encoder(input_ids=prompt_input["input_ids"].to(device), attention_mask=prompt_input["attention_mask"].to(device) if use_attention_mask else None)[0]
        prompt_embeds = torch.cat([prompt_embeds, prompt_emb], dim=0)
    
    if removing_input is not None:
        removing_emb = text_encoder(input_ids=removing_input["input_ids"].to(device), attention_mask=removing_input["attention_mask"].to(device) if use_attention_mask else None)[0]
        prompt_embeds = torch.cat([prompt_embeds, removing_emb], dim=0)

    # Duplicate the embeddings for each image.
    if num_images_per_prompt > 1:
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.reshape(batch_size * num_images_per_prompt, seq_len, -1)
    
    return prompt_embeds

def gather_parameters(method: str, unet: UNet2DConditionModel) -> tuple[list[str], list[torch.nn.Parameter]]:
    """Gather the parameters to be optimized by the optimizer (U-Net)."""
    names, parameters = [], []
    for name, param in unet.named_parameters():
        if method == "full":
            # Train all layers.
            names.append(name)
            parameters.append(param)
        elif method == "selfattn":
            # Attention layer 1 is the self-attention layer.
            if "attn1" in name:
                names.append(name)
                parameters.append(param)
        elif method == "xattn":
            # Attention layer 2 is the cross-attention layer.
            if "attn2" in name:
                names.append(name)
                parameters.append(param)
        elif method == "noxattn":
            # Train all layers except the cross attention and time_embedding layers.
            if name.startswith("conv_out.") or ("time_embed" in name) or ("attn2" in name):
                continue
            else:
                names.append(name)
                parameters.append(param)
        elif method == "notime":
            # Train all layers except the time_embedding layer.
            if name.startswith("conv_out.") or ("time_embed" in name):
                continue
            names.append(name)
            parameters.append(param)
        # for Adaptive-Guided-Erasure: layerの名前が合ってるかはチェックが必要
        elif method == "xlayer":
            if "attn2" in name:
                if "output_blocks.6." in name or "output_blocks.8." in name:
                    parameters.append(name)
        elif method == "selflayer":
            if "attn1" in name:
                if "input_blocks.4." in name or "input_blocks.7." in name:
                    parameters.append(param)
        else:
            raise ValueError(f"Unknown finetuning method: {method}")

    return names, parameters

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

# EAP and AGE
def get_english_tokens():
    data_path = 'data/english_3000.csv'
    df = pd.read_csv(data_path)
    vocab = {}
    for ir, row in df.iterrows():
        vocab[row['word']] = ir
    assert(len(vocab) == LEN_EN_3K_VOCAB)
    return vocab

@torch.no_grad()
def get_vocab(tokenizer: CLIPTokenizer, model_name, vocab='EN3K'):
    if vocab == 'CLIP':
        if model_name == 'SD-v1-4':
            tokenizer_vocab = tokenizer.get_vocab()
        elif model_name == 'SD-v2-1':
            tokenizer_vocab = tokenizer.encoder
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
    elif vocab == 'EN3K':
        tokenizer_vocab = get_english_tokens()
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")
    
    return tokenizer_vocab

@torch.no_grad()
def get_condition(prompt: str | list[str], tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel) -> torch.Tensor:
    token_ids = tokenize([prompt] if isinstance(prompt, str) else prompt, tokenizer).input_ids
    return text_encoder(token_ids.to(text_encoder.device))[0]

@torch.no_grad()
def create_embedding_matrix(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    start=0,
    end=LEN_TOKENIZER_VOCAB,
    model_name='SD-v1-4',
    save_mode='array',
    remove_end_token=False,
    vocab='EN3K'
):

    if type(vocab) == str:
        tokenizer_vocab = get_vocab(tokenizer, model_name, vocab=vocab)
    else:
        tokenizer_vocab = vocab

    if save_mode == 'array':
        all_embeddings = []
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            if remove_end_token:
                token_ = token.replace('</w>','')
            else:
                token_ = token
            emb_ = get_condition(token_, tokenizer, text_encoder)
            all_embeddings.append(emb_)
        return torch.cat(all_embeddings, dim=0) # shape (49408, 77, 768)
    elif save_mode == 'dict':
        all_embeddings = {}
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            if remove_end_token:
                token_ = token.replace('</w>','')
            else:
                token_ = token
            emb_ = get_condition(token_, tokenizer, text_encoder)
            all_embeddings[token] = emb_
        return all_embeddings
    else:
        raise ValueError("save_mode should be either 'array' or 'dict'")

@torch.no_grad()
def save_embedding_matrix(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, model_name='SD-v1-4', save_mode='array', vocab='EN3K'):
    model_suffix = "" if model_name == 'SD-v1-4' else "_v2-1"
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            end = min(LEN_TOKENIZER_VOCAB, start + 5000)
            embedding_matrix = create_embedding_matrix(tokenizer, text_encoder, start=start, end=end, model_name=model_name, save_mode=save_mode)
            torch.save(embedding_matrix, f'models/embedding_matrix_{start}_{end}_{save_mode}{model_suffix}.pt')
    
    elif vocab == 'EN3K':
        embedding_matrix = create_embedding_matrix(tokenizer, text_encoder, start=0, end=LEN_EN_3K_VOCAB, model_name=model_name, save_mode=save_mode, vocab='EN3K')
        torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_EN3K{model_suffix}.pt')
    
    elif vocab == 'Imagenet':
        embedding_matrix = create_embedding_matrix(tokenizer, text_encoder, start=0, end=1000, model_name=model_name, save_mode=save_mode, vocab='Imagenet')
        torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_Imagenet{model_suffix}.pt')

    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")

def my_kmean(sorted_sim_dict, num_centers, compute_mode):
    if compute_mode == 'numpy':
        from sklearn.cluster import KMeans
        similarities = np.array([sorted_sim_dict[token].item() for token in sorted_sim_dict])
        similarities = similarities.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(similarities)
        cluster_centers = kmeans.cluster_centers_
    elif compute_mode == 'torch':
        from torch_kmeans import KMeans
        similarities = torch.stack([sorted_sim_dict[token] for token in sorted_sim_dict])
        similarities = torch.unsqueeze(similarities, dim=0)
        similarities = torch.unsqueeze(similarities, dim=2) # [1, N, 1]
        kmeans = KMeans(n_clusters=num_centers).fit(similarities)
        cluster_centers = kmeans.cluster_centers

    # find the closest token to each cluster center
    cluster_dict = {}
    for i, center in enumerate(cluster_centers):
        closest_token = None
        closest_similarity = -float('inf')
        for j, token in enumerate(sorted_sim_dict):
            similarity = sorted_sim_dict[token].item()
            if abs(similarity - center) < abs(closest_similarity - center):
                closest_similarity = similarity
                closest_token = token
        cluster_dict[closest_token] = (closest_token, closest_similarity, i)

    return cluster_dict

@torch.no_grad()
def learn_k_means_from_input_embedding(sim_dict: dict, num_centers=5, compute_mode='numpy'):
    """
    Given a model, a set of tokens, and a concept, learn k-means clustering on the search_closest_tokens's output
    """
    if num_centers <= 0:
        return list(sim_dict.keys())
    if len(list(sim_dict.keys())) <= num_centers:
        return list(sim_dict.keys())

    return list(my_kmean(sim_dict, num_centers, compute_mode).keys())

def get_similarities(sim: str, concept_embedding: torch.Tensor, embedding_matrix: torch.Tensor) -> torch.Tensor:
    if sim == 'cosine':
        return F.cosine_similarity(concept_embedding, embedding_matrix, dim=-1)
    elif sim == 'l2':
        return - F.pairwise_distance(concept_embedding, embedding_matrix, p=2)

def detect_special_tokens(text: str) -> bool:
    text = text.lower()
    for i in range(len(text)):
        if text[i] not in 'abcdefghijklmnopqrstuvwxyz</> ': # include space
            return True
    return False

@torch.no_grad()
def search_closest_tokens(
    concept: str, 
    tokenizer: CLIPTokenizer, 
    text_encoder: CLIPTextModel, 
    k: int=5, 
    reshape: bool=True, 
    sim: str='cosine', 
    model_name: str='SD-v1-4', 
    ignore_special_tokens: bool=True, 
    vocab: str='EN3K'
):
    """
    Given a concept, i.e., "nudity", search for top-k closest tokens in the embedding space
    """

    tokenizer_vocab = get_vocab(tokenizer, model_name, vocab=vocab)
    # inverse the dictionary
    tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}

    concept_embedding = get_condition(concept, tokenizer, text_encoder)

    # Calculate the cosine similarity between the concept and all tokens
    # load the embedding matrix 
    all_similarities = []
    model_suffix = "" if model_name == 'SD-v1-4' else "_v2-1"
    if model_name not in ['SD-v1-4' or 'SD-v2-1']:
        raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
    
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            end = min(LEN_TOKENIZER_VOCAB, start+5000)
            embedding_matrix: torch.Tensor = torch.load(f'models/embedding_matrix_{start}_{end}_array{model_suffix}.pt')
            if reshape:
                concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
                embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
            similarities = get_similarities(sim, concept_embedding, embedding_matrix)
            all_similarities.append(similarities)
    elif vocab == 'EN3K':
        embedding_matrix = torch.load(f'models/embedding_matrix_array_EN3K{model_suffix}.pt')
        if reshape:
            concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        similarities = get_similarities(sim, concept_embedding, embedding_matrix)
        all_similarities.append(similarities)
    
    elif vocab == 'Imagenet':
        embedding_matrix = create_embedding_matrix(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            start=0,
            end=1000,
            model_name=model_name,
            save_mode='array',
            vocab='Imagenet'
        )
        if reshape:
            concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        similarities = get_similarities(sim, concept_embedding, embedding_matrix)
        all_similarities.append(similarities)
    
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K' or 'Imagenet' ")

    similarities = torch.cat(all_similarities, dim=0)
    # sorting the similarities
    sorted_similarities, indices = torch.sort(similarities, descending=True)
    
    sim_dict = {}
    for im, i in enumerate(indices):
        if i.item() not in tokenizer_vocab_indexing:
            continue
        if ignore_special_tokens:
            if detect_special_tokens(tokenizer_vocab_indexing[i.item()]):
                continue
        token = tokenizer_vocab_indexing[i.item()]
        sim_dict[token] = sorted_similarities[im]
    
    top_k_tokens = list(sim_dict.keys())[:k]
    return top_k_tokens, sim_dict

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    
    return extra_step_kwargs

def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,
    latents: torch.Tensor,
    conditional_emb: torch.Tensor,
    guidance_scale: float=1.0,
) -> torch.Tensor:
    do_guidance = abs(guidance_scale) > 1.0
    latent_model_input = (torch.cat([latents] * 2) if do_guidance else latents).to(unet.device)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    
    # predict the noise residual
    noise_pred = unet(latent_model_input, timestep.to(unet.device), encoder_hidden_states=conditional_emb.to(unet.device)).sample

    # perform guidance
    if do_guidance:
        noise_pred_out = torch.chunk(noise_pred, 2, dim=0)
        noise_pred_uncond, noise_pred_prompt = noise_pred_out[0], noise_pred_out[1]
        
        cond_guidance = noise_pred_prompt - noise_pred_uncond
        noise_pred = noise_pred_uncond + (guidance_scale * cond_guidance)
    
    return noise_pred


# Sample latents from unet and DDIM scheduler until the given timestep.
@torch.no_grad()
def sample_until(
    until: int,
    latents: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
    extra_step_kwargs: dict[str, Any] | None=None,
):
    """Sample latents until t for a given prompt."""
    timesteps = scheduler.timesteps

    # Denoising loop
    for i, t in enumerate(timesteps):
        noise_pred = predict_noise(unet, scheduler, t, latents, prompt_embeds, guidance_scale)

        latents = scheduler.step(model_output=noise_pred, timestep=t, sample=latents, **extra_step_kwargs).prev_sample

        if i == until - 1:
            break

    return latents

def apply_model(unet: UNet2DConditionModel, z: torch.Tensor, t_enc_ddpm: torch.Tensor, emb_0: torch.Tensor) -> torch.Tensor:
    device = unet.device
    return unet(z.to(device), t_enc_ddpm.to(device), encoder_hidden_states=emb_0.to(device)).sample

def id2embedding(tokenizer: CLIPTokenizer, all_embeddings: torch.Tensor, input_ids: torch.Tensor, device) -> torch.Tensor:
    input_one_hot = F.one_hot(input_ids.view(-1), num_classes = len(tokenizer.get_vocab())).float()
    input_one_hot = torch.unsqueeze(input_one_hot,0).to(device)
    return input_one_hot @ all_embeddings

def init_adv(k, tokenizer, all_embeddings, device, batch = 1, attack_init_embd = None):
    # Different attack types have different initializations (Attack types: add, insert)
    adv_embedding = torch.nn.Parameter(torch.randn([batch, k, 768])).to(device)
    
    if attack_init_embd is not None:
        # Use the provided initial adversarial embedding
        adv_embedding.data = attack_init_embd[:,1:1+k].data
    else:
        # Random sample k words from the vocabulary as the initial adversarial words
        tmp_ids = torch.randint(0,len(tokenizer),(batch, k)).to(device)
        tmp_embeddings = id2embedding(tokenizer, all_embeddings, tmp_ids, device)
        tmp_embeddings = tmp_embeddings.reshape(batch, k, 768)
        adv_embedding.data = tmp_embeddings.data
    return adv_embedding.detach().requires_grad_(True)
    
def soft_prompt_attack(
    word: str,
    unet: UNet2DConditionModel,
    unet_orig: UNet2DConditionModel,
    tokenizer: CLIPTokenizer,
    text_encoder: CustomCLIPTextModel,
    scheduler: DDIMScheduler,
    emb_0: torch.Tensor,
    emb_p: torch.Tensor,
    devices: list[torch.device],
    criteria,
    all_embeddings: torch.Tensor,
    args: Arguments,
    attack_init_embd: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    k = args.adv_prompt_num
    orig_prompt_len = len(word.split())
    if args.adv_attack_type == 'add':
        k = orig_prompt_len
        
    quick_sample_till_t = lambda x, s, code, t: sample_until(
        until=t,
        latents=code,
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=x,
        guidance_scale=s,
    )
    
    # Word Tokenization
    text_input = tokenize(word, tokenizer)
    sot_id, mid_id, replace_id, eot_id = split_id(text_input.input_ids.to(devices[0]), k, orig_prompt_len)
    
    # Word embedding for the prompt
    text_embeddings = id2embedding(tokenizer, all_embeddings, text_input.input_ids.to(devices[0]), devices[0])
    sot_embd, mid_embd, _, eot_embd = split_embd(text_embeddings, k, orig_prompt_len)
    
    if args.adv_attack_init == 'latest':
        adv_embedding = init_adv(k, tokenizer, all_embeddings, devices[0], 1, attack_init_embd)
    elif args.adv_attack_init == 'random':
        adv_embedding = init_adv(k, tokenizer, all_embeddings, devices[0], 1)
    
    adv_embedding.requires_grad = True
    attack_opt = optim.Adam([adv_embedding], lr=args.adv_attack_lr)
    
    if args.adv_attack_embd_type == 'condition_embd':
        input_adv_condition_embedding = construct_embd(k, adv_embedding, args.adv_attack_type, sot_embd, mid_embd, eot_embd)
        adv_input_ids = construct_id(k, replace_id, args.adv_attack_type, sot_id, eot_id, mid_id)
    
    print(f'[{args.adv_attack_type}] Starting {args.adv_attack_method} attack on "{word}"')
    for i in range(args.adv_attack_step):
        # ===== Randomly sample a time step from 0 to 1000 =====
        t_enc = torch.randint(args.ddim_steps, (1,), device=devices[0]) # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/args.ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/args.ddim_steps)*1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        start_code = torch.randn((1, 4, 64, 64)).to(devices[0]) # random inital noise            
    
        with torch.no_grad():
            # generate an image with the concept from ESD model
            z = quick_sample_till_t(
                torch.cat([emb_0, emb_p], dim=0) if args.start_guidance > 1 else emb_p,
                args.start_guidance, start_code, int(t_enc)) # emb_p seems to work better instead of emb_0
            e_p = apply_model(unet_orig, z, t_enc_ddpm, emb_p)
        
        # Construct input_ids and input_embeds for the ESD model
        if args.adv_attack_embd_type == 'word_embd':
            input_adv_word_embedding = construct_embd(k, adv_embedding, args.adv_attack_type, sot_embd, mid_embd, eot_embd)
            adv_input_ids = construct_id(k, replace_id, args.adv_attack_type, sot_id, eot_id, mid_id)
            input_adv_condition_embedding = text_encoder(input_ids = adv_input_ids.to(devices[0]), inputs_embeds=input_adv_word_embedding)[0]
        
        # get conditional score from ESD model with adversarial condition embedding
        e_n = apply_model(unet, z, t_enc_ddpm, input_adv_condition_embedding)
        
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss: torch.Tensor = criteria(e_n.to(devices[0]), e_p.to(devices[0]))
        loss.backward(retain_graph=True)
        
        if args.adv_attack_method == 'pgd':
            attack_opt.step()
        elif args.adv_attack_method == 'fast_at':
            adv_embedding.grad.sign_()
            attack_opt.step()
        else:
            raise ValueError('attack_method must be either pgd or fast_at')
        
        print(f'Attack_Loss: {loss.item()}')
    
    if args.adv_attack_embd_type == 'condition_embd':
        return input_adv_condition_embedding, adv_input_ids 
    elif args.adv_attack_embd_type == 'word_embd':
        return input_adv_word_embedding, adv_input_ids 
    else:
        raise ValueError('attack_embd_type must be either condition_embd or word_embd')
        
def split_embd(input_embed, k, orig_prompt_len):
    sot_embd, mid_embd, replace_embd, eot_embd = torch.split(input_embed, [1, orig_prompt_len, k, 76-orig_prompt_len-k ], dim=1)
    return sot_embd, mid_embd, replace_embd, eot_embd
    
def split_id(input_ids, k, orig_prompt_len):
    sot_id, mid_id, replace_id, eot_id = torch.split(input_ids, [1, orig_prompt_len, k, 76-orig_prompt_len-k], dim=1)
    return sot_id, mid_id, replace_id, eot_id

def construct_embd(k, adv_embedding: torch.Tensor, insertion_location, sot_embd: torch.Tensor, mid_embd: torch.Tensor, eot_embd: torch.Tensor):
    if insertion_location == 'prefix_k':     # Prepend k words before the original prompt
        embedding = torch.cat([sot_embd,adv_embedding, mid_embd,eot_embd], dim=1)
    elif insertion_location == 'replace_k':  # Replace k words in the original prompt
        replace_embd = eot_embd[:,0,:].repeat(1,mid_embd.shape[1],1)
        embedding = torch.cat([sot_embd,adv_embedding, replace_embd,eot_embd], dim=1)
    elif insertion_location == 'add':      # Add perturbation to the original prompt
        replace_embd = eot_embd[:,0,:].repeat(1,k,1)
        embedding = torch.cat([sot_embd ,adv_embedding + mid_embd, replace_embd,eot_embd], dim=1)
    elif insertion_location == 'suffix_k':   # Append k words after the original prompt
        embedding = torch.cat([sot_embd, mid_embd, adv_embedding, eot_embd], dim=1)
    elif insertion_location == 'mid_k':      # Insert k words in the middle of the original prompt
        embedding = [sot_embd,]
        total_num = mid_embd.size(1)
        embedding.append(mid_embd[:,:total_num//2,:])
        embedding.append(adv_embedding)
        embedding.append(mid_embd[:,total_num//2:,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding, dim=1)
    elif insertion_location == 'insert_k':   # seperate k words into the original prompt with equal intervals
        embedding = [sot_embd,]
        total_num = mid_embd.size(1)
        internals = total_num // (k+1)
        for i in range(k):
            embedding.append(mid_embd[:,internals*i:internals*(i+1),:])
            embedding.append(adv_embedding[:,i,:].unsqueeze(1))
        embedding.append(mid_embd[:,internals*(i+1):,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding, dim=1)
    elif insertion_location == 'per_k_words':
        embedding = [sot_embd,]
        for i in range(adv_embedding.size(1) - 1):
            embedding.append(adv_embedding[:,i,:].unsqueeze(1))
            embedding.append(mid_embd[:,3*i:3*(i+1),:])
        embedding.append(adv_embedding[:,-1,:].unsqueeze(1))
        embedding.append(mid_embd[:,3*(i+1):,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding, dim=1)
    return embedding

def construct_id(k, adv_id: torch.Tensor, insertion_location, sot_id, eot_id: torch.Tensor, mid_id: torch.Tensor):
    if insertion_location == 'prefix_k':
        input_ids = torch.cat([sot_id,adv_id,mid_id,eot_id],dim=1)
        
    elif insertion_location == 'replace_k':
        replace_id = eot_id[:,0].repeat(1,mid_id.shape[1])
        input_ids = torch.cat([sot_id,adv_id,replace_id,eot_id],dim=1)
    
    elif insertion_location == 'add':
        replace_id = eot_id[:,0].repeat(1,k)
        input_ids = torch.cat([sot_id,mid_id,replace_id,eot_id],dim=1)
    
    elif insertion_location == 'suffix_k':
        input_ids = torch.cat([sot_id,mid_id,adv_id,eot_id],dim=1)
        
    elif insertion_location == 'mid_k':
        input_ids = [sot_id,]
        total_num = mid_id.size(1)
        input_ids.append(mid_id[:,:total_num//2])
        input_ids.append(adv_id)
        input_ids.append(mid_id[:,total_num//2:])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
        
    elif insertion_location == 'insert_k':
        input_ids = [sot_id,]
        total_num = mid_id.size(1)
        internals = total_num // (k+1)
        for i in range(k):
            input_ids.append(mid_id[:,internals*i:internals*(i+1)])
            input_ids.append(adv_id[:,i].unsqueeze(1))
        input_ids.append(mid_id[:,internals*(i+1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
        
    elif insertion_location == 'per_k_words':
        input_ids = [sot_id,]
        for i in range(adv_id.size(1) - 1):
            input_ids.append(adv_id[:,i].unsqueeze(1))
            input_ids.append(mid_id[:,3*i:3*(i+1)])
        input_ids.append(adv_id[:,-1].unsqueeze(1))
        input_ids.append(mid_id[:,3*(i+1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
    return input_ids

def get_train_loss_retain(
    args: Arguments,
    unet: UNet2DConditionModel,
    unet_orig: UNet2DConditionModel,
    text_encoder: CustomCLIPTextModel,
    scheduler: DDIMScheduler,
    emb_0: torch.Tensor,
    emb_p: torch.Tensor,
    retain_emb_p: torch.Tensor,
    emb_n: torch.Tensor | None,
    retain_emb_n: torch.Tensor,
    devices: list[torch.device],
    criteria,
    adv_input_ids,
    adv_embd: torch.Tensor | None
) -> torch.Tensor:
    """
        emb_0: unconditional embedding
        emb_p: conditional embedding (for ground truth concept)
        emb_n: conditional embedding (for modified concept)
        
        adv_input_ids: input_ids for adversarial word embedding
        adv_emb_n: adversarial conditional embedding
        adv_word_emb_n: adversarial word embedding
    """

    quick_sample_till_t = lambda x, s, code, batch, t: sample_until(
        until=t,
        latents=code,
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=x,
        guidance_scale=s,
    )
    
    t_enc = torch.randint(args.ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc) / args.ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
    if args.adv_retain_train == 'reg':
        retain_start_code = torch.randn((args.adv_retain_batch, 4, 64, 64)).to(devices[0])
    
    with torch.no_grad():
        # generate an image with the concept from ESD model
        z = quick_sample_till_t(
            torch.cat([emb_0, emb_p], dim=0) if args.start_guidance > 1 else emb_p, 
            args.start_guidance, start_code, 1, int(t_enc)) # emb_p seems to work better instead of emb_0
        # get conditional and unconditional scores from frozen model at time step t and image z
        e_0 = apply_model(unet_orig, z, t_enc_ddpm, emb_0)
        e_p = apply_model(unet_orig, z, t_enc_ddpm, emb_p)
        
        if args.adv_retain_train == 'reg':
            retain_z = quick_sample_till_t(
                torch.cat([emb_0, retain_emb_p], dim=0) if args.start_guidance > 1 else retain_emb_p,
                args.start_guidance, retain_start_code, args.adv_retain_batch, int(t_enc)) # emb_p seems to work better instead of emb_0
            retain_e_p = apply_model(unet_orig, retain_z, t_enc_ddpm, retain_emb_p)
    
    if adv_embd is None:
        e_n = apply_model(unet, z, t_enc_ddpm, emb_n)
    else:
        if args.adv_attack_embd_type == 'condition_embd':
            # Train with adversarial conditional embedding
            e_n = apply_model(unet, z, t_enc_ddpm, adv_embd)
        elif args.adv_attack_embd_type == 'word_embd':
            # Train with adversarial word embedding
            adv_emb_n = text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=adv_embd.to(devices[0]))[0]
            e_n = apply_model(unet, z, t_enc_ddpm, adv_emb_n)
        else:
            raise ValueError('attack_embd_type must be either condition_embd or word_embd')
    
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
    unlearn_loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (args.negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) 
    if args.adv_retain_train == 'reg':
        retain_e_n = apply_model(unet, retain_z, t_enc_ddpm, retain_emb_n)
        retain_e_p.requires_grad = False
        retain_loss = criteria(retain_e_n.to(devices[0]), retain_e_p.to(devices[0]))
        return unlearn_loss + args.adv_retain_loss_w * retain_loss

    return unlearn_loss


"""
utils for MACE

"""

def importance_sampling_fn(t, temperature=0.05):
    """Importance Sampling Function f(t)"""
    return 1 / (1 + np.exp(-temperature * (t - 200))) - 1 / (1 + np.exp(-temperature * (t - 400)))

class AttnController:
    def __init__(self) -> None:
        self.attn_probs = []
        self.logs = []
        
    def __call__(self, attn_prob, m_name, preserve_prior, latent_num) -> Any:
        bs, _ = self.concept_positions.shape
        
        if preserve_prior:
            attn_prob = attn_prob[:attn_prob.shape[0] // latent_num]
            
        if self.use_gsam_mask:
            d = int(attn_prob.shape[1] ** 0.5)
            resized_mask = F.interpolate(self.mask, size=(d, d), mode='nearest')
            
            resized_mask = (resized_mask > 0.5).view(-1)
            attn_prob = attn_prob[:, resized_mask, :]
            target_attns = attn_prob[:, :, self.concept_positions[0]]
        else:
            head_num = attn_prob.shape[0] // bs
            target_attns = attn_prob.masked_select(self.concept_positions[:,None,:].repeat(head_num, 1, 1)).reshape(-1, self.concept_positions[0].sum())
        
        self.attn_probs.append(target_attns)
        self.logs.append(m_name)
        
    def set_concept_positions(self, concept_positions, mask=None, use_gsam_mask=False):
        self.concept_positions = concept_positions
        self.mask = mask
        self.use_gsam_mask = use_gsam_mask
        
    def loss(self):
        return sum(torch.norm(item) for item in self.attn_probs)
        
    def zero_attn_probs(self):
        self.attn_probs = []
        self.logs = []
        self.concept_positions = None

class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, attn_controller=None, module_name=None) -> None:
        self.attn_controller = attn_controller
        self.module_name = module_name

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        if key.shape[1] == 77 and self.attn_controller is not None:
            self.attn_controller(attention_probs, self.module_name, preserve_prior=True, latent_num=hidden_states.shape[0])
            
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class LoRAAttnProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, attn_controller=None, module_name=None, 
                 network_alpha=None, **kwargs):
        super().__init__()

        self.attn_controller = attn_controller
        self.module_name = module_name
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)

    def __call__(self, attn: Attention, hidden_states, *args, **kwargs):
        
        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

        attn._modules.pop("processor")
        # attn.set_processor(AttnProcessor(self.attn_controller, self.module_name))
        attn.processor = AttnProcessor(self.attn_controller, self.module_name)

        return attn.processor(attn, hidden_states, *args, **kwargs)

def get_ca_layers(unet: UNet2DConditionModel, with_to_k=True):

    sub_nets = unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ## get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [deepcopy(l.to_k) for l in ca_layers]
    
    return projection_matrices, ca_layers, og_matrices

@torch.no_grad()
def prepare_k_v(
    text_encoder: CLIPTextModel, 
    projection_matrices, 
    ca_layers,
    og_matrices,
    test_set,
    tokenizer: CLIPTokenizer,
    with_to_k=True,
    all_words=False,
    prepare_k_v_for_lora=False
):
    
    all_contexts, all_valuess = [], []
    
    for curr_item in test_set:
        gc.collect()
        torch.cuda.empty_cache()
        
        #### restart LDM parameters
        num_ca_clip_layers = len(ca_layers)
        for idx_, l in enumerate(ca_layers):
            l.to_v = deepcopy(og_matrices[idx_])
            projection_matrices[idx_] = l.to_v
            if with_to_k:
                l.to_k = deepcopy(og_matrices[num_ca_clip_layers + idx_])
                projection_matrices[num_ca_clip_layers + idx_] = l.to_k
        
        old_embs, new_embs = [], []
        extended_old_indices, extended_new_indices = [], []
        
        #### indetify corresponding destinations for each token in old_emb
        # Bulk tokenization
        texts_old = [item[0] for item in curr_item["old"]]
        texts_new = [item[0] for item in curr_item["new"]]
        texts_combined = texts_old + texts_new

        tokenized_inputs = tokenize(texts_combined, tokenizer)
        
        # Text embeddings
        text_embeddings = text_encoder(tokenized_inputs.input_ids.to(text_encoder.device))[0]
        old_embs.extend(text_embeddings[:len(texts_old)])
        new_embs.extend(text_embeddings[len(texts_old):])

        # Find matching indices
        for old_text, new_text in zip(texts_old, texts_new):
            tokens_a = tokenizer(old_text).input_ids
            tokens_b = tokenizer(new_text).input_ids
            
            old_indices, new_indices = find_matching_indices(tokens_a, tokens_b)
            
            if old_indices[-1] >= new_indices[-1]:
                extended_old_indices.append(old_indices + list(range(old_indices[-1] + 1, 77)))
                extended_new_indices.append(new_indices + list(range(new_indices[-1] + 1, 77 - (old_indices[-1] - new_indices[-1]))))
            else:
                extended_new_indices.append(new_indices + list(range(new_indices[-1] + 1, 77)))
                extended_old_indices.append(old_indices + list(range(old_indices[-1] + 1, 77 - (new_indices[-1] - old_indices[-1]))))

        #### prepare batch: for each pair of setences, old context and new values
        contexts, valuess = [], []
        if not all_words:
            for idx, (old_emb, new_emb) in enumerate(zip(old_embs, new_embs)):
                context = old_emb[extended_old_indices[idx]].detach()
                values = []
                for layer in projection_matrices:
                    values.append(layer(new_emb[extended_new_indices[idx]]).detach())
                contexts.append(context)
                valuess.append(values)
        
            all_contexts.append(contexts)
            all_valuess.append(valuess)
        else:
            if prepare_k_v_for_lora:
                # prepare for lora, then no need to use new_emb
                for idx, old_emb in enumerate(old_embs):
                    context = old_emb.detach()
                    values = []
                    for layer in projection_matrices:
                        values.append(layer(old_emb).detach())
                    contexts.append(context)
                    valuess.append(values)
            else:
                # need to use new_emb
                for idx, (old_emb, new_emb) in enumerate(zip(old_embs, new_embs)):
                    context = old_emb.detach()
                    values = []
                    for layer in projection_matrices:
                        values.append(layer(new_emb).detach())
                    contexts.append(context)
                    valuess.append(values)
        
            all_contexts.append(contexts)
            all_valuess.append(valuess)
    
    return all_contexts, all_valuess
    
def find_matching_indices(old, new):
    # Find the starting common sequence
    start_common = 0
    for i, j in zip(old, new):
        if i == j:
            start_common += 1
        else:
            break

    # Find the ending common sequence
    end_common_old = len(old) - 1
    end_common_new = len(new) - 1
    while end_common_old >= start_common and end_common_new >= start_common:
        if old[end_common_old] == new[end_common_new]:
            end_common_old -= 1
            end_common_new -= 1
        else:
            break

    return list(range(start_common)) + list(range(end_common_old + 1, len(old))), \
           list(range(start_common)) + list(range(end_common_new + 1, len(new)))

@torch.no_grad()
def closed_form_refinement(projection_matrices, all_contexts=None, all_valuess=None, lamb=0.5, preserve_scale=1, cache_dict=None, cache_dict_path=None, cache_mode=False):
    
    if cache_dict_path is not None:
        cache_dict = torch.load(cache_dict_path, map_location=projection_matrices[0].weight.device)
        
    for layer_num in tqdm(range(len(projection_matrices))):
        gc.collect()
        torch.cuda.empty_cache()
        
        mat1 = lamb * projection_matrices[layer_num].weight
        mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device=projection_matrices[layer_num].weight.device)
        
        total_for_mat1 = torch.zeros_like(projection_matrices[layer_num].weight)
        total_for_mat2 = torch.zeros_like(mat2)

        if all_contexts is not None and all_valuess is not None:
            for contexts, valuess in zip(all_contexts, all_valuess):
                # Convert contexts and values to tensors
                contexts_tensor = torch.stack(contexts, dim=2)
                values_tensor = torch.stack([vals[layer_num] for vals in valuess], dim=2)
                
                # Aggregate sums for mat1, mat2 using matrix multiplication
                for_mat1 = torch.bmm(values_tensor, contexts_tensor.permute(0, 2, 1)).sum(dim=0)
                for_mat2 = torch.bmm(contexts_tensor, contexts_tensor.permute(0, 2, 1)).sum(dim=0)
                
                total_for_mat1 += for_mat1
                total_for_mat2 += for_mat2

            del for_mat1, for_mat2
            
        if cache_mode: 
            # cache the results
            if cache_dict[f'{layer_num}_for_mat1'] is None:
                cache_dict[f'{layer_num}_for_mat1'] = total_for_mat1
                cache_dict[f'{layer_num}_for_mat2'] = total_for_mat2
            else:
                cache_dict[f'{layer_num}_for_mat1'] += total_for_mat1
                cache_dict[f'{layer_num}_for_mat2'] += total_for_mat2
        else:
            # CFR calculation
            if cache_dict_path is not None or cache_dict is not None:
                total_for_mat1 += preserve_scale * cache_dict[f'{layer_num}_for_mat1']
                total_for_mat2 += preserve_scale * cache_dict[f'{layer_num}_for_mat2']
                
            total_for_mat1 += mat1
            total_for_mat2 += mat2
            
            projection_matrices[layer_num].weight.data = total_for_mat1 @ torch.inverse(total_for_mat2) 
            
        del total_for_mat1, total_for_mat2
