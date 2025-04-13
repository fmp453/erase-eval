# Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation (EAP)

import os
import random
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from sklearn.cluster import KMeans

from torch.autograd import Variable
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from utils import Arguments
from train_methods.consts import ddim_alphas, LEN_TOKENIZER_VOCAB, LEN_EN_3K_VOCAB


@torch.no_grad()
def sample_until(
    until: int,
    latents: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
    extra_step_kwargs: Optional[dict[str, Any]]=None,
):
    """Sample latents until t for a given prompt."""
    timesteps = scheduler.timesteps

    do_guidance = abs(guidance_scale) > 1.0
    device = unet.device

    # Denoising loop
    for i, t in enumerate(timesteps):
        t = t.to(device)
        latent_model_input = (torch.cat([latents] * 2) if do_guidance else latents).to(device)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds.to(device)).sample

        # perform guidance
        if do_guidance:
            noise_pred_out = torch.chunk(noise_pred, 2, dim=0)
            noise_pred_uncond, noise_pred_prompt = noise_pred_out[0], noise_pred_out[1]
            
            cond_guidance = noise_pred_prompt - noise_pred_uncond
            noise_pred = noise_pred_uncond + (guidance_scale * cond_guidance)

        latents = scheduler.step(model_output=noise_pred, timestep=t, sample=latents).prev_sample

        if i == until - 1:
            break

    return latents

@torch.no_grad()
def get_learned_conditioning(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, prompt: list[str]):
    input_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        truncation=True
    ).input_ids
    # text_encoder(input_ids=uncond_input["input_ids"].to(device), attention_mask=uncond_input["attention_mask"].to(device) if use_attention_mask else None)[0]
    device = text_encoder.device
    emb = text_encoder(input_ids.to(device))[0]
    return emb

def save_to_dict(var, name, dict):
    if var is not None:
        if isinstance(var, torch.Tensor):
            var = var.cpu().detach().numpy()
        if isinstance(var, list):
            var = [v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for v in var]
    else:
        return dict
    
    if name not in dict:
        dict[name] = []
    
    dict[name].append(var)
    return dict

def get_english_tokens():
    data_path = 'captions/english_3000.csv'
    df = pd.read_csv(data_path)
    vocab = {}
    for ir, row in df.iterrows():
        vocab[row['word']] = ir
    assert(len(vocab) == LEN_EN_3K_VOCAB)
    return vocab

def retrieve_embedding_token(model_name, query_token, vocab='EN3K'):
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            end = min(LEN_TOKENIZER_VOCAB, start+5000)
            if model_name == 'SD-v1-4':
                embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_dict.pt')
            elif model_name == 'SD-v2-1':
                embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_dict_v2-1.pt')
            else:
                raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
            if query_token in embedding_matrix:
                return embedding_matrix[query_token]
    elif vocab == 'EN3K':
        if model_name == 'SD-v1-4':
            embedding_matrix = torch.load(f'models/embedding_matrix_dict_EN3K.pt')
        elif model_name == 'SD-v2-1':
            embedding_matrix = torch.load(f'models/embedding_matrix_dict_EN3K_v2-1.pt')
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
        if query_token in embedding_matrix:
            return embedding_matrix[query_token]
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")

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
def create_embedding_matrix(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, start=0, end=LEN_TOKENIZER_VOCAB, model_name='SD-v1-4', save_mode='array', remove_end_token=False, vocab='EN3K'):

    tokenizer_vocab = get_vocab(tokenizer, model_name, vocab=vocab)

    if save_mode == 'array':
        all_embeddings = []
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            # print(token, tokenizer_vocab[token])
            if remove_end_token:
                token_ = token.replace('</w>','')
            else:
                token_ = token
            emb_ = get_learned_conditioning(tokenizer=tokenizer, text_encoder=text_encoder, prompt=[token_])
            all_embeddings.append(emb_)
        return torch.cat(all_embeddings, dim=0) # shape (49408, 77, 768)
    elif save_mode == 'dict':
        all_embeddings = {}
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            # print(token, tokenizer_vocab[token])
            if remove_end_token:
                token_ = token.replace('</w>','')
            else:
                token_ = token
            emb_ = get_learned_conditioning(tokenizer=tokenizer, text_encoder=text_encoder, prompt=[token_])
            all_embeddings[token] = emb_
        return all_embeddings
    else:
        raise ValueError("save_mode should be either 'array' or 'dict'")

@torch.no_grad()
def save_embedding_matrix(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, model_name='SD-v1-4', save_mode='array', vocab='EN3K'):
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            end = min(LEN_TOKENIZER_VOCAB, start+5000)
            embedding_matrix = create_embedding_matrix(tokenizer, text_encoder, start=start, end=end, model_name=model_name, save_mode=save_mode)
            if model_name == 'SD-v1-4':
                torch.save(embedding_matrix, f'models/embedding_matrix_{start}_{end}_{save_mode}.pt')
            elif model_name == 'SD-v2-1':
                torch.save(embedding_matrix, f'models/embedding_matrix_{start}_{end}_{save_mode}_v2-1.pt')
    elif vocab == 'EN3K':
        embedding_matrix = create_embedding_matrix(tokenizer, text_encoder, start=0, end=LEN_EN_3K_VOCAB, model_name=model_name, save_mode=save_mode, vocab='EN3K')
        if model_name == 'SD-v1-4':
            torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_EN3K.pt')
        elif model_name == 'SD-v2-1':
            torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_EN3K_v2-1.pt')
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")

@torch.no_grad()
def search_closest_tokens(concept, tokenizer, text_encoder, k=5, reshape=True, sim='cosine', model_name='SD-v1-4', ignore_special_tokens=True, vocab='EN3K'):
    """
    Given a concept, i.e., "nudity", search for top-k closest tokens in the embedding space
    """

    tokenizer_vocab = get_vocab(tokenizer, model_name, vocab=vocab)
    # inverse the dictionary
    tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}

    # Get the embedding of the concept
    concept_embedding = get_learned_conditioning(tokenizer, text_encoder, [concept]) # shape (1, 77, 768)

    # Calculate the cosine similarity between the concept and all tokens
    # load the embedding matrix 
    all_similarities = []
    
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            end = min(LEN_TOKENIZER_VOCAB, start+5000)
            if model_name == 'SD-v1-4':
                embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_array.pt')
            elif model_name == 'SD-v2-1':
                embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_array_v2-1.pt')
            else:
                raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
            
            if reshape == True:
                concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
                embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
            if sim == 'cosine':
                similarities = F.cosine_similarity(concept_embedding, embedding_matrix, dim=-1)
            elif sim == 'l2':
                similarities = - F.pairwise_distance(concept_embedding, embedding_matrix, p=2)
            all_similarities.append(similarities)
    elif vocab == 'EN3K':
        if model_name == 'SD-v1-4':
            embedding_matrix = torch.load(f'models/embedding_matrix_array_EN3K.pt')
        elif model_name == 'SD-v2-1':
            embedding_matrix = torch.load(f'models/embedding_matrix_array_EN3K_v2-1.pt')
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
        if reshape == True:
            concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        if sim == 'cosine':
            similarities = F.cosine_similarity(concept_embedding, embedding_matrix, dim=-1)
        elif sim == 'l2':
            similarities = - F.pairwise_distance(concept_embedding, embedding_matrix, p=2)
        all_similarities.append(similarities)
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")

    similarities = torch.cat(all_similarities, dim=0)
    # sorting the similarities
    sorted_similarities, indices = torch.sort(similarities, descending=True)
    
    sim_dict = {}
    for im, i in enumerate(indices):
        if ignore_special_tokens:
            if detect_special_tokens(tokenizer_vocab_indexing[i.item()]):
                continue
        token = tokenizer_vocab_indexing[i.item()]
        sim_dict[token] = sorted_similarities[im]
    
    top_k_tokens = list(sim_dict.keys())[:k]
    return top_k_tokens, sim_dict

def detect_special_tokens(text):
    text = text.lower()
    for i in range(len(text)):
        if text[i] not in 'abcdefghijklmnopqrstuvwxyz</>':
            return True
    return False

def apply_model(unet: UNet2DConditionModel, z: torch.Tensor, t_enc_ddpm: torch.Tensor, emb_0: torch.Tensor):
    # get conditional and unconditional scores from frozen model at time step t and image z
    device = unet.device
    z = z.to(device)
    t_enc_ddpm = t_enc_ddpm.to(device)
    emb_0 = emb_0.to(device)

    noise_pred = unet(z, t_enc_ddpm, encoder_hidden_states=emb_0).sample
    return noise_pred

def my_kmean(sorted_sim_dict, num_centers, compute_mode):
    similarities = np.array([sorted_sim_dict[token].item() for token in sorted_sim_dict])
    similarities = similarities.reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(similarities)
    cluster_centers = kmeans.cluster_centers_
    
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
def learn_k_means_from_input_embedding(sim_dict, num_centers=5, compute_mode='numpy'):
    """
    Given a model, a set of tokens, and a concept, learn k-means clustering on the search_closest_tokens's output
    """
    if num_centers <= 0:
        print(f"Number of centers should be greater than 0. Returning the tokens themselves.")
        return list(sim_dict.keys())
    if len(list(sim_dict.keys())) <= num_centers:
        print(f"Number of tokens is less than the number of centers. Returning the tokens themselves.")
        return list(sim_dict.keys())

    return list(my_kmean(sim_dict, num_centers, compute_mode).keys())

def make_ddim_sampling_parameters(alphacums, ddim_timesteps):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    return alphas

def train(args: Arguments):
    
    prompt = args.concepts

    # prompt
    preserved = ""

    if args.seperator is not None:
        erased_words = prompt.split(args.seperator)
        erased_words = [word.strip() for word in erased_words]
        preserved_words = preserved.split(args.seperator)
        preserved_words = [word.strip() for word in preserved_words]
    else:
        erased_words = [prompt]
        preserved_words = [preserved]
    
    print('to be erased:', erased_words)
    print('to be preserved:', preserved_words)
    preserved_words.append('')

    devices = args.device.split(",")
    if len(devices) > 1:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[1]}")]
    else:
        devices = [torch.device(f"cuda:{devices[0]}"), torch.device(f"cuda:{devices[0]}")]

    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    unet_orig: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")

    vae.eval()
    text_encoder.eval()
    text_encoder.to(devices[0])
    unet_orig.eval()
    unet_orig.to(devices[1])
    unet.to(devices[0])

    # choose parameters to train based on train_method
    parameters = []
    for name, param in unet.named_parameters():
        # train all layers except x-attns and time_embed layers
        if args.eap_method == 'noxattn':
            if not (name.startswith('out.') or 'attn2' in name or 'time_embed' in name):
                parameters.append(param)
        # train only self attention layers
        if args.eap_method == 'selfattn':
            if 'attn1' in name:
                parameters.append(param)
        # train only x attention layers
        if args.eap_method == 'xattn':
            if 'attn2' in name:
                parameters.append(param)
        # train only qkv layers in x attention layers
        if args.eap_method == 'xattn_matching':
            if 'attn2' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name):
                parameters.append(param)
                # return_nodes[name] = name
        # train all layers
        if args.eap_method == 'full':
            parameters.append(param)
        # train all layers except time embed layers
        if args.eap_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                parameters.append(param)
        if args.eap_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    parameters.append(param)
        if args.eap_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    parameters.append(param)
    
    unet.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda cond, s, code, t: sample_until(
        until=t,
        latents=code,
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=cond,
        guidance_scale=s
    )
    
    losses = []
    opt = optim.Adam(parameters, lr=args.eap_lr)
    criteria = nn.MSELoss()
    history_dict = {}

    def create_prompt(word, retrieve=True):
        if retrieve:
            return retrieve_embedding_token(model_name='SD-v1-4', query_token=word, vocab=args.vocab)
        else:
            prompt = f'{word}'
            emb = get_learned_conditioning(tokenizer, text_encoder, [prompt])
            init = emb
            return init

    """
    Algorithm description: 
    Given a pre-defined set of concepts to preserve, use the adversarial prompt learning to learn only one concept from the set that maximizes a loss function. 
    The index of the concept is represented by a index one-hot vector. 
    Step by step: 
    - Step 0: get text embeddings of these concepts from the set. Concat the embeddings to form a matrix. 
    - Step 1: Init one-hot vector with the first concept (or random concept from the set)
    
    """

    # create embedding matrix for all tokens in the dictionary
    
    if not os.path.exists('models/embedding_matrix_dict_EN3K.pt'):
        save_embedding_matrix(tokenizer, text_encoder, model_name='SD-v1-4', save_mode='dict', vocab='EN3K')

    if not os.path.exists('models/embedding_matrix_array_EN3K.pt'):
        save_embedding_matrix(tokenizer, text_encoder, model_name='SD-v1-4', save_mode='array', vocab='EN3K')

    
    # shorten the list of tokens_embedding to 5000, using the similarities between tokens
    tokens_embedding = []
    all_sim_dict = dict()
    for word in erased_words:
        top_k_tokens, sorted_sim_dict = search_closest_tokens(word, tokenizer, text_encoder, k=args.gumbel_k_closest, sim='l2', model_name='SD-v1-4', ignore_special_tokens=args.ignore_special_tokens, vocab=args.vocab)
        tokens_embedding.extend(top_k_tokens)
        all_sim_dict[word] = {key:sorted_sim_dict[key] for key in top_k_tokens}

    if args.gumbel_num_centers > 0:
        assert args.gumbel_num_centers % len(erased_words) == 0, 'Number of centers should be divisible by number of erased words'
    preserved_dict = dict()

    for word in erased_words:
        temp = learn_k_means_from_input_embedding(sim_dict=all_sim_dict[word], num_centers=args.gumbel_num_centers)
        preserved_dict[word] = temp

    history_dict = save_to_dict(preserved_dict, f'preserved_set_0', history_dict)

    # create a matrix of embeddings for the preserved set
    print('Creating preserved matrix')
    one_hot_dict = dict()
    preserved_matrix_dict = dict()
    for erase_word in erased_words:
        preserved_set = preserved_dict[erase_word]
        for i, word in enumerate(preserved_set):
            if i == 0:
                preserved_matrix = create_prompt(word)
            else:
                preserved_matrix = torch.cat((preserved_matrix, create_prompt(word)), dim=0)
        preserved_matrix = preserved_matrix.flatten(start_dim=1) # [n, 77*768]
        one_hot = torch.zeros((1, preserved_matrix.shape[0]), device=devices[0], dtype=preserved_matrix.dtype) # [1, n]
        one_hot = one_hot + 1 / preserved_matrix.shape[0]
        one_hot = Variable(one_hot, requires_grad=True)
        one_hot_dict[erase_word] = one_hot
        preserved_matrix_dict[erase_word] = preserved_matrix
    
    history_dict = save_to_dict(one_hot_dict, f'one_hot_dict_0', history_dict)

    # optimizer for all one-hot vectors
    opt_one_hot = optim.Adam([one_hot for one_hot in one_hot_dict.values()], lr=args.gumbel_lr)

    def gumbel_softmax(logits, temperature=args.gumbel_temp, hard=args.gumbel_hard, eps=1e-10):
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = logits + gumbel
        y = F.softmax(y / temperature, dim=-1)
        if hard != 0:
            y_hard = torch.zeros_like(logits)
            y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        return y

    pbar = tqdm(range(args.pgd_num_steps * args.eap_iterations))
    scheduler.set_timesteps(args.ddim_steps, devices[1])
    for i in pbar:
        word = random.sample(erased_words,1)[0]

        opt.zero_grad()
        opt_one_hot.zero_grad()

        prompt_0 = ''
        prompt_n = f'{word}'

        # get text embeddings for unconditional and conditional prompts
        emb_0 = get_learned_conditioning(tokenizer, text_encoder, [prompt_0])
        emb_n = get_learned_conditioning(tokenizer, text_encoder, [prompt_n])

        # get the emb_r 
        emb_r = torch.reshape(torch.matmul(gumbel_softmax(one_hot_dict[word]), preserved_matrix_dict[word]).unsqueeze(0), (1, 77, 768))
        assert emb_r.shape == emb_n.shape

        t_enc = torch.randint(args.ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc) / args.ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate an image with the concept
            z = quick_sample_till_t(torch.cat([emb_0, emb_n], dim=0), args.start_guidance, start_code, int(t_enc))
            z_r = quick_sample_till_t(torch.cat([emb_0, emb_r], dim=0), args.start_guidance, start_code, int(t_enc))

            # get conditional and unconditional scores from frozen model at time step t and image z
            e_0_org = apply_model(unet_orig, z, t_enc_ddpm, emb_0)
            e_n_org = apply_model(unet_orig, z, t_enc_ddpm, emb_n)
            e_r_org = apply_model(unet_orig, z_r, t_enc_ddpm, emb_r)

        # get conditional score
        e_n_wo_prompt = apply_model(unet, z, t_enc_ddpm, emb_n)
        e_r_wo_prompt = apply_model(unet, z_r, t_enc_ddpm, emb_r)

        e_0_org.requires_grad = False
        e_n_org.requires_grad = False
        e_r_org.requires_grad = False

        # using DDIM inversion to project the x_t to x_0
        # check that the alphas is in descending order
        alpha_bar_t = ddim_alphas[int(t_enc)]
        z_n_wo_prompt_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_n_wo_prompt) / torch.sqrt(alpha_bar_t)
        z_r_wo_prompt_pred = (z_r - torch.sqrt(1 - alpha_bar_t) * e_r_wo_prompt) / torch.sqrt(alpha_bar_t)

        z_n_org_pred = (z.to(e_n_org.device) - torch.sqrt(1 - alpha_bar_t) * e_n_org) / torch.sqrt(alpha_bar_t)
        z_0_org_pred = (z.to(e_0_org.device) - torch.sqrt(1 - alpha_bar_t) * e_0_org) / torch.sqrt(alpha_bar_t)
        z_r_org_pred = (z_r.to(e_r_org.device) - torch.sqrt(1 - alpha_bar_t) * e_r_org) / torch.sqrt(alpha_bar_t)

        # First stage, optimizing additional prompt

        if i % args.pgd_num_steps == 0:
            # for erased concepts, output aligns with target concept with or without prompt
            loss= criteria(z_n_wo_prompt_pred.to(devices[0]), z_0_org_pred.to(devices[0]) - (args.negative_guidance * (z_n_org_pred.to(devices[0]) - z_0_org_pred.to(devices[0]))))
            loss += criteria(z_r_wo_prompt_pred.to(devices[0]), z_r_org_pred.to(devices[0])) # for preserved concepts, output are the same without prompt

            # update weights to erase the concept
            loss.backward()
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            history_dict = save_to_dict(loss.item(), 'loss', history_dict)
            opt.step()
        else:
            # update the one_hot vector
            opt.zero_grad()
            opt_one_hot.zero_grad()
            loss = - criteria(z_r_wo_prompt_pred.to(devices[0]), z_r_org_pred.to(devices[0])) # maximize the preserved loss
            loss.backward()
            preserved_set = preserved_dict[word]
            opt_one_hot.step()
            history_dict = save_to_dict([one_hot_dict[word].cpu().detach().numpy(), i, preserved_set[torch.argmax(one_hot_dict[word], dim=1)], word], 'one_hot', history_dict)

    unet.eval()
    unet.save_pretrained(args.save_dir)

def main(args: Arguments):
    train(args)

