import os
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import trange
from PIL import Image
from torch.autograd import Variable
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL

from train_methods.train_utils import sample_until, apply_model
from utils_age import ConceptDict
from utils_age import learn_k_means_from_input_embedding, save_embedding_matrix, search_closest_tokens, get_condition, ddim_alphas
from utils import Arguments

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)
    plt.close()


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



def train_age(
        args: Arguments
        # prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, seperator=None, image_size=512, ddim_steps=50, args=None
    ):
    '''
    train_method : str
        The parameters to train for erasure (noxattn, xattan).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.
    '''

    # PROMPT CLEANING
    prompt = args.concepts
    preserved = ""

    if args.seperator is not None:
        erased_words: list[str] = prompt.split(args.seperator)
        erased_words = [word.strip() for word in erased_words]
        preserved_words = preserved.split(args.seperator)
        preserved_words = [word.strip() for word in preserved_words]
    else:
        erased_words = [prompt]
        preserved_words = [preserved]
    
    print('to be erased:', erased_words)
    print('to be preserved:', preserved_words)
    preserved_words.append('')

    device = torch.device(f'cuda:{args.device.split(",")[0]}')
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    orig_unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")    

    unet.eval()
    vae.eval()
    text_encoder.eval()
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    orig_unet.to(device)

    parameters = []
    for name, param in unet.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                parameters.append(param)
        # train only qkv layers in x attention layers
        if train_method == 'xattn_matching':
            if 'attn2' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name):
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    parameters.append(param)

    unet.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda x, s, code, t: sample_until(
        until=t,
        latents=code,
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=x,
        guidance_scale=s,
        # extra_step_kwargs: Optional[dict[str, Any]]=None,
    )

    losses = []
    opt = optim.Adam(parameters, lr=lr)
    criteria = nn.MSELoss()
    history_dict = {}

    pbar = trange(args.pgd_num_steps*iterations)

    def create_prompt(word: str) -> torch.Tensor:
        return get_condition(word, tokenizer, text_encoder)
        

    # create a matrix of embeddings for the entire vocabulary
    if not Path('models/embedding_matrix_dict_EN3K.pt').exists():
        save_embedding_matrix(tokenizer=tokenizer, text_encoder=text_encoder, model_name='SD-v1-4', save_mode='dict', vocab='EN3K')

    if not Path('models/embedding_matrix_array_EN3K.pt').exists():
        save_embedding_matrix(tokenizer=tokenizer, text_encoder=text_encoder, model_name='SD-v1-4', save_mode='array', vocab='EN3K')
    
    if not Path('models/embedding_matrix_array_Imagenet.pt').exists():
        save_embedding_matrix(tokenizer=tokenizer, text_encoder=text_encoder, model_name='SD-v1-4', save_mode='array', vocab='Imagenet')
    
    if not Path('models/embedding_matrix_dict_Imagenet.pt').exists():
        save_embedding_matrix(tokenizer=tokenizer, text_encoder=text_encoder, model_name='SD-v1-4', save_mode='dict', vocab='Imagenet')


    # Search the closest tokens in the vocabulary for each erased word, using the similarity matrix
    # if vocab in ['EN3K', 'Imagenet', 'CLIP'], then use the pre-defined vocabulary
    # if vocab in concept_dict.all_concepts, then use the custom concepts, i.e., 'nudity', 'artistic', 'human_body'
    # if vocab is 'keyword', then use the keywords in the erased words, defined in utils_concept.py

    concept_dict = ConceptDict()
    concept_dict.load_all_concepts()

    print('ignore_special_tokens:', args.ignore_special_tokens)
    
    all_sim_dict = dict()
    for word in erased_words:
        if args.vocab in ['EN3K', 'Imagenet', 'CLIP']:
            vocab = args.vocab
        elif args.vocab in concept_dict.all_concepts:
            # i.e., nudity, artistic, human_body
            vocab = concept_dict.get_concepts_as_dict(args.vocab)
        elif args.vocab == 'keyword':
            # i.e., 'Cassette Player', 'Chain Saw', 'Church', 
            vocab = concept_dict.get_concepts_as_dict(word)
        else:
            raise ValueError(f'Word {word} not found in concept dictionary, it should be either in EN3K, Imagenet, CLIP, or in the concept dictionary')
        
        top_k_tokens, sorted_sim_dict = search_closest_tokens(word, tokenizer, text_encoder, k=args.gumbel_k_closest, sim='l2', model_name='SD-v1-4', ignore_special_tokens=args.ignore_special_tokens, vocab=vocab)
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
    weight_pi_dict = dict()
    preserved_matrix_dict = dict()
    for erase_word in erased_words:
        preserved_set = preserved_dict[erase_word]
        for i, word in enumerate(preserved_set):
            if i == 0:
                preserved_matrix = create_prompt(word)
            else:
                preserved_matrix = torch.cat((preserved_matrix, create_prompt(word)), dim=0)
            print(i, word, preserved_matrix.shape)    
        preserved_matrix = preserved_matrix.flatten(start_dim=1) # [n, 77*768]
        weight_pi = torch.zeros((1, preserved_matrix.shape[0]), device=devices[0], dtype=preserved_matrix.dtype) # [1, n]
        weight_pi = weight_pi + 1 / preserved_matrix.shape[0]
        weight_pi = Variable(weight_pi, requires_grad=True)
        weight_pi_dict[erase_word] = weight_pi
        preserved_matrix_dict[erase_word] = preserved_matrix
    
    print('weight_pi_dict:', weight_pi_dict)
    history_dict = save_to_dict(weight_pi_dict, f'one_hot_dict_0', history_dict)

    # optimizer for all pi vectors
    opt_weight_pi = optim.Adam([weight_pi for weight_pi in weight_pi_dict.values()], lr=args.gumbel_lr)

    """
    Gumbel-Softmax function
        if `hard` is 1, then it is one-hot, if `hard` is 0, then it is a new soft version, which takes the top-k highest values and normalize them to 1
    """
    def gumbel_softmax(logits, temperature=args.gumbel_temp, hard=args.gumbel_hard, eps=1e-10, k=args.gumbel_topk):
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temperature, dim=-1)
        if hard == 1:
            y_hard = torch.zeros_like(logits)
            y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        elif hard == 0:
            top_k_values, _ = torch.topk(y, k, dim=-1)
            top_k_mask = y >= top_k_values[..., -1].unsqueeze(-1)
            y = y * top_k_mask.float()
            y = y / y.sum(dim=-1, keepdim=True)
        return y

    for i in pbar:
        word = random.sample(erased_words,1)[0]

        opt.zero_grad()
        unet.zero_grad()
        orig_unet.zero_grad()
        opt_weight_pi.zero_grad()

        c_e = f'{word}'

        emb_c_e = get_condition(c_e, tokenizer, text_encoder)

        emb_c_t = torch.reshape(torch.matmul(gumbel_softmax(weight_pi_dict[word]), preserved_matrix_dict[word]).unsqueeze(0), (1, 77, 768))
        assert emb_c_t.shape == emb_c_e.shape

        # clone the emb_c_t for the time step
        emb_0 = emb_c_t.clone().detach()

        t_enc = torch.randint(ddim_steps, (1,), device=unet.device)
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=unet.device)

        start_code = torch.randn((1, 4, 64, 64)).to(unet.device)

        with torch.no_grad():
            # generate an image with the concept
            z_c_e = quick_sample_till_t(emb_c_e.to(unet.device), start_guidance, start_code, int(t_enc))
            z_c_t = quick_sample_till_t(emb_c_t.to(unet.device), start_guidance, start_code, int(t_enc))

            # get conditional and unconditional scores from frozen model at time step t and image z_c_e
            eps_0_org = apply_model(orig_unet, z_c_e.to(orig_unet.device), t_enc_ddpm.to(orig_unet.device), emb_0.to(orig_unet.device))
            eps_e_org = apply_model(orig_unet, z_c_e.to(orig_unet.device), t_enc_ddpm.to(orig_unet.device), emb_c_e.to(orig_unet.device))
            eps_t_org = apply_model(orig_unet, z_c_t.to(orig_unet.device), t_enc_ddpm.to(orig_unet.device), emb_c_t.to(orig_unet.device))

        # breakpoint()
        # get conditional score
        eps_e = apply_model(unet, z_c_e.to(unet.device), t_enc_ddpm.to(unet.device), emb_c_e.to(unet.device))
        eps_t = apply_model(unet, z_c_t.to(unet.device), t_enc_ddpm.to(unet.device), emb_c_t.to(unet.device))

        eps_0_org.requires_grad = False
        eps_e_org.requires_grad = False
        eps_t_org.requires_grad = False

        # using DDIM inversion to project the x_t to x_0
        # check that the alphas is in descending order
        assert torch.all(ddim_alphas[:-1] >= ddim_alphas[1:])
        alpha_bar_t = ddim_alphas[int(t_enc)]
        eps_e_pred = (z_c_e - torch.sqrt(1 - alpha_bar_t) * eps_e) / torch.sqrt(alpha_bar_t)
        eps_t_pred = (z_c_t - torch.sqrt(1 - alpha_bar_t) * eps_t) / torch.sqrt(alpha_bar_t)

        eps_e_org_pred = (z_c_e - torch.sqrt(1 - alpha_bar_t) * eps_e_org) / torch.sqrt(alpha_bar_t)
        eps_0_org_pred = (z_c_e - torch.sqrt(1 - alpha_bar_t) * eps_0_org) / torch.sqrt(alpha_bar_t)
        eps_t_org_pred = (z_c_t - torch.sqrt(1 - alpha_bar_t) * eps_t_org) / torch.sqrt(alpha_bar_t)

        if i % args.pgd_num_steps == 0:
            # optimize the model
            loss = torch.tensor(0)
            loss += criteria(eps_e_pred.to(devices[0]), eps_0_org_pred.to(devices[0]) - (negative_guidance * (eps_e_org_pred.to(devices[0]) - eps_0_org_pred.to(devices[0]))))
            loss += args.lamda * criteria(eps_t_pred.to(devices[0]), eps_t_org_pred.to(devices[0])) # preserved concepts, output are the same without prompt

            # update weights to erase the concept
            loss.backward()
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            history_dict = save_to_dict(loss.item(), 'loss', history_dict)
            opt.step()
        else:
            # update the weight_pi vector
            opt.zero_grad()
            opt_weight_pi.zero_grad()
            unet.zero_grad()
            orig_unet.zero_grad()
            # weight_pi.grad = None
            loss = 0 
            loss -= criteria(eps_e_pred.to(devices[0]), eps_0_org_pred.to(devices[0]))
            loss -= args.lamda * criteria(eps_t_pred.to(devices[0]), eps_t_org_pred.to(devices[0])) # maximize the preserved loss
            loss.backward()
            preserved_set = preserved_dict[word]
            opt_weight_pi.step()
            history_dict = save_to_dict([weight_pi_dict[word].cpu().detach().numpy(), i, preserved_set[torch.argmax(weight_pi_dict[word], dim=1)], word], 'weight_pi', history_dict)
            history_dict = save_to_dict(weight_pi_dict, f'one_hot_dict_{i}', history_dict)

    unet.eval()
    unet.save_pretrained(args.save_dir)

def main(args: Arguments):
    train_age(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Finetuning stable diffusion model to erase concepts')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of erased_words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--info', help='info to add to model name', type=str, required=False, default='')
    parser.add_argument('--save_freq', help='frequency to save data, per iteration', type=int, required=False, default=10)
    parser.add_argument('--models_path', help='method of prompting', type=str, required=True, default='models')

    parser.add_argument('--gumbel_lr', help='learning rate for prompt', type=float, required=False, default=1e-3)
    parser.add_argument('--gumbel_temp', help='temperature for gumbel softmax', type=float, required=False, default=2)
    parser.add_argument('--gumbel_hard', help='hard for gumbel softmax, 0: soft, 1: hard', type=int, required=False, default=0, choices=[0,1])
    parser.add_argument('--gumbel_num_centers', help='number of centers for kmeans, if <= 0 then do not apply kmeans', type=int, required=False, default=100)
    parser.add_argument('--gumbel_update', help='update frequency for preserved set, if <= 0 then do not update', type=int, required=False, default=100)
    parser.add_argument('--gumbel_time_step', help='time step for the starting point to estimate epsilon', type=int, required=False, default=0)
    parser.add_argument('--gumbel_multi_steps', help='multi steps for calculating the output', type=int, required=False, default=2)
    parser.add_argument('--gumbel_k_closest', help='number of closest tokens to consider', type=int, required=False, default=1000)
    parser.add_argument('--gumbel_topk', help='number of top-k values in the soft gumbel softmax to be considered', type=int, required=False, default=5)
    parser.add_argument('--ignore_special_tokens', help='ignore special tokens in the embedding matrix', required=False, default=True)
    parser.add_argument('--vocab', help='vocab', type=str, required=False, default='EN3K')
    parser.add_argument('--pgd_num_steps', help='number of step to optimize adversarial concepts', type=int, required=False, default=2)
    parser.add_argument('--lamda', help='lambda for the loss function', type=float, required=False, default=1)

    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    train_age(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, args=args)

