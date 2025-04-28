import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torch.autograd import Variable
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL

from train_methods.train_utils import sample_until, apply_model
from train_methods.utils_age import ConceptDict
from train_methods.train_utils import save_embedding_matrix, learn_k_means_from_input_embedding, search_closest_tokens, get_condition
from train_methods.consts import ddim_alphas
from utils import Arguments


def train_age(args: Arguments) -> None:
    '''
    train_method : str
        The parameters to train for erasure (noxattn, xattan).
    '''

    # const
    ddim_steps = args.ddim_steps
    train_method = args.age_method

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
    opt = optim.Adam(parameters, lr=args.age_lr)
    criteria = nn.MSELoss()
    history_dict = {}

    pbar = trange(args.pgd_num_steps * args.age_iters)

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
        
        top_k_tokens, sorted_sim_dict = search_closest_tokens(
            word,
            tokenizer,
            text_encoder,
            k=args.gumbel_k_closest,
            sim='l2',
            model_name='SD-v1-4',
            ignore_special_tokens=args.ignore_special_tokens,
            vocab=vocab
        )
        all_sim_dict[word] = {key:sorted_sim_dict[key] for key in top_k_tokens}

    if args.gumbel_num_centers > 0:
        assert args.gumbel_num_centers % len(erased_words) == 0, 'Number of centers should be divisible by number of erased words'
    preserved_dict = dict()

    for word in erased_words:
        temp = learn_k_means_from_input_embedding(sim_dict=all_sim_dict[word], num_centers=args.gumbel_num_centers)
        preserved_dict[word] = temp

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
        weight_pi = torch.zeros((1, preserved_matrix.shape[0]), device=unet.device, dtype=preserved_matrix.dtype) # [1, n]
        weight_pi = weight_pi + 1 / preserved_matrix.shape[0]
        weight_pi = Variable(weight_pi, requires_grad=True)
        weight_pi_dict[erase_word] = weight_pi
        preserved_matrix_dict[erase_word] = preserved_matrix
    
    print('weight_pi_dict:', weight_pi_dict)

    # optimizer for all pi vectors
    opt_weight_pi = optim.Adam([weight_pi for weight_pi in weight_pi_dict.values()], lr=args.gumbel_lr)

    """
    Gumbel-Softmax function
        if `hard` is 1, then it is one-hot, if `hard` is 0, then it is a new soft version, which takes the top-k highest values and normalize them to 1
    """
    def gumbel_softmax(logits, temperature=args.gumbel_temp, hard=args.gumbel_hard, eps=1e-10, k=args.gumbel_topk) -> torch.Tensor:
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
            z_c_e = quick_sample_till_t(emb_c_e.to(unet.device), args.start_guidance, start_code, int(t_enc))
            z_c_t = quick_sample_till_t(emb_c_t.to(unet.device), args.start_guidance, start_code, int(t_enc))

            # get conditional and unconditional scores from frozen model at time step t and image z_c_e
            eps_0_org = apply_model(orig_unet, z_c_e.to(orig_unet.device), t_enc_ddpm.to(orig_unet.device), emb_0.to(orig_unet.device))
            eps_e_org = apply_model(orig_unet, z_c_e.to(orig_unet.device), t_enc_ddpm.to(orig_unet.device), emb_c_e.to(orig_unet.device))
            eps_t_org = apply_model(orig_unet, z_c_t.to(orig_unet.device), t_enc_ddpm.to(orig_unet.device), emb_c_t.to(orig_unet.device))

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
            loss: torch.Tensor = 0
            loss += criteria(eps_e_pred.to(unet.device), eps_0_org_pred.to(unet.device) - (args.negative_guidance * (eps_e_org_pred.to(unet.device) - eps_0_org_pred.to(unet.device))))
            loss += args.age_lamda * criteria(eps_t_pred.to(unet.device), eps_t_org_pred.to(unet.device)) # preserved concepts, output are the same without prompt

            loss.backward()
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            opt.step()
        else:
            # update the weight_pi vector
            opt.zero_grad()
            opt_weight_pi.zero_grad()
            unet.zero_grad()
            orig_unet.zero_grad()
            loss = 0 
            loss -= criteria(eps_e_pred.to(unet.device), eps_0_org_pred.to(unet.device))
            loss -= args.age_lamda * criteria(eps_t_pred.to(unet.device), eps_t_org_pred.to(unet.device)) # maximize the preserved loss
            loss.backward()
            preserved_set = preserved_dict[word]
            opt_weight_pi.step()
            
    unet.eval()
    unet.save_pretrained(args.save_dir)

def main(args: Arguments):
    train_age(args)

