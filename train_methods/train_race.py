# R.a.c.e.: Robust adversarial concept erasure for secure text-to-image diffusion model

import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from diffusers import UNet2DConditionModel

from train_methods.train_utils import get_models, gather_parameters, sample_until, get_condition, get_devices, apply_model
from utils import Arguments


def pgd_attack(
    unet: UNet2DConditionModel,
    latent_model_input: torch.Tensor,
    t: int,
    text_embeddings: torch.Tensor,
    gaussian_noise: float,
    adv_loss: nn.MSELoss | nn.L1Loss,
    txt_min_max: dict[str, torch.Tensor],
    num_iter: int=10,
    alpha: float=2/255,
    epsilon: float=8/255,
) -> tuple[torch.Tensor, torch.Tensor]:
    #REF:https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html#PGD

    conditional_emb = text_embeddings.clone().detach().to(text_embeddings.device)
    adv_conditional_emb = conditional_emb.clone().detach()
    
    max_val = txt_min_max['max']
    min_val = txt_min_max['min']
    
    adv_conditional_emb = adv_conditional_emb + torch.empty_like(adv_conditional_emb).uniform_(-epsilon, epsilon)
    adv_conditional_emb = torch.clamp(adv_conditional_emb, min=min_val, max=max_val).detach()
    
    for _ in range(num_iter):
        adv_conditional_emb.requires_grad = True
        
        noise_pred = apply_model(unet, latent_model_input, t, adv_conditional_emb)
        
        # - delta is necessary because we are minimizing the loss
        loss: torch.Tensor = -adv_loss(gaussian_noise, noise_pred) 
        grad = torch.autograd.grad(loss, adv_conditional_emb, retain_graph=False, create_graph=False)[0]
        
        adv_conditional_emb = adv_conditional_emb.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_conditional_emb - conditional_emb, min=-epsilon, max=epsilon)
        adv_conditional_emb = torch.clamp(conditional_emb + delta, min=min_val, max=max_val).detach()

    return adv_conditional_emb, loss


def training(args: Arguments):
    prompt = args.concepts

    if args.seperator is not None:
        words = prompt.split(args.seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]

    devices = get_devices(args)

    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    tokenizer, text_encoder, _, unet_orig, ddim_scheduler, _ = get_models(args)
    text_encoder.eval()
    text_encoder.to(devices[0])
    unet_orig.eval()
    unet_orig.to(devices[1])
    unet.to(devices[0])
    
    if args.race_adv_train: #If it is adv training, load pretrained ESD model's weight.
        try:
            print("Adv Training is activated")
            print("load Unet part from pretrained esd")
            unet.from_pretrained(args.race_esd_path)
        except Exception as e:
            print(f'Model path is not valid, please check the file name and structure: {e}')
            exit() 

    _, parameters = gather_parameters(args.esd_method, unet)

    unet.train()
    quick_sample_till_t = lambda x, s, code, t: sample_until(
        until=t,
        latents=code,
        unet=unet,
        scheduler=ddim_scheduler,
        prompt_embeds=x,
        guidance_scale=s,
    )

    if args.race_lasso:
        print("Lasso is activated")
        lasso_loss_fn = nn.L1Loss()
        lasso_lambda = 0.1
        original_parameters = [param.clone().detach() for param in parameters]

    opt = optim.Adam(parameters, lr=args.race_lr)
    criteria = nn.MSELoss()
    
    #RACE
    if args.race_adv_train:
        if args.race_adv_loss == "l1":
            adv_loss = nn.L1Loss()
        elif args.race_adv_loss == "l2":
            adv_loss = nn.MSELoss()
        else:
            raise ValueError('Invalid adv_loss')
    
    
    # TRAINING CODE
    pbar = trange(args.race_iterations)
    for _ in pbar:
        word = random.sample(words, 1)[0]
        emb_0 = get_condition([''], tokenizer, text_encoder)
        emb_p = get_condition([word], tokenizer, text_encoder)
        emb_n = get_condition([f'{word}'], tokenizer, text_encoder)
        
        txt_emb_min_max = {'min': emb_n.min(), 'max': emb_n.max()}

        opt.zero_grad()

        t_enc = torch.randint(args.ddim_steps, (1,))
        og_num = round((int(t_enc) / args.ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / args.ddim_steps) * 1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            z = quick_sample_till_t(emb_p, args.start_guidance, start_code, int(t_enc))
            e_0 = apply_model(unet_orig, z, t_enc_ddpm, emb_0)
            e_p = apply_model(unet_orig, z, t_enc_ddpm, emb_p)
        
        if args.race_adv_train:
            attacked_emb_n, loss = pgd_attack(
                unet, 
                z, 
                t_enc_ddpm, 
                emb_n.to(devices[0]), 
                start_code, 
                adv_loss, 
                txt_emb_min_max, 
                alpha=args.race_epsilon/4., 
                num_iter=10, 
                epsilon=args.race_epsilon
            )
        else:
            attacked_emb_n = emb_n

        # get conditional score from ESD model
        e_n = apply_model(unet, z, t_enc_ddpm, attacked_emb_n)
        e_0.requires_grad = False
        e_p.requires_grad = False
        loss: torch.Tensor = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (args.negative_guidance * (e_p.to(devices[0]) - e_0.to(devices[0]))))

        if args.race_lasso:
            lasso_loss = sum(lasso_loss_fn(param, original_param) for param, original_param in zip(parameters, original_parameters))
            loss += lasso_lambda * lasso_loss

        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        opt.step()

        if loss > 500.:
            raise ValueError('Loss is too high, training is not working')
        
    unet.eval()
    unet.save_pretrained(args.save_dir)


def main(args: Arguments):
    training(args)
