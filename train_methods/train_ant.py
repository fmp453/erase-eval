# https://github.com/lileyang1210/ANT
# Set You Straight: Auto-Steering Denoising Trajectories to Sidestep Unwanted Concepts (ACMMM 2025)


import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import UNet2DConditionModel
from tqdm import tqdm

from train_methods.train_utils import get_devices, get_models, get_condition, apply_model, gather_parameters, sample_until, seed_everything
from utils import Arguments


def train_ant(args: Arguments):
    if args.ant_if_gradient:
        seed_everything(args.seed)

    devices = get_devices(args)
    gradients = defaultdict(float)
    save_path = Path(args.save_dir)
    gradient_path = Path("gradient", f"{args.ant_method}_{args.ant_lr}")

    words = [args.concepts.split(",")]

    tokenizer, text_encoder, _, unet, scheduler, _ = get_models(args)
    original_unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")

    _, parameters = gather_parameters(args.ant_method, unet)

    text_encoder.to(devices[0])
    unet.to(devices[0])
    original_unet.to(devices[1])
    unet.train()
    quick_sample_till_t = lambda x, s, code, t: sample_until(
        until=t,
        latents=code,
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=x,
        guidance_scale=s,
    )

    losses = []
    opt = optim.Adam(parameters, lr=args.ant_lr)
    criteria = nn.MSELoss()
    history = []

    pbar = tqdm(range(args.ant_iterations))
    for _ in pbar:
        word = random.sample(words, 1)[0]
        emb_0 = get_condition([''], tokenizer, text_encoder)
        emb_p = get_condition([word], tokenizer, text_encoder)
        emb_n = get_condition([f'{word}'], tokenizer, text_encoder)

        opt.zero_grad()

        t_enc_plus = torch.randint(args.ant_before_step, (1,))
        t_enc_minus = torch.randint(args.ant_before_step, args.ddim_steps, (1,))
        # Time step from 1000 to 0
        og_num_plus = round((int(t_enc_plus) / args.ddim_steps) * 1000)
        og_num_minus = round((int(t_enc_minus) / args.ddim_steps) * 1000)
        og_num_lim_plus = round((int(t_enc_plus + 1) / args.ddim_steps) * 1000)
        og_num_lim_minus = round((int(t_enc_minus + 1) / args.ddim_steps) * 1000)

        t_enc_ddpm_plus = torch.randint(og_num_plus, og_num_lim_plus, (1,))
        t_enc_ddpm_minus = torch.randint(og_num_minus, og_num_lim_minus, (1,))
        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            z_plus = quick_sample_till_t(emb_p.to(devices[0]), args.start_guidance, start_code, int(t_enc_plus))
            z_minus = quick_sample_till_t(emb_p.to(devices[0]), args.start_guidance, start_code, int(t_enc_minus))
            e_0_plus = apply_model(original_unet, z_plus, t_enc_ddpm_plus, emb_0)
            e_0_minus = apply_model(original_unet, z_minus, t_enc_ddpm_minus, emb_0)
            e_n0_plus = apply_model(unet, z_plus, t_enc_ddpm_plus, emb_0)
            e_n0_minus = apply_model(unet, z_minus, t_enc_ddpm_minus, emb_0)
            e_p_plus = apply_model(original_unet, z_plus, t_enc_ddpm_plus, emb_p)
            e_p_minus = apply_model(original_unet, z_minus, t_enc_ddpm_minus, emb_p)
        
        e_n_plus = apply_model(unet, z_plus, t_enc_ddpm_plus, emb_n)
        e_n_minus = apply_model(unet, z_minus, t_enc_ddpm_minus, emb_n)
        e_0_plus.requires_grad = False
        e_0_minus.requires_grad = False
        e_p_plus.requires_grad = False
        e_p_minus.requires_grad = False
        
        # The loss function of ANT model
        loss_1 = criteria(e_n_plus.to(devices[0]), e_0_plus.to(devices[0]) + (args.negative_guidance * (e_p_plus.to(devices[0]) - e_0_plus.to(devices[0])))) 
        loss_3 = criteria(e_n_minus.to(devices[0]), e_0_minus.to(devices[0]) - (args.negative_guidance * (e_p_minus.to(devices[0]) - e_0_minus.to(devices[0]))))
        loss_2 = criteria(e_0_plus.to(devices[0]), e_n0_plus.to(devices[0]))
        loss_4 = criteria(e_0_minus.to(devices[0]), e_n0_minus.to(devices[0]))
        loss: torch.Tensor = loss_3 + args.ant_alpha_2 * loss_4 + args.ant_alpha_1 * (loss_1 + args.ant_alpha_2 * loss_2)
        loss.backward()

        if args.ant_if_gradient:
            with torch.no_grad():
                for name, param in unet.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data.cpu()
        if args.ant_mask_path:
            mask: dict[str, torch.Tensor] = torch.load(args.ant_mask_path)
            for name, param in unet.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name].to(devices[0])

        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

    unet.eval()

    if args.ant_if_gradient:
        with torch.no_grad(): 
            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])
            gradient_path.mkdir(parents=True, exist_ok=True)
            torch.save(gradients, f"{gradient_path}/gradient_{args.seed}.pt")
    else:
        save_path.mkdir(parents=True, exist_ok=True)
        unet.save_pretrained(save_path)

def main(args: Arguments):
    train_ant(args)
