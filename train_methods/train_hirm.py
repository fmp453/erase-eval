# https://github.com/Coffeeloveman/HiRM/blob/main/train.py


from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import trange
from transformers import CLIPTextModel, CLIPTokenizer

from utils import Arguments
from train_methods.train_utils import get_devices
        
       
def get_text_embeddings(text_encoder: CLIPTextModel, tokenized_text: torch.Tensor):
    # ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L377

    device = text_encoder.device
    weight_dtype = text_encoder.dtype

    text_embedding = text_encoder(tokenized_text.to(device))[0].to(weight_dtype)
    return text_embedding


def forward_with_cache(
    model: nn.Module, inputs: torch.Tensor, module: nn.Module, no_grad=True
) -> torch.Tensor:
    # define a tensor with the size of our cached activations
    cache = []
    
    def hook(module, input, output):
        
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]


def freeze_and_unfreeze_text_encoder(
    text_encoder: CLIPTextModel, method="all"
) -> CLIPTextModel:
    if method == "all":
        return text_encoder
    
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    mlps = False
    final_attn = False
    attns = False
    first_attn = False

    match method:
        case "mlp-only":
            mlps = True
        case "attn-only":
            attns = True
        case "mlp-attn":
            mlps = True
            attns = True
        case "mlp-final-attn":
            mlps = True
            final_attn = True    
        case "first-attn":
            first_attn = True
            mlps = True
        case "mix-attn":
            mlps = True
            final_attn = True
            first_attn = True
        case "final-attn":
            final_attn = True

    for param_name, module in text_encoder.named_modules():
        if mlps and "0.mlp.fc" in param_name and "10.mlp.fc" not in param_name:
            for param in module.parameters():
                param.requires_grad = True
        
        if attns and ".self_attn." in param_name:
            for param in module.parameters():
                param.requires_grad = True
        
        if final_attn and "11.self_attn." in param_name:
            for param in module.parameters():
                param.requires_grad = True       
        
        if first_attn and "0.self_attn." in param_name and "10.self_attn." not in param_name:
            for param in module.parameters():
                param.requires_grad = True

    return text_encoder


def train(args: Arguments):
    
    target_concept = args.concepts

    seed = args.seed
    coeff = args.hirm_steering_coeff
    save_path = args.save_dir
    num_epochs = args.hirm_epochs
    device = get_devices(args)[0]

    Path(save_path).mkdir(exist_ok=True)

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        args.sd_version, subfolder="tokenizer"
    )
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        args.sd_version, subfolder="text_encoder"
    )
    text_encoder.to(device)
        
    optimizer = optim.AdamW(
        text_encoder.parameters(),
        args.hirm_lr,
        (args.hirm_beta1, args.hirm_beta2),
        args.hirm_weight_decay,
        args.hirm_eps,
    )

    control_vectors_list = []
    torch.manual_seed(seed)
    
    # set random unit vector 
    single_random_vector = torch.rand(1, 77, 768, dtype=torch.float32, device=device)
    random_vector = single_random_vector.repeat(4, 1, 1) 
    control_vec = random_vector[0] / random_vector[0].norm() * coeff
    control_vectors_list.append(control_vec)

    history = {
        "loss": []
    }

    cnt = 0
    pbar = trange(0, num_epochs, desc="Epoch")

    text_encoder = freeze_and_unfreeze_text_encoder(text_encoder, method="first-attn")
    
    #set target layer
    target_model = text_encoder
    target_model.to(torch.float32)
    target_module = text_encoder.text_model.encoder.layers[11].self_attn.out_proj

    
    tokenized = tokenizer(
        target_concept,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    for epoch in pbar:

        loss_avg = 0
        text_encoder.train()
        
        _ = get_text_embeddings(text_encoder, tokenized)

        tokenized = tokenized.to(device)
        control_vec = torch.stack(control_vectors_list).to(device) 

        unlearn_inputs = {
            "input_ids": tokenized,
        }

        #forget loss
        updated_forget_activations = forward_with_cache(
            target_model, unlearn_inputs, module=target_module, no_grad=False
        ).to(device)
        
        loss = F.mse_loss(updated_forget_activations, control_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_avg += loss.detach().item()
        cnt += 1
        
        history["loss"].append(loss.detach().item())
        
        pbar.set_postfix(OrderedDict(loss=loss_avg / (cnt + 1e-9)))

    text_encoder.save_pretrained(f"{save_path}/epoch-{epoch+1}")


def main(args: Arguments):
    train(args)
