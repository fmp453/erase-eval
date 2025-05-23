# Unified Concept Editing in Diffusion Models (UCE)

import ast
import copy
import operator
from functools import reduce

import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from train_methods.train_utils import get_devices, get_models, tokenize
from utils import Arguments


def edit_model(
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    old_text_,
    new_text_,
    retain_text_,
    layers_to_edit=None,
    lamb=0.1,
    erase_scale = 0.1,
    preserve_scale = 0.1,
    with_to_k=True,
    technique='tensor'
):
    
    ### collect all the cross attns modules
    sub_nets = unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
        
    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)
    
    if retain_text_ is None:
        ret_texts = ['']
    else:
        ret_texts = retain_text_

    print(old_texts, new_texts)
    ######### START ERASING #########
    for layer_num in range(len(projection_matrices)):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

            for cnt, t in enumerate(zip(old_texts, new_texts)):
                old_text = t[0]
                new_text = t[1]
                texts = [old_text, new_text]
                text_input = tokenize(texts, tokenizer)
                with torch.no_grad():
                    text_embeddings = text_encoder(text_input.input_ids.to(text_encoder.device))[0]
                  
                final_token_idx = text_input.attention_mask[0].sum().item()-2
                final_token_idx_new = text_input.attention_mask[1].sum().item()-2
                farthest = max([final_token_idx_new, final_token_idx])
                
                old_emb = text_embeddings[0]
                old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)]
                new_emb = text_embeddings[1]
                new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)]
                
                context = old_emb.detach()
                
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        if technique == 'tensor':
                            o_embs = layer(old_emb).detach()
                            u = o_embs
                            u = u / u.norm()
                            
                            new_embs = layer(new_emb).detach()
                            new_emb_proj = (u*new_embs).sum()
                            
                            target = new_embs - (new_emb_proj)*u 
                            values.append(target.detach()) 
                        elif technique == 'replace':
                            values.append(layer(new_emb).detach())
                        else:
                            values.append(layer(new_emb).detach())

                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += erase_scale*for_mat1
                mat2 += erase_scale*for_mat2

            for old_text, new_text in zip(ret_texts, ret_texts):
                text_input = tokenize([old_text, new_text], tokenizer)
                with torch.no_grad():
                    text_embeddings = text_encoder(text_input.input_ids.to(text_encoder.device))[0]
                old_emb, new_emb = text_embeddings
                context = old_emb.detach()
                
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        values.append(layer(new_emb[:]).detach())

                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += preserve_scale*for_mat1
                mat2 += preserve_scale*for_mat2
                #update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return unet


def train(args: Arguments):    
    
    technique = args.technique
    device = get_devices(args)[0]
    erase_scale = args.erase_scale
    anchor_concepts = args.anchor_concept

    concepts = args.concepts.split(',')
    concepts = [con.strip() for con in concepts]
    
    old_texts = []
    
    # removing prompt cleaning
    additional_prompts = []
    concepts_ = []
    for concept in concepts:
        old_texts.append(f'{concept}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(concept=concept))
        length = 1 + len(additional_prompts)
        concepts_.extend([concept] * length)
    
    guided_concepts = [con.strip() for con in anchor_concepts.split(',')]
    if len(guided_concepts) == 1:
        new_texts = [guided_concepts[0] for _ in old_texts]
    else:
        new_texts = [[con]*length for con in guided_concepts]
        new_texts = reduce(operator.concat, new_texts)

    retain_texts = ['']

    tokenizer, text_encoder, _, unet, _, _ = get_models(args)
    text_encoder.to(device)
    unet.to(device)

    unet = edit_model(
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        old_text_=old_texts,
        new_text_=new_texts,
        retain_text_=retain_texts,
        lamb=0.5,
        erase_scale=erase_scale,
        technique=technique
    )
    
    unet.eval()
    # contiguous
    for param in unet.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    unet.save_pretrained(args.save_dir)

def main(args: Arguments):
    train(args)
