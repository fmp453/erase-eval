# On Mechanistic Knowledge Localization in Text-to-Image Generative Models (LocoGen)

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel

from utils import Arguments


# Total number of computable operations / modules -- 709
def high_level_layers(unet: UNet2DConditionModel):
    # Counter for the list
    
    # Total list of all modules
    named_module_list = []
    for n, _ in unet.named_modules():
        named_module_list.append(n)

    # Ends with 'attn2', 'attn1'
    attn_list = []
    for item in named_module_list:
        if 'attn2' in item and ('to_k' in item or 'to_v' in item):
            attn_list.append(item)
    
    return attn_list

# Model Editing Function - Non-SDXL models
def train_edit(args: Arguments, layer_edit_modules, key_embeddings, value_embeddings):
    
    # Iterate through each of the modules and then update the modules based on the closed-form expression
    for layer_num in range(0, len(layer_edit_modules)):
        # Updatability Part
        with torch.no_grad():
            
            # Current Weight Matrix; 
            curr_weight_matrix = layer_edit_modules[layer_num].weight

            ############  First part of the solution ############
            id_matrix_mat_1 = args.reg_key * torch.eye(layer_edit_modules[layer_num].weight.shape[1], device = layer_edit_modules[layer_num].weight.device)
            x_matrix = torch.matmul(key_embeddings.T, key_embeddings)
            mat1 = torch.inverse(x_matrix + id_matrix_mat_1)

            ############  Second part of the solution ###########
            # X^{T}Y
            x_matrix_mat_2 = torch.matmul(key_embeddings.T, torch.matmul(value_embeddings, curr_weight_matrix.T))
            additional_reg = args.reg_key * curr_weight_matrix.T 
            mat2 = x_matrix_mat_2 + additional_reg

            # Final Update due to the least squared solution
            final_update = torch.matmul(mat1, mat2)

            # Update the layer 
            layer_edit_modules[layer_num].weight = torch.nn.Parameter(final_update.T)
            
    return 

# LocoEdit function
def train(args: Arguments):
    
    device = torch.device(f'cuda:{args.device.split(",")[0]}')

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    
    text_encoder.to(device)
    unet.to(device)

    # Relevant layers 
    relevant_layers = sorted(high_level_layers(unet))

    # Start location # 
    start_loc = args.start_loc # start_loc = 9 is also another option
    relevant_edit_layers = relevant_layers[start_loc*2: start_loc*2 + args.seq]
    print(f'Relevant Editing layers : {relevant_edit_layers}')

    def design_styles(artist):
        # Prompts : With basic augmentations
        prompts = ['' + artist, 'a painting in the style of ' + artist, 'a photo in the style of ' + artist, 'a picture in the style of ' + artist]
        return prompts
    
    def design_objects(artist):
        # Prompts : With basic augmentations
        prompts = ['' + artist, 'an image of ' + artist, 'a photo of ' + artist, 'a picture of ' + artist]
        return prompts
    
    # Function to generate output embeddings from the text-encoder
    def generate_text_embeddings(tokenizer: CLIPTokenizer, text_encoder, key_prompt, value=False):
        # Obtaining the embeddings of the last subject token
        # Key : Text-Embedding
        key_embeddings = []
        key_tokens = []
        for prompt_curr in key_prompt:
            text_input_curr= tokenizer(prompt_curr, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",)
            # Append the embeddings
            with torch.no_grad():
                text_embeddings = text_encoder(text_input_curr.input_ids.to(text_encoder.device))[0]
                key_embeddings.append(text_embeddings[0])
                key_tokens.append(text_input_curr['input_ids'][0])

        # Storing the final key embeddings
        final_key_embeddings = []

        # Iterate through the text embeddings and extract the last-subject-token embeddings
        c = 0
        for txt_embedding in key_embeddings:
            token_ids = key_tokens[c]
            # Position
            pos = 0
            for tok in token_ids:
                # If last subject token is encountered -- then break 
                if tok == 49407:
                    break
                pos += 1
            
            if value == False:
                # Relevant embedding
                if args.eos == 'False':
                    # Embedding
                    rel_embedding = txt_embedding[pos-1]
                    final_key_embeddings.append(rel_embedding.reshape(1,-1))
                
                else:
                    # EOS function
                    print(f'Using EOS')
                    for k in range(pos-1, pos):
                        rel_embedding = txt_embedding[k]
                        final_key_embeddings.append(rel_embedding.reshape(1,-1))
            else:
                # Embedding
                rel_embedding = txt_embedding[pos-1]
                final_key_embeddings.append(rel_embedding.reshape(1,-1))
            c += 1
        # Size of embeddings
        final_keys = torch.cat(final_key_embeddings, dim=0)

        # Return the final Keys
        return final_keys 

    # Key prompts
    key_prompt = design_styles(args.concepts) if args.loco_concept_type == "style" else design_objects(args.concepts)
    print(key_prompt)
    layer_edit_modules = []
    for l in relevant_edit_layers:
        # Iterate through the modules in UNet
        for n, m in unet.named_modules():
            if n == l:
                layer_edit_modules.append(m)
    
    # Finished storing the layers which are edited
    print(f'Number of the layers which are edited : {len(layer_edit_modules)}')
    key_embeddings = generate_text_embeddings(tokenizer, text_encoder, key_prompt)
    target_prompt = ['a painting'] * len(key_embeddings) if args.loco_concept_type == "style" else ["a photo"] * len(key_embeddings)
    
    print(target_prompt)
    # Flag
    value_embeddings = generate_text_embeddings(tokenizer, text_encoder, target_prompt, value=True)

    print(f'Size of Key Embeddings : {len(key_embeddings)}')
    print(f'Size of Value embeddings : {len(value_embeddings)}')

    train_edit(args, layer_edit_modules, key_embeddings, value_embeddings)

    # contiguous
    for param in unet.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    unet.save_pretrained(args.save_dir)
    

def main(args: Arguments):
    train(args)
