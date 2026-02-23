# Localizing and Editing Knowledge in Text-to-Image Generative Models (DiffQuickFix)
# almost of all is copied from https://github.com/adobe-research/DiffQuickFixRelease/blob/main/causal_trace_model_edit.ipynb

from copy import deepcopy

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.models.clip.modeling_clip import CLIPAttention
from diffusers import StableDiffusionPipeline

from train_methods.train_utils import get_devices, seed_everything, tokenize
from utils import Arguments



# maybe under construction
def train_edit(model, projection_matrices, og_matrices, contexts, values, old_texts, new_texts, lamb=0.01):
    # Contexts for the three projection matrices
    context_k = contexts
  
    # Values for the three projection matrices
    values_k = values
   
    # Iterate through the projection matrices
    for layer_num in range(len(projection_matrices)):
        # First layer is v, then k, then q
        with torch.no_grad():
            # mat1 : \lambda Q + \sum v k^{T}
            mat1 = lamb * projection_matrices[layer_num].weight 
            
            # mat2 : \lambda I  + \sum k k^{T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

            # Number of updates
            c_total = 0
            for context, value in zip(context_k, values_k):
                # Context vector # 
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                # 1 x 768
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])

                # 768 x 1
                value_vector = value.reshape(value.shape[0], value.shape[1], 1)

                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)

                # Update the matrix
                mat1 += for_mat1 
                mat2 += for_mat2 

                c_total += 1
            

        # Projection matric
        projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    # Finished Updating the Weight Matrices 
    print(f'Finished Updating the weight matrices of the self-attention layers ... ')

    return 

def train(args: Arguments):
    seed = args.seed
    anchor_prompt = args.concepts

   # Define the target prompt: Prompt to which the original anchor_prompt needs to be translated to 
    target_prompt = args.anchor_concept

    torch.cuda.empty_cache()
    device = get_devices(args)[0]
    
    sd_pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(args.sd_version)
    sd_pipeline.safety_checker = None
    sd_pipeline = sd_pipeline.to(device)
    text_encoder: CLIPTextModel = sd_pipeline.text_encoder
    tokenizer: CLIPTokenizer = sd_pipeline.tokenizer

    seed_everything(seed)

    # Define the self-attention layer which needs to be edited 
    self_layer = 0 ## Default = 0, found via causal tracing in the paper

    # Regularization for the optimization
    reg = 0.01 

    # Use EOS # : This is a flag variable which lets you use the EOS tokens also in the editing optimization step
    use_eos = 1

    # Obtain the projection layers of the self-attentions which needs to be udpated 
    # Layers which need to get appended
    ca_layers: list[CLIPAttention] = []
    for name, m in text_encoder.named_modules():
        if f'encoder.layers.{self_layer}.self_attn' == name:
            ca_layers.append(m)

    # Projection Matrices
    projection_matrices = [l.out_proj for l in ca_layers]
    og_matrices = [deepcopy(l.out_proj) for l in ca_layers]

    # CREATING THE TRAINING DATA FOR MODEL EDITING 
    # # Setup sentences #
    old_texts = [anchor_prompt]
    new_texts = [target_prompt]

    # Use simple augmentations to expand the training data 
    base = old_texts[0] if old_texts[0][0:1] != "A" else "a" + old_texts[0][1:]
    old_texts.append("A photo of " + base)
    old_texts.append("An image of " + base)
    old_texts.append("A picture of " + base)
    base = new_texts[0] if new_texts[0][0:1] != "A" else "a" + new_texts[0][1:]
    new_texts.append("A photo of " + base)
    new_texts.append("An image of " + base)
    new_texts.append("A picture of " + base)

    # Obtain the projection matrix embeddings for the tokens of the training data 
    text_modules = []
    module_k_proj = None 

    # Iterate through the text_model modules
    for n, m in text_encoder.named_modules():
        # Only use the out_proj layer as output
        if f'encoder.layers.{self_layer}.self_attn.out_proj' in n:
            text_modules.append(n)
            module_k_proj = m 

    # Context / Values 
    context_k, values_k = [], []

    # Hook attachment for text-embedder
    def hook_attach(name):
        def hook(model, inp: torch.Tensor, out: torch.Tensor):
            if name == 'out_proj':
                context_k.append(inp[0][0].detach())
                values_k.append(out[1].detach())

        return hook 

    # Register the hook to obtain the token embeddings # 
    h1 = module_k_proj.register_forward_hook(hook_attach('out_proj'))

    # # Stores the index of the subjects of the old and new texts
    subject_old_texts = []
    subject_new_texts = []

    # Function to obtain the relevant set of tokens from the tokenizer output #
    def find_index(token_temp):
        # Subject old / new tokens
        sub_old_tokens = token_temp[0]
        sub_new_tokens = token_temp[1]

        return_old = 0
        return_new = 0
        c = 0
        for tok in sub_old_tokens:
            if tok == 49407:
                return_old = c - 1
                break
            c += 1

        c = 0
        for tok in sub_new_tokens:
            if tok == 49407:
                return_new = c - 1
                break
            c += 1

        return return_old, return_new

    t_c = 0
    
    # Iterate through the texts and obtain the token embeddings 
    for old_text, new_text in zip(old_texts, new_texts):
        # Tokens Old
        tokens: torch.Tensor = tokenize(prompt=[old_text, new_text], tokenizer=tokenizer).input_ids 
        
        # Obtain the relevant token indexes 
        old_index, new_index = find_index(tokens)
        
        temp_k = []
        temp_k.append(old_index)
        for j in range(old_index+1, 77):
            # ADDED - EOS Token for the keys
            temp_k.append(j)

        # Subject old text tokens
        subject_old_texts.append(temp_k)

        temp_val = []
        temp_val.append(new_index)
        for j in range(new_index+1, 77):
            # ADDED - EOS Token for the values
            temp_val.append(j)

        subject_new_texts.append(temp_val)

        # Call the text-embedding model so that the relevant tokens are saved # 
        with torch.no_grad():
            _ = text_encoder(tokens.to(device))[0]

        t_c += 1

    # Refined context / values
    refined_context_k, refined_values_k = [], []
    contextss = []
    valuess = []

    # Iterate through the context, values vectors
    for i in range(0, len(context_k)):
        # Update the contexts 
        key_indexes = []
        values_indexes = []

        refined_context_k.append(context_k[i][subject_old_texts[i][0]])

        # Changed
        key_indexes.append(subject_old_texts[i][0])

        temp = 0
        for j in range(1, len(subject_old_texts[i])):
            if use_eos == 1:
                refined_context_k.append(context_k[i][subject_old_texts[i][j]])
                key_indexes.append(subject_old_texts[i][j]) # Added
            temp += 1

        refined_values_k.append(values_k[i][subject_new_texts[i][0]])
        values_indexes.append(subject_new_texts[i][0])

        temp = 0
        if len(subject_old_texts[i]) <= len(subject_new_texts[i]):
            max_turn = len(subject_old_texts[i])
        else:
            max_turn = len(subject_new_texts[i])
        
        for j in range(1, max_turn):
            if use_eos == 1:
                refined_values_k.append(values_k[i][subject_new_texts[i][j]])
                values_indexes.append(subject_new_texts[i][j]) # Changed

            temp += 1

        if len(key_indexes) <= len(values_indexes):
                contextss.append(context_k[i][key_indexes])
                valuess.append(values_k[i][values_indexes])
        else:
            contextss.append(context_k[i][key_indexes])
            diff = len(key_indexes) - len(values_indexes)
            for k in range(0, diff):
                values_indexes.append(76)

            valuess.append(values_k[i][values_indexes])


    #  END OF OBTAINING THE RELEVANT TOKEN EMBEDDINGS
    # Check the number of keys and values 
    print(f'Length of Keys: {len(refined_context_k)}')
    print(f'Length of Values: {len(refined_values_k)}')

    train_edit(text_encoder, projection_matrices, og_matrices, contextss, valuess, old_texts, new_texts, lamb=reg)
    print(f'Training Finished .....')
    h1.remove()

    text_encoder.save_pretrained(args.save_dir)

def main(args: Arguments):
    train(args)
