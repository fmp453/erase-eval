# MACE: Mass Concept Erasure in Diffusion Models

import os
import gc
import math
import random
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler

from train_methods.train_utils import prepare_k_v, get_ca_layers, closed_form_refinement, importance_sampling_fn
from train_methods.train_utils import AttnController, LoRAAttnProcessor
from train_methods.segment_anything.segment_anything import SamPredictor, sam_hq_model_registry
from train_methods.groundingdino.models import build_model
from train_methods.groundingdino.util.slconfig import SLConfig
from train_methods.groundingdino.util.utils import clean_state_dict
from train_methods.mace_data import MACEDataset

from utils import Arguments

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def inference(args: Arguments, device: str, multi_concept: list[list[str]], output_dir: str):
    
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_version).to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    torch.Generator(device=device).manual_seed(42)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    num_images = 8
    cnt = 0
    for single_concept in multi_concept:
        c, t = single_concept
        cnt += 1
        c = c.replace("-", "")
        output_dir = f"{output_dir}/{c}".replace(" ", "-")
        os.makedirs(output_dir, exist_ok=True)

        if t == "object":
            prompt = f"a photo of the {c}"
        elif t == "style":
            prompt = f"a photo in the style of {c}"
        else:
            raise ValueError("unknown concept type.")
        
        images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=num_images).images
        for i, im in enumerate(images):
            im.save(f"{output_dir}/{prompt.replace(' ', '-')}_{i}.jpg")
        
        del images
        torch.cuda.empty_cache()
        gc.collect()
    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

def load_model(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def get_phrases_from_posmap(posmap: torch.BoolTensor, tokenized: dict, tokenizer: AutoTokenizer, left_idx: int = 0, right_idx: int = 255):
    assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
    if posmap.dim() == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def get_mask(input_image, text_prompt, model, predictor, device, output_dir=None, box_threshold=0.3, text_threshold=0.25):
    
    # make dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
    image = input_image
    
    # run grounding dino model
    boxes_filt, _ = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device=device)
        
    image_np = image.cpu().numpy()
    image_np = ((image_np / max(image_np.max().item(), abs(image_np.min().item())) + 1) * 255 * 0.5).astype(np.uint8)
    
    # C x H x W  to  H x W x C
    if image_np.ndim == 3 and image_np.shape[0] in {1, 3}:
        image_np = image_np.transpose(1, 2, 0)

    image = image_np
    predictor.set_image(image)

    size = image.shape
    H, W = size[0], size[1]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    if len(transformed_boxes) == 0:
        masks = torch.ones((1, 1, H, W), dtype=torch.bool)
    else:
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )

    final_mask = torch.zeros_like(masks[0].unsqueeze(0))
    # if many masks
    if masks.shape[0] > 1:
         for i in range(masks.shape[0]):
            final_mask = final_mask | masks[i]
    else:
        final_mask = masks
    
    return final_mask

def making_data(args: Arguments):
    device = torch.device(f'cuda:{args.device.split(",")[0]}')

    multi_concept = []
    concepts = args.concepts.split(",")
    concept_types = args.mace_concept_type.split(",")
    
    assert len(concepts) == len(concept_types)
    for i in range(len(concept_types)):
        multi_concept.append([concepts[i], concept_types[i]])
    
    # generate 8 images per concept using the original model for performing erasure
    inference(args, device, multi_concept=multi_concept, output_dir=args.data_dir)

    # get and save masks for each image
    grounded_model = load_model(args.grounded_config, args.grounded_checkpoint)
        
    predictor = SamPredictor(sam_hq_model_registry['vit_h'](checkpoint=args.sam_hq_checkpoint).to(device))
    
    transform = transforms.ToTensor()
    for root, _, files in os.walk(args.data_dir):
        mask_save_path = root.replace(f'{os.path.basename(root)}', f'{os.path.basename(root)}-mask')
        os.makedirs(mask_save_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(root, file)
            # print(file_path)
            # read images and get masks
            image = Image.open(file_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            tensor_image = transform(image).to(device)
            GSAM_mask = get_mask(tensor_image, os.path.basename(root), grounded_model, predictor, device)
            # save masks
            GSAM_mask = (GSAM_mask.to(torch.uint8) * 255).squeeze()
            save_mask = to_pil_image(GSAM_mask)
            save_mask.save(f"{os.path.join(mask_save_path, file).replace('.jpg', '_mask.jpg')}")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    concept_positions = [example["concept_positions"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    masks = [example["instance_masks"] for example in examples]
    instance_prompts =  [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    if masks[0] is not None: 
        ## object/celebrity erasure
        masks = torch.stack(masks)
    else:
        ## artistic style erasure
        masks = None
    
    input_ids = torch.cat(input_ids, dim=0)
    concept_positions = torch.cat(concept_positions, dim=0).type(torch.BoolTensor)

    batch = {
        "instance_prompts": instance_prompts,
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "masks": masks,
        "concept_positions": concept_positions,
    }
    return batch

def cfr_lora_training(args: Arguments):
    
    device = torch.device(f'cuda:{args.device.split(",")[0]}')
    mapping_concept = args.anchor_concept.split(",")
    multi_concept = []
    concepts = args.concepts.split(",")
    concept_types = args.mace_concept_type.split(",")
    
    assert len(concepts) == len(concept_types)
    for i in range(len(concept_types)):
        multi_concept.append([concepts[i], concept_types[i]])
    
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.sd_version, subfolder="tokenizer")
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.sd_version, subfolder="unet")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.sd_version, subfolder="text_encoder")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.sd_version, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_version, subfolder="scheduler")
    
    unet.to(device)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    optimizer_class = torch.optim.AdamW
    
    train_dataset = MACEDataset(
        tokenizer=tokenizer,
        size=args.image_size,
        multi_concept=multi_concept,
        mapping=mapping_concept,
        batch_size=args.mace_train_batch_size,
        train_seperate=args.mace_train_seperate,
        input_data_path=args.data_dir.replace(" ", "-"),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mace_train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.mace_dataloader_num_workers,
    )
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataloader)
        
    # Afterwards we recalculate our number of training epochs
    args.mace_num_train_epochs = math.ceil(args.mace_max_train_steps / num_update_steps_per_epoch)
        
    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(device)
    text_encoder.to(device)

    # stage 1: closed-form refinement
    projection_matrices, ca_layers, og_matrices = get_ca_layers(unet, with_to_k=True)
    
    # to save memory
    CFR_dict = {}
    max_concept_num = args.mace_max_memory # the maximum number of concept that can be processed at once
    if len(train_dataset.dict_for_close_form) > max_concept_num:
        
        for layer_num in tqdm(range(len(projection_matrices))):
            CFR_dict[f'{layer_num}_for_mat1'] = None
            CFR_dict[f'{layer_num}_for_mat2'] = None
            
        for i in tqdm(range(0, len(train_dataset.dict_for_close_form), max_concept_num)):
            contexts_sub, valuess_sub = prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, train_dataset.dict_for_close_form[i:i+5], tokenizer, all_words=False)
            closed_form_refinement(projection_matrices, contexts_sub, valuess_sub, cache_dict=CFR_dict, cache_mode=True)
            
            del contexts_sub, valuess_sub
            gc.collect()
            torch.cuda.empty_cache()
            
    else:
        for layer_num in tqdm(range(len(projection_matrices))):
            CFR_dict[f'{layer_num}_for_mat1'] = .0
            CFR_dict[f'{layer_num}_for_mat2'] = .0
            
        contexts, valuess = prepare_k_v(text_encoder, projection_matrices, ca_layers, og_matrices, train_dataset.dict_for_close_form, tokenizer, all_words=False)
    
    del ca_layers, og_matrices

    # Load cached prior knowledge for preserving
    prior_preservation_cache_dict = {}
    for layer_num in tqdm(range(len(projection_matrices))):
        prior_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
        prior_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
            
    # Load cached domain knowledge for preserving
    domain_preservation_cache_dict = {}
    for layer_num in tqdm(range(len(projection_matrices))):
        domain_preservation_cache_dict[f'{layer_num}_for_mat1'] = .0
        domain_preservation_cache_dict[f'{layer_num}_for_mat2'] = .0
    
    # integrate the prior knowledge, domain knowledge and closed-form refinement
    cache_dict = {}
    for key in CFR_dict:
        cache_dict[key] = args.mace_train_preserve_scale * (prior_preservation_cache_dict[key] + args.mace_preserve_weight * domain_preservation_cache_dict[key]) + CFR_dict[key]
    
    # closed-form refinement
    projection_matrices, _, _ = get_ca_layers(unet, with_to_k=True)
    
    if len(train_dataset.dict_for_close_form) > max_concept_num:
        closed_form_refinement(projection_matrices, lamb=args.mace_lamb, preserve_scale=1, cache_dict=cache_dict)
    else:
        closed_form_refinement(projection_matrices, contexts, valuess, lamb=args.mace_lamb, preserve_scale=args.mace_train_preserve_scale, cache_dict=cache_dict)
    
    del contexts, valuess, cache_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    # stage 2: multi-lora training
    for i in range(train_dataset._concept_num): # the number of concept/lora
        
        attn_controller = AttnController()
        if i != 0:
            unet.set_default_attn_processor()
        for name, m in unet.named_modules():
            if name.endswith('attn2') or name.endswith('attn1'):
                cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                m.set_processor(LoRAAttnProcessor(
                    hidden_size=hidden_size, 
                    cross_attention_dim=cross_attention_dim, 
                    rank=args.mace_rank, 
                    attn_controller=attn_controller, 
                    module_name=name, 
                    preserve_prior=False,
                ))

        ### set lora
        lora_attn_procs = {}
        for key, value in zip(unet.attn_processors.keys(), unet.attn_processors.values()):
            if key.endswith("attn2.processor"):
                lora_attn_procs[f'{key}.to_k_lora'] = value.to_k_lora
                lora_attn_procs[f'{key}.to_v_lora'] = value.to_v_lora
                
        lora_layers = AttnProcsLayers(lora_attn_procs)

        # values from the original implementation
        optimizer = optimizer_class(
            lora_layers.parameters(),
            lr=args.mace_lr,
            weight_decay=0.01,
        )
        
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.mace_lr_warmup_steps,
            num_training_steps=args.mace_max_train_steps,
            num_cycles=args.mace_lr_num_cycles,
            power=args.mace_lr_power,
        )
        
        # Train
        print("Running training")
        global_step = 0
        first_epoch = 0

        if args.mace_importance_sampling:
            print("Using relation-focal importance sampling, which can make training more efficient and is particularly beneficial in erasing mass concepts with overlapping terms.")
            
            list_of_candidates = [x for x in range(noise_scheduler.config.num_train_timesteps)]
            prob_dist = [importance_sampling_fn(x) for x in list_of_candidates]
            prob_sum = 0
            # normalize the prob_list so that sum of prob is 1
            for j in prob_dist:
                prob_sum += j
            prob_dist = [x / prob_sum for x in prob_dist]
        
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, args.mace_max_train_steps))
        progress_bar.set_description("Steps")
    
        debug_once = True
                
        if args.mace_train_seperate:
            train_dataset.concept_number = i 
        
        for epoch in range(first_epoch, args.mace_num_train_epochs):
            unet.train()
                
            torch.cuda.empty_cache()
            gc.collect()
            
            for step, batch in enumerate(train_dataloader):
        
                # show
                if debug_once:
                    print('====================')
                    print(f'Concept {i}: {batch["instance_prompts"][0]}')
                    print('====================')
                    debug_once = False
                    
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                if args.mace_importance_sampling:
                    timesteps = np.random.choice(list_of_candidates, size=bsz, replace=True, p=prob_dist)
                    timesteps = torch.tensor(timesteps).to(device)
                else:
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                    
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                
                # set concept_positions for this batch 
                if args.use_gsam_mask:
                    GSAM_mask = batch['masks']
                else:
                    GSAM_mask = None
                
                attn_controller.set_concept_positions(batch["concept_positions"].to(device), GSAM_mask, use_gsam_mask=args.use_gsam_mask)

                # Predict the noise residual
                _ = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = attn_controller.loss()
                
                loss.backward()
                nn.utils.clip_grad_norm_(lora_layers.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=False)
                attn_controller.zero_attn_probs()

                # Checks if the accelerator has performed an optimization step behind the scenes
                progress_bar.update(1)
                global_step += 1
                    
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if global_step >= args.mace_max_train_steps:
                    break

        del lora_attn_procs, lora_layers, optimizer, lr_scheduler, attn_controller
        torch.cuda.empty_cache()

        if not args.mace_train_seperate:
            break
    
    unet.save_pretrained(args.save_dir)
    
def main(args: Arguments):
    args.data_dir = args.data_dir.replace(" ", "-")

    # when erasing style, the segmentation mask should not be used.
    # ref: https://github.com/Shilin-LU/MACE/issues/9
    if args.mace_concept_type == "style":
        args.use_gsam_mask = False

    # stage 0 (Preparing data)
    making_data(args)
    # stage 1 & 2 (CFR and LoRA training)
    cfr_lora_training(args)

    if os.path.isdir("mace-data"):
        shutil.rmtree("mace-data")

    if os.path.isdir("mace-data-mask"):
        shutil.rmtree("mace-data-mask")

