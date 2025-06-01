import os
from typing import Literal

import pandas as pd
import safetensors
import torch
from diffusers import LMSDiscreteScheduler
from PIL import Image
from safetensors.torch import load_file

# from src.eval.evaluation.eval_util import clip_score, create_meta_json
# import train_util

from train_methods.train_ace import ACELayer, ACENetwork
from train_methods.train_utils import get_models, get_devices, tokenize, get_condition
from utils import Arguments

RES = [8, 16, 32, 64]
MATCHING_METRICS = Literal[
    "clipcos",
    "clipcos_tokenuni",
    "tokenuni",
]

def calculate_matching_score(
    prompt_tokens: list[torch.Tensor],
    prompt_embeds: torch.Tensor,
    erased_prompt_tokens: list[torch.Tensor],
    erased_prompt_embeds: torch.Tensor,
    matching_metric: MATCHING_METRICS,
    special_token_ids: set,
):
    scores = []
    if "clipcos" in matching_metric:
        clipcos = torch.cosine_similarity(
            prompt_embeds.flatten(1, 2),
            erased_prompt_embeds.flatten(1, 2),
            dim=-1).cpu()
        scores.append(clipcos)
    if "tokenuni" in matching_metric:
        prompt_set = set(prompt_tokens[0].tolist()) - special_token_ids
        tokenuni = []
        for ep in erased_prompt_tokens:
            ep_set = set(ep.tolist()) - special_token_ids
            tokenuni.append(len(prompt_set.intersection(ep_set)) / len(ep_set))
        scores.append(torch.tensor(tokenuni).to("cpu"))
    return torch.max(torch.stack(scores), dim=0)[0]

def get_images(latents, vae):
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def save_images(pil_images,
                folder_path,
                case_number,
                concept):
    attn_score_erased = None
    attn_score_list = []
    for num, im in enumerate(pil_images):
        os.makedirs(f"{folder_path}/{concept}", exist_ok=True)
        im.save(f"{folder_path}/{concept}/{case_number}_{num}.png")
    return attn_score_list, attn_score_erased


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if isinstance(sd[key], torch.Tensor):
            sd[key] = sd[key].to(dtype)

    return sd, metadata

def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata

@torch.no_grad()
def generate_images(
    args: Arguments,
    # model_name,
    # prompts_path,
    # save_path,
    # from_case=0,
    # aligned_multipliers=None,
    # need_mid_image=False,
    # check_rate=0.5,
    # erased_concept=None,
    # lora_rank=4,
    # matching_metric: MATCHING_METRICS = "clipcos_tokenuni",
    # edit_concept_path=None,
    # specific_concept=None,
    # specific_concept_set=None,
    # is_Mace=False,
    # lora_path=None,
    # cab_path=None,
    # is_specific=True,
    # lora_name=None,
    # tensor_name=None,
    # test_unet=False,
    # model_path="CompVis/stable-diffusion-v1-4",
    # is_textencoder=False,
    # model_weight_path=None
):

    weight_dtype = torch.float32

    tokenizer, text_encoder, vae, unet, _, _ = get_models(args)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))
    device = get_devices(args)[0]

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    if is_specific:
        specific_concept_set = set().add(specific_concept)
    print(specific_concept_set)

    spm_paths = [lp for lp in args.erased_model_dir.split(",")]
    used_multipliers = []
    network = ACENetwork(
        unet,
        rank=lora_rank,
        alpha=1.0,
        module=ACELayer,
    ).to(device, dtype=weight_dtype)
    spms, metadatas = zip(*[
        load_state_dict(spm_model_path, weight_dtype) for spm_model_path in spm_paths
    ])

    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    erased_prompts_count = [len(ep) for ep in erased_prompts]
    erased_prompts_flatten = [item for sublist in erased_prompts for item in sublist]
    erased_prompt_embeds, erased_prompt_tokens = train_util.encode_prompts(
        tokenizer, text_encoder, erased_prompts_flatten, return_tokens=True
    )
    
    name_path = ''
    concept_set = set()
    for name in model_name:
        if name_path == '':
            name_path = name_path + name
        else:
            name_path = name_path + '-' + name

    print(f"erased_prompts is {erased_prompts}")
    edit_concept_set = set()
    if edit_concept_path != None:
        with open(edit_concept_path, "r") as f:
            for line in f:
                edit_concept_set.add(line.strip())
    else:
        edit_concept_set.add(None)
    
    for edit_concept in edit_concept_set:
        folder_path = f'{save_path}/{name_path}_edit_{edit_concept}/{tensor_name}'
        os.makedirs(folder_path, exist_ok=True)
        
        prompt = [args.prompt] * args.num_images_per_prompt

        seed = args.seed
        weighted_spm = dict.fromkeys(spms[0].keys())
        prompt_embeds, prompt_tokens = train_util.encode_prompts(
            tokenizer, text_encoder, [prompt], return_tokens=True
        )
        multipliers = calculate_matching_score(
            prompt_tokens,
            prompt_embeds,
            erased_prompt_tokens,
            erased_prompt_embeds,
            matching_metric=args.matching_metric,
            special_token_ids=special_token_ids,
            weight_dtype=weight_dtype
        )
        multipliers = torch.split(multipliers, erased_prompts_count)
        spm_multipliers = torch.tensor(multipliers).to("cpu", dtype=weight_dtype)
        for spm, multiplier in zip(spms, spm_multipliers):
            max_multiplier = torch.max(multiplier)
            for key, value in spm.items():
                if weighted_spm[key] is None:
                    weighted_spm[key] = value * max_multiplier
                else:
                    weighted_spm[key] += value * max_multiplier
            used_multipliers.append(max_multiplier.item())
        network.load_state_dict(weighted_spm)

        concept = args.concepts
        concept_set.add(concept)
        
        height = args.image_size
        width = args.image_size
        num_inference_steps = args.ddim_steps
        guidance_scale = args.guidance_scale
        generator = torch.manual_seed(seed)  # Seed generator to create the inital latent noise
        batch_size = len(prompt)
        text_embeddings = get_condition(prompt, tokenizer, text_encoder)
        uncond_embeddings = get_condition([""], tokenizer, text_encoder)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        latents = torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(unet.device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)
        step = 0
        pbar = tqdm(scheduler.timesteps)
        for t in pbar:
            pbar.set_postfix({"guidance": guidance_scale})
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                with network:
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            # print(f"latents size is {latents.shape}")
            step += 1
            if step == int(check_rate * num_inference_steps) and need_mid_image:
                pil_mid_images = get_images(latents, vae)
                folder_mid_path = os.path.join(folder_path, f"mid_{check_rate}")
                os.makedirs(folder_mid_path, exist_ok=True)
                save_images(pil_images=pil_mid_images,
                            folder_path=folder_mid_path,
                            case_number=case_number,
                            concept=concept)

                del pil_mid_images
        pil_images = get_images(latents, vae)
        save_images(pil_images=pil_images,
                    folder_path=folder_path,
                    case_number=case_number,
                    concept=concept)

        del latents

def main(args):
    model_name = args.model_name
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    from_case = args.from_case

    test_unet = args.test_unet
    need_mid_image = args.need_mid_image
    check_rate = args.check_rate
    erased_concept = args.erased_concept
    lora_rank = args.lora_rank
    edit_concept_path = args.edit_concept_path
    specific_concept = args.specific_concept
    specific_concept_path = args.specific_concept_path
    is_Mace = args.is_Mace
    is_specific = args.is_specific
    lora_path = args.lora_path
    model_path = args.model_path
    model_concept_path = args.model_concept_path
    lora_name = args.lora_name
    tensor_name = args.tensor_name
    is_coco = args.is_coco
    is_SD = args.is_SD
    is_text_encoder = args.is_text_encoder
    model_weight_path = args.model_weight_path
    specific_concept_set = set()
    generate_concept_set = set()
    generate_concept_path = args.generate_concept_path
    if specific_concept_path is not None:
        with open(specific_concept_path, "r") as concepts:
            for concept in concepts:
                specific_concept_set.add(concept.strip())
    else:
        specific_concept_set.add(None)
    if generate_concept_path is None:
        generate_concept_set = specific_concept_set
    else:
        with open(generate_concept_path, "r") as concepts:
            for concept in concepts:
                generate_concept_set.add(concept.strip())
    if model_concept_path is not None:
        df_model = pd.read_csv(model_concept_path)
        model_name_dict = {}
        for _, row in df_model.iterrows():
            if row.concept in specific_concept_set:
                model_name_dict[row.concept] = row.prompt
    else:
        model_name_dict = None
    for concept in specific_concept_set:
        lora_path_tem = None
        lora_name_tem = None
        tensor_name_tem = None
        if is_lora:
            model_name_tem = model_name[0].format(concept)
            if model_concept_path is None:
                if lora_path is None:
                    lora_name_tem = lora_name.format(concept)
                    if tensor_name is not None:
                        tensor_name_tem = tensor_name.format(concept)
                else:
                    lora_path_tem = lora_path.format(concept, concept, concept)
                print(lora_path_tem, lora_name_tem)
            else:
                model_prompt = model_name_dict[concept]
                if lora_path is None:
                    lora_name_tem = lora_name.format(concept)
                    if tensor_name is not None:
                        tensor_name_tem = tensor_name.format(concept)
                else:
                    lora_path_tem = lora_path.format(model_prompt, model_prompt, model_prompt)
        elif model_concept_path is not None:
            model_name_tem = model_name[0].format(model_name_dict[concept])
        elif is_SD:
            model_name_tem = model_name[0]
        else:
            model_name_tem = model_name[0].format(concept)
        generate_images([model_name_tem],
                        prompts_path,
                        save_path,
                        device=device,
                        guidance_scale=guidance_scale,
                        image_size=image_size,
                        ddim_steps=ddim_steps,
                        num_samples=num_samples,
                        from_case=from_case,
                        is_lora=is_lora,
                        aligned_multipliers=args.multipliers,
                        need_mid_image=need_mid_image,
                        check_rate=check_rate,
                        erased_concept=erased_concept,
                        lora_rank=lora_rank,
                        edit_concept_path=edit_concept_path,
                        specific_concept=specific_concept,
                        lora_path=lora_path_tem,
                        specific_concept_set=generate_concept_set,
                        is_Mace=is_Mace,
                        is_specific=is_specific,
                        lora_name=lora_name_tem,
                        tensor_name=tensor_name_tem,
                        is_coco=is_coco,
                        test_unet=test_unet,
                        model_path=model_path,
                        is_textencoder=is_text_encoder,
                        model_weight_path=model_weight_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', nargs='+', type=str, required=False)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=False)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--from_ckpt', action='store_true', help='whether get pretrained model from ckpt 500',
                        required=False, default=False)
    parser.add_argument('--need_attention_map', action='store_true', help='whether obtain maps of generated images',
                        required=False, default=False)
    parser.add_argument('--need_mid_image', action='store_true', help='whether obtain mid of generated images',
                        required=False, default=False)
    parser.add_argument('--is_specific', action='store_true',
                        required=False, default=False)
    parser.add_argument('--test_unet', action='store_true',
                        required=False, default=False)
    parser.add_argument('--check_rate', help='check rate of generate images', type=float, required=False, default=0.5)
    parser.add_argument('--multipliers', help='coefficient of spm', nargs='*', type=float, required=False)
    parser.add_argument('--erased_concept', nargs='*', type=str, required=False)
    parser.add_argument('--lora_rank', help='lora rank of model used to train', type=int, required=False, default=4)
    parser.add_argument('--edit_concept_path', type=str, required=False, default=None)
    parser.add_argument('--specific_concept', type=str, required=False, default=None)
    parser.add_argument('--specific_concept_path', type=str, required=False, default=None)
    parser.add_argument('--fuse_lora_config_path', type=str, required=False, default=None)
    parser.add_argument('--lora_path', type=str, required=False, default=None)
    parser.add_argument('--lora_name', type=str, required=False, default=None)
    parser.add_argument('--model_concept_path', type=str, required=False, default=None)
    parser.add_argument('--cab_path', type=str, required=False)
    parser.add_argument('--model_weight_path', type=str, required=False)
    parser.add_argument('--tensor_name', type=str, required=False)
    parser.add_argument('--generate_concept_path', type=str, required=False)
    args = parser.parse_args()
    main(args)
