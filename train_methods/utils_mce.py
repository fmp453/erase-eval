import time
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Subset

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from train_methods.mce_models.hooks import (
    CrossAttentionExtractionHook,
    LinearLayerHooker,
    NormHooker,
    FeedForwardHooker,
)

if TYPE_CHECKING:
    from diffusers import FluxPipeline
    from open_clip.model import CLIP
    from open_clip.tokenizer import HFTokenizer
    from torchvision.transforms import Compose

    from utils import Arguments

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def scale_noise(
    scheduler,
    sample: torch.Tensor,
    timestep: float | torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.Tensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.Tensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample

@torch.no_grad()
def calc_v_flux(
    pipe: FluxPipeline,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: float,
    text_ids,
    latent_image_ids,
    t: torch.Tensor
) -> torch.Tensor:
    timestep = t.expand(latents.shape[0])
    noise_pred = pipe.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]

    return noise_pred

@torch.no_grad()
def FlowEditFLUX(
    pipe: FluxPipeline,
    scheduler,
    x_src: torch.Tensor,
    src_prompt,
    tar_prompt,
    negative_prompt="",
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
):
    device = x_src.device
    orig_height, orig_width = x_src.shape[2] * pipe.vae_scale_factor // 2, x_src.shape[3] * pipe.vae_scale_factor // 2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src.dtype,
        device=x_src.device,
        generator=None,
        latents=x_src,
    )
    x_src_packed: torch.Tensor = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )

    pipe._num_timesteps = len(timesteps)

    # src prompts
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src_packed.clone()

    for i, t in enumerate(timesteps):
        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)

                zt_src = (1 - t_i) * x_src_packed + (t_i) * fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                # Merge in the future to avoid double computation
                Vt_src = calc_v_flux(
                    pipe,
                    latents=zt_src,
                    prompt_embeds=src_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids=latent_src_image_ids,
                    t=t,
                )

                Vt_tar = calc_v_flux(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=latent_tar_image_ids,
                    t=t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)  # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:  # i >= T_steps-n_min # regular sampling last n_min steps

            if i == T_steps - n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            Vt_tar = calc_v_flux(
                pipe,
                latents=xt_tar,
                prompt_embeds=tar_prompt_embeds,
                pooled_prompt_embeds=tar_pooled_prompt_embeds,
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids=latent_tar_image_ids,
                t=t,
            )

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out, out

def get_all_concept_template_with_synonyms(concept):
    from train_methods.templates import SYNONYMS_DICT
    syn_with_concept = []
    for k, v in SYNONYMS_DICT.items():
        if k in concept:
            concept_with_v = [concept.replace(k, v) for v in v]
            syn_with_concept = [concept] + concept_with_v
            break

    if len(syn_with_concept) == 0:
        raise ValueError(f"concept {concept} not found in SYNONYMS_DICT")
    else:
        return syn_with_concept


@torch.no_grad()
def calculate_clip_score(clip_dict, image, text, device, single_word=False, negative_word=None, keepmodel=False):
    SIMPLE_IMAGENET_TEMPLATES = (
        lambda c: f"itap of a {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"art of the {c}.",
        lambda c: f"a photo of the small {c}.",
    )

    clip_model: CLIP = clip_dict["clip_model"]
    transform: Compose = clip_dict["transform"]
    tokenizer: HFTokenizer = clip_dict["tokenizer"]

    clip_model = clip_model.to(device)
    clip_model.eval()

    if isinstance(image, list):
        raise ValueError("Image should be a single image path or a PIL image")

    # preprocess image and text
    if isinstance(image, str):
        image: torch.Tensor = transform(Image.open(image)).unsqueeze(0).to(device)
    else:
        import pdb

        pdb.set_trace()
        image = transform(image).unsqueeze(0).to(device)

    # calculate CLIP score
    if single_word:
        prompt_list = [t(text) for t in SIMPLE_IMAGENET_TEMPLATES]
        text = tokenizer(prompt_list).to(device)
        text_embed = clip_model.encode_text(text)
        text_embed = text_embed.reshape(len(SIMPLE_IMAGENET_TEMPLATES), -1).mean(dim=0)
        text_embed /= text_embed.norm(dim=-1, keepdim=True)

        if negative_word:
            prompt_list = [t(negative_word) for t in SIMPLE_IMAGENET_TEMPLATES]
            negative_text = tokenizer(prompt_list).to(device)
            negative_text_embed = clip_model.encode_text(negative_text)
            negative_text_embed = negative_text_embed.reshape(len(SIMPLE_IMAGENET_TEMPLATES), -1).mean(dim=0)
            negative_text_embed /= negative_text_embed.norm(dim=-1, keepdim=True)
    else:
        text = tokenizer(text).to(device)
        text_embed = clip_model.encode_text(text)
        text_embed /= text_embed.norm(dim=-1, keepdim=True)

    image_embed = clip_model.encode_image(image)
    image_embed /= image_embed.norm(dim=-1, keepdim=True)

    score = image_embed @ text_embed.t()

    if negative_word:
        neg_score = image_embed @ negative_text_embed.t()
        # normal score with negative score
        score = F.softmax(torch.cat([score, neg_score], dim=-1), dim=-1)[0]
    score = score.cpu().detach().item()

    # delete the unused tensor
    # release GPU memory
    del image, text, text_embed, image_embed
    if not keepmodel:
        del clip_model
    return score


def foreground_score_calculation(d, args: Arguments, clip_dict, device):
    from train_methods.templates import CON_DECON_DICT, SYNONYMS_DICT
    # get adjective synonyms
    update_con_decon_dict = CON_DECON_DICT.copy()
    # get all concept adj list, e.g. gun, nude and etc.
    concept_adj_list = [*SYNONYMS_DICT.keys()]

    for k, v in CON_DECON_DICT.items():
        for adj in concept_adj_list:
            if adj in k:
                # update con_decon_dict with new key and value
                synon_adj_list = SYNONYMS_DICT[adj]
                for synon_adj in synon_adj_list:
                    new_key = k.replace(adj, synon_adj)
                    update_con_decon_dict[new_key] = v

    img_path = d["path"]
    # use only the concepts for evaluation, e.g. a photo of <concept>
    foreground_prompt = args.concepts
    foreground_promptwnconcept = update_con_decon_dict[args.concepts]

    score = calculate_clip_score(clip_dict, img_path, foreground_prompt, device, single_word=True)
    score_wnconcept = calculate_clip_score(
        clip_dict,
        img_path,
        foreground_promptwnconcept,
        device,
        single_word=True,
    )
    # (negative score mean the image is not related to the concept)
    score = score - score_wnconcept
    # use score as penalty (1 meanes highest penalty)
    score = 1 if score < 0 else score
    return score


def background_score_calculation(d, args: Arguments, clip_dict, device):
    img_path = d["path"]
    prompt = d["prompt"]
    # get the prompt w/o concept
    if args.mce_with_synonyms:
        concept_with_synoyms = get_all_concept_template_with_synonyms(args.concepts)
        for c in concept_with_synoyms:
            if c in prompt:
                background = prompt.replace(c, "")
                break
    else:
        background = prompt.replace(args.concepts, "")
    score_wconcept = calculate_clip_score(clip_dict, img_path, background, device)

    # replace base name
    basename = "deconcept_" + Path(img_path).name
    img_path = Path(Path(img_path).parent, basename)
    score_woconcept = calculate_clip_score(clip_dict, img_path, background, device)
    return abs(score_wconcept - score_woconcept)

def dataset_filter(dataset: Dataset, args: Arguments, device: torch.device) -> tuple[Subset, int]:
    # clearn cache to avoid OOM
    torch.cuda.empty_cache()

    if isinstance(args.mce_filter_ratio * args.mce_size, float):
        raise ValueError("filter_ratio * data.size must be an integer, change the filter_ratio or data.size")

    # prepare CLIP encoder
    model: CLIP
    model, transform, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="datacomp_xl_s13b_b90k"
    )

    clip_dict = {
        "clip_model": model,
        "preprocess": preprocess,
        "transform": transform,
        "tokenizer": open_clip.get_tokenizer(model_name="ViT-B-16"),
        "logit_scale": model.logit_scale,
        "clip_config": open_clip.get_model_config(model_name="ViT-B-16")
    }

    filter_score_list = []
    for idx in range(len(dataset)):
        d = dataset[idx]
        value = d["value"]
        # calculate filter score for image w concepts
        # calculate background score and foreground score (lower the better)
        if value:
            background_score = background_score_calculation(d, args, clip_dict, device)
            if args.mce_with_fg_filter:
                foreground_score = foreground_score_calculation(d, args, clip_dict, device)
            else:
                foreground_score = 0
            score = background_score + foreground_score
            filter_score_list.append(score)
        else:
            filter_score_list.append(0)

    pos = [1 for _ in range(len(dataset))]
    # keep remove the lowest similarity score
    filter_score_list = torch.tensor(filter_score_list)
    threshold = torch.quantile(filter_score_list, args.mce_filter_ratio)

    for idx, score in enumerate(filter_score_list):
        if score > threshold:
            pos[idx] = 0
            pos[idx + 1] = 0

    subset_pos = [i for i, p in enumerate(pos) if p == 1]
    print(f"filter out {len(dataset) - len(subset_pos)} images")
    dataset = Subset(dataset, subset_pos)
    # release GPU memory from clip model
    del clip_dict
    return dataset, len(subset_pos)


def init_hooker(
    args: Arguments,
    pipe,
    project_folder
) -> tuple[list[CrossAttentionExtractionHook | FeedForwardHooker | LinearLayerHooker | NormHooker], list[float]]:
    """
    Initialize hookers for training
    return:
        hookers: list of hookers
        hookers_tuple: list of tuples of hookers and their names
        lr_list: list of learning rates
    """
    if any(model in args.mce_model for model in ["flux", "sd3"]):
        # TODO: temporary solusion, fix this in the future
        if args.mce_model == "sd3":
            args.mce_n_lr = 0
        return init_attn_ffn_norm_hooker(args, pipe, project_folder)
    else:
        return init_linear_hooker(args, pipe, project_folder)



def init_attn_ffn_norm_hooker(
    args: Arguments, 
    pipe, 
    project_folder
) -> tuple[list[CrossAttentionExtractionHook | FeedForwardHooker | NormHooker], list[float]]:
    """
    Initialize cross attention, feedforward, and norm hookers for training
    return:
        hookers: list of hookers
        hookers_tuple: list of tuples of hookers and their names
        lr_list: list of learning rates
    """
    cross_attn_hooker = CrossAttentionExtractionHook(
        pipe,
        regex=args.mce_regex,
        head_num_filter=args.mce_head_num_filter,
        masking=args.mce_masking,
        dst=Path(project_folder, "attn"),
        epsilon=args.mce_epsilon,
        model_name=args.mce_model,
        attn_name=args.mce_attn_name,
        use_log=False,
        eps=args.mce_masking_eps,
    )
    cross_attn_hooker.add_hooks(init_value=args.mce_init_lambda)

    # initialize feedforward hooks
    ff_hooker = FeedForwardHooker(
        pipe,
        regex=args.mce_regex,
        masking=args.mce_masking,
        dst=Path(project_folder, "ffn"),
        epsilon=args.mce_epsilon,
        eps=args.mce_masking_eps,
        use_log=False,
    )
    ff_hooker.add_hooks(init_value=args.mce_init_lambda)
    hookers = [cross_attn_hooker, ff_hooker]
    assert isinstance(args.mce_attn_lr, float)
    assert isinstance(args.mce_ff_lr, float)
    assert isinstance(args.mce_n_lr, float)
    lr_list = [args.mce_attn_lr, args.mce_ff_lr]

    # initialize norm hooks if lr is not 0
    if args.mce_n_lr != 0:
        norm_hooker = NormHooker(
            pipe,
            regex=args.mce_regex,
            masking=args.mce_masking,
            dst=Path(project_folder, "norm"),
            epsilon=args.mce_epsilon,
            eps=args.mce_masking_eps,
            use_log=False,
        )
        norm_hooker.add_hooks(init_value=args.mce_init_lambda)
        hookers.append(norm_hooker)
        lr_list.append(args.mce_n_lr)
    return hookers, lr_list


def init_linear_hooker(
    args: Arguments, pipe, project_folder
) -> tuple[list[LinearLayerHooker], list[float]]:
    """
    Initialize linear hooker for training
    return:
        hookers: list of hookers
        hookers_tuple: list of tuples of hookers and their names
        lr_list: list of learning rates
    """
    linear_hooker = LinearLayerHooker(
        pipe,
        regex=args.mce_regex,
        masking=args.mce_masking,
        dst=Path(project_folder, "ffn"),
        epsilon=args.mce_epsilon,
        eps=args.mce_masking_eps,
        use_log=False,
    )
    hookers = [linear_hooker]
    linear_hooker.add_hooks(init_value=args.mce_init_lambda)
    assert isinstance(args.mce_ff_lr, float)
    lr_list = [args.mce_ff_lr]
    return hookers, lr_list


def get_file_name(save_dir: str, prompt: str = None, seed: int = 44):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if len(prompt) > 30:
        prompt = prompt[:30].replace(" ", "_")
    name = f"{prompt}_seed_{seed}_{timestr}.png"
    return Path(save_dir, name)

@torch.no_grad()
def save_image_seed(
    pipe,
    prompts: str,
    steps: int,
    device: torch.device,
    seed: int,
    save_dir=None,
    save_path=None,
    width=None,
    height=None,
    output_type="pil",
    hookers: list | None = None,
):
    assert hookers is None, "hookers is not required for this function"
    g_cpu = torch.Generator(device).manual_seed(seed)
    image = pipe(
        prompts, generator=g_cpu, num_inference_steps=steps, width=width, height=height, output_type=output_type
    )
    image: dict[str, list[Image.Image]]

    if save_path is not None:
        image["images"][0].save(save_path)
        return

    if save_dir is None:
        return image["images"]
    else:
        if isinstance(prompts, str):
            prompts = [prompts]
        for img, prompt in zip(image["images"], prompts):
            name = get_file_name(save_dir, prompt=prompt, seed=seed)
            Path(save_dir).mkdir(exist_ok=True)
            img.save(name)
        return None
