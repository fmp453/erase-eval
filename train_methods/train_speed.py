import re
import time
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from kmeans_pytorch import kmeans
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel

from train_methods.train_utils import seed_everything, get_devices, tokenize
from utils import Arguments

@torch.no_grad()
def get_text_embedding(
    text_encoder: CLIPTextModel,
    tokenized: torch.Tensor,
) -> torch.Tensor:
    return text_encoder(
        tokenized.input_ids.to(text_encoder.device)
    ).last_hidden_state[0]


def generate_perturbed_embs(
    ret_embs: torch.Tensor,
    P: torch.Tensor,
    erase_weight,
    num_per_sample: int,
    mini_batch: int = 8,
) -> torch.Tensor:
    ret_embs = ret_embs.squeeze(1)
    out_embs, norm_list = [], []
    for i in range(0, ret_embs.size(0), mini_batch):
        mini_ret_embs = ret_embs[i:i + mini_batch]
        for _ in range(num_per_sample):
            noise = torch.randn_like(mini_ret_embs)
            perturbed_embs = mini_ret_embs + noise @ P
            out_embs.append(perturbed_embs)
            norm_list.append(torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1))
    out_embs = torch.cat(out_embs, dim=0)
    norm_list = torch.cat(norm_list, dim=0)
    return out_embs[norm_list > norm_list.mean()].unsqueeze(1) # shape: [Num, 1, 768]


@torch.no_grad()
def train(
    args: Arguments,
    target_concepts,
    anchor_concepts,
    retain_texts,
    chunk_size=128,
    emb_size=768,
):
    device = get_devices(args)[0]
    pipeline = StableDiffusionPipeline.from_pretrained(args.sd_version).to(device)
    unet: UNet2DConditionModel = pipeline.unet
    tokenizer: CLIPTokenizer = pipeline.tokenizer
    text_encoder: CLIPTextModel = pipeline.text_encoder

    I = torch.eye(emb_size, device=device)
    if args.speed_params == 'KV':
        edit_dict = {k: v for k, v in unet.state_dict().items() if 'attn2.to_k' in k or 'attn2.to_v' in k}
    elif args.speed_params == 'V':
        edit_dict = {k: v for k, v in unet.state_dict().items() if 'attn2.to_v' in k}
    elif args.speed_params == 'K':
        edit_dict = {k: v for k, v in unet.state_dict().items() if 'attn2.to_k' in k}

    null_inputs = tokenize([''], tokenizer)
    null_hidden = get_text_embedding(text_encoder, null_inputs)
    _, cluster_centers = kmeans(X=null_hidden[1:], num_clusters=3, distance='euclidean', device=device)
    K2 = torch.cat([null_hidden[[0], :], cluster_centers.to(device)], dim=0).T
    I2 = torch.eye(len(K2.T), device=device)

    # region [Target and Anchor]
    sum_anchor_target, sum_target_target = [], []
    for i in range(0, len(target_concepts)):
        target_inputs = tokenize([target_concepts[i]], tokenizer)
        target_embs = get_text_embedding(text_encoder, target_inputs)
        anchor_inputs = tokenize([anchor_concepts[i]], tokenizer)
        anchor_embs = get_text_embedding(text_encoder, anchor_inputs)
        if target_concepts == ['nudity']:
            target_embs = target_embs[1:, :]  # all tokens
            anchor_embs = anchor_embs[1:, :]  # all tokens
        else:
            target_embs = target_embs[[(target_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token
            anchor_embs = anchor_embs[[(anchor_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token
        sum_target_target.append(target_embs.T @ target_embs)
        sum_anchor_target.append(anchor_embs.T @ target_embs)
    sum_target_target, sum_anchor_target = torch.stack(sum_target_target).mean(0), torch.stack(sum_anchor_target).mean(0)
    # endregion

    # region [Retain]
    last_ret_embs = []
    retain_texts = [text for text in retain_texts if not any(re.search(r'\b' + re.escape(concept.lower()) + r'\b', text.lower()) for concept in target_concepts)]
    assert len(retain_texts) + len(target_concepts) == len(set(retain_texts + target_concepts))
    for j in range(0, len(retain_texts), chunk_size):
        ret_inputs = tokenize([retain_texts[j:j + chunk_size]], tokenizer)
        ret_embs = get_text_embedding(text_encoder, ret_inputs)
        if retain_texts == ['']: 
            last_ret_embs.append(ret_embs[:, 1:, :].permute(1, 0, 2))
        else:
            last_subject_indices = ret_inputs.attention_mask.sum(1) - 2
            last_ret_embs.append(ret_embs[torch.arange(ret_embs.size(0)), last_subject_indices].unsqueeze(1))
    last_ret_embs = torch.cat(last_ret_embs)
    last_ret_embs = last_ret_embs[torch.randperm(last_ret_embs.size(0))]  # shuffle
    # endregion

    for (layer_name, layer_weight) in tqdm(edit_dict.items(), desc="Model Editing"):

        erase_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ (I + sum_target_target).inverse()
        _, _, V0 = torch.svd(layer_weight)
        P0_min = V0[:, -1:] @ V0[:, -1:].T

        if args.speed_aug_num > 0 and not args.speed_disable_filter:
            weight_norm_init = torch.matmul(last_ret_embs.squeeze(1), erase_weight.T).norm(dim=1)
            layer_ret_embs = last_ret_embs[weight_norm_init > weight_norm_init.mean()]
        else:
            layer_ret_embs = last_ret_embs

        sum_ret_ret, valid_num = [], 0
        for j in range(0, len(layer_ret_embs), chunk_size):
            chunk_ret_embs = layer_ret_embs[j:j + chunk_size]
            if args.speed_aug_num > 0:
                chunk_ret_embs = torch.cat(
                    [
                        chunk_ret_embs, 
                        generate_perturbed_embs(
                            chunk_ret_embs,
                            P0_min,
                            erase_weight,
                            num_per_sample=args.speed_aug_num
                        )
                    ],
                    dim=0
                )
            valid_num += chunk_ret_embs.shape[0]
            sum_ret_ret.append((chunk_ret_embs.transpose(1, 2) @ chunk_ret_embs).sum(0))
        sum_ret_ret = torch.stack(sum_ret_ret, dim=0).sum(0) / valid_num

        U, S, _ = torch.svd(sum_ret_ret)
        P = U[:, S < args.speed_threshold] @ U[:, S < args.speed_threshold].T
        M = (sum_target_target @ P + args.speed_retain_scale * I).inverse()
        delta_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ P @ (I - M @ K2 @ (K2.T @ P @ M @ K2 + args.speed_lamb * I2).inverse() @ K2.T @ P) @ M

        # Save edited weights
        edit_dict[layer_name] = layer_weight + delta_weight

    print(f"Current model status: Edited {str(target_concepts)} into {str(anchor_concepts)}")
    return edit_dict


def main(args: Arguments):
    seed_everything(args.seed)

    target_concepts = [con.strip() for con in args.concepts.split(',')]
    anchor_concepts = args.anchor_concept

    # The filename only displays the first 5 target concepts in multi-concept erasure
    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}"
    anchor_concepts = [x.strip() for x in anchor_concepts.split(',')]
    if len(anchor_concepts) == 1:
        anchor_concepts = anchor_concepts * len(target_concepts)
        if anchor_concepts[0] == "":
            file_suffix += '-to_null'
        else:
            file_suffix += f'-to_{anchor_concepts[0]}'
    else:
        assert len(target_concepts) == len(anchor_concepts)
        file_suffix += f'-to_{anchor_concepts[0]}_etc'

    retain_texts = []
    if args.speed_retain_path is not None:
        assert args.speed_retain_path.endswith('.csv')
        df = pd.read_csv(args.speed_retain_path)
        for head in args.speed_heads.split(','):
            retain_texts += df[head.strip()].unique().tolist()
    else:
        retain_texts.append("")

    edit_dict = train(
        args=args,
        target_concepts=target_concepts, 
        anchor_concepts=anchor_concepts, 
        retain_texts=retain_texts, 
    )

    save_path = Path(args.save_dir)
    file_name = args.speed_file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{file_suffix}"
    save_path.mkdir(exist_ok=True)
    torch.save(edit_dict, save_path / f"{file_name}.pt")
