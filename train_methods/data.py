import os
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

from train_methods.train_utils import prompt_augmentation

class MACEDataset(Dataset):
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        size: int=512,
        multi_concept=None,
        mapping=None,
        batch_size=None,
        train_seperate=False,
        aug_length: int=30,
        prompt_len: int=30,
        input_data_path=None
    ):  
        self.size = size
        self.tokenizer = tokenizer
        self.batch_counter = 0
        self.batch_size = batch_size
        self.concept_number = 0
        self.train_seperate = train_seperate
        self.aug_length = aug_length
        
        self.all_concept_image_path  = []
        self.all_concept_mask_path  = []
        single_concept_images_path = []
        self.instance_prompt  = []
        self.target_prompt  = []
        
        self.num_instance_images = 0
        self.dict_for_close_form = []
        self.class_images_path = []
        
        for concept_idx, (data, mapping_concept) in enumerate(zip(multi_concept, mapping)):
            c, t = data
            
            if input_data_path is not None:
                p = Path(os.path.join(input_data_path, c.replace("-", " ")).replace(" ", "-"))
                if not p.exists():
                    raise ValueError(f"Instance {p} images root doesn't exists.")
                
                if t == "object":
                    p_mask = Path(os.path.join(input_data_path, c.replace("-", " ")).replace(f'{c.replace("-", " ")}', f'{c.replace("-", " ")}-mask').replace(" ", "-"))
                    if not p_mask.exists():
                        raise ValueError(f"Instance {p_mask} images root doesn't exists.")
            else:
                raise ValueError(f"Input data path is not provided.")    
            
            image_paths = list(p.iterdir())
            single_concept_images_path = []
            single_concept_images_path += image_paths
            self.all_concept_image_path.append(single_concept_images_path)
            
            if t == "object":
                mask_paths = list(p_mask.iterdir())
                single_concept_masks_path = []
                single_concept_masks_path += mask_paths
                self.all_concept_mask_path.append(single_concept_masks_path)
                     
            erased_concept = c.replace("-", " ")
            
            sampled_indices = random.sample(range(0, prompt_len), self.aug_length)
            self.instance_prompt.append(prompt_augmentation(erased_concept, augment=True, sampled_indices=sampled_indices, concept_type=t))
            self.target_prompt.append(prompt_augmentation(mapping_concept, augment=True, sampled_indices=sampled_indices, concept_type=t))
                
            self.num_instance_images += len(single_concept_images_path)
            
            entry = {"old": self.instance_prompt[concept_idx], "new": self.target_prompt[concept_idx]}
            self.dict_for_close_form.append(entry)
                         
        self.image_transforms = transforms.Compose(
            [
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self._concept_num = len(self.instance_prompt)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_instance_images // self._concept_num, self.num_class_images)
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        if not self.train_seperate:
            if self.batch_counter % self.batch_size == 0:
                self.concept_number = random.randint(0, self._concept_num - 1)
            self.batch_counter += 1
        
        instance_image = Image.open(self.all_concept_image_path[self.concept_number][index % self._length])
        
        if len(self.all_concept_mask_path) == 0:
            # artistic style erasure
            binary_tensor = None
        else:
            # object/celebrity erasure
            instance_mask = Image.open(self.all_concept_mask_path[self.concept_number][index % self._length])
            instance_mask = instance_mask.convert('L')
            trans = transforms.ToTensor()
            binary_tensor = trans(instance_mask)
        
        prompt_number = random.randint(0, len(self.instance_prompt[self.concept_number]) - 1)
        instance_prompt, target_tokens = self.instance_prompt[self.concept_number][prompt_number]
        
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_prompt"] = instance_prompt
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_masks"] = binary_tensor

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        prompt_ids = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length
        ).input_ids

        concept_ids = self.tokenizer(target_tokens, add_special_tokens=False).input_ids

        pooler_token_id = self.tokenizer("<|endoftext|>", add_special_tokens=False).input_ids[0]

        concept_positions = [0] * self.tokenizer.model_max_length
        for i, tok_id in enumerate(prompt_ids):
            if tok_id == concept_ids[0] and prompt_ids[i:i + len(concept_ids)] == concept_ids:
                concept_positions[i:i + len(concept_ids)] = [1]*len(concept_ids)
            if tok_id == pooler_token_id:
                concept_positions[i] = 1
        example["concept_positions"] = torch.tensor(concept_positions)[None]               
    
        return example

class AblatingConceptDataset(Dataset):
    # ref: https://huggingface.co/spaces/nupurkmr9/concept-ablation/blob/main/concept-ablation-diffusers/utils.py
    def __init__(self, concept_type, image_dir, prompt_path, tokenizer, concept, anchor_concept=None, size=512, hflip=False, aug=True):
        
        self.size = size
        self.tokenizer = tokenizer
        self.interpolation = Image.Resampling.BILINEAR
        self.aug = aug
        self.concept_type = concept_type
        self.concept = concept
        self.anchor_concept = anchor_concept

        self.instance_images_path = []
        self.class_images_path = []
        inst_images_path = []
        for i, j in product(range(200), range(5)):
            inst_images_path.append(f"{image_dir}/{i:03}-{j}.png")
        inst_prompt: list[str] = pd.read_csv(prompt_path)["prompt"].to_list()
        inst_prompt = [x.lower() for x in inst_prompt]
        
        # caption_target : prompt
        # class_prompt or instance prompt: anchor prompt
        for i, j in product(range(200), range(5)):
            self.instance_images_path.append((inst_images_path[i * 5 + j], inst_prompt[i], self.concept))

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose([
            self.flip,
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def preprocess(self, image: np.ndarray, scale, resample):
        outer, inner = self.size, scale
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // 8, self.size // 8))
        if scale > self.size:
            instance_image = image[top: top + inner, left: left + inner, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            instance_image[top: top + inner, left: left + inner, :] = image
            mask[top // 8 + 1: (top + scale) // 8 - 1, left // 8 + 1: (left + scale) // 8 - 1] = 1.
        return instance_image, mask

    def __getprompt__(self, instance_prompt: str, instance_target):
        if self.concept_type == 'style':
            r = np.random.choice([0, 1, 2])
            instance_prompt = f'{instance_prompt}, in the style of {instance_target}' if r == 0 else f'in {instance_target}\'s style, {instance_prompt}' if r == 1 else f'in {instance_target}\'s style, {instance_prompt}'
        elif self.concept_type == 'object':
            # cat+grumpy cat
            instance_prompt = instance_prompt.replace(self.anchor_concept, self.concept)
        return instance_prompt

    def __getitem__(self, index):
        instance_image, instance_prompt, instance_target = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)
        # modify instance prompt according to the concept_type to include target concept
        # multiple style/object fine-tuning
        if ';' in instance_target:
            instance_target = instance_target.split(';')
            instance_target = instance_target[index % len(instance_target)]

        instance_anchor_prompt = instance_prompt
        instance_prompt = self.__getprompt__(instance_prompt, instance_target)
        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = np.random.randint(self.size // 3, self.size + 1) if np.random.uniform() < 0.66 else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example = {}
        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["instance_anchor_prompt_ids"] = self.tokenizer(
            instance_anchor_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example


class DocoDataset(Dataset):
    # 多分上と同じ
    def __init__(
        self,
        concepts_list: list[str],
        concept_type: str,
        tokenizer: CLIPTokenizer,
        size: int=512,
        center_crop=False,
        hflip=False,
        aug=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.Resampling.BILINEAR
        self.aug = aug
        self.concept_type = concept_type

        self.instance_images_path = []
        self.class_images_path = []
        for concept in concepts_list:
            with open(concept["instance_data_dir"], "r") as f:
                inst_images_path = f.read().splitlines()
            with open(concept["instance_prompt"], "r") as f:
                inst_prompt = f.read().splitlines()
            inst_img_path = [
                (x, y, concept["caption_target"])
                for (x, y) in zip(inst_images_path, inst_prompt)
            ]
            self.instance_images_path.extend(inst_img_path)

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image: np.ndarray, scale, resample):
        outer, inner = self.size, scale
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // 8, self.size // 8))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // 8 + 1 : (top + scale) // 8 - 1,
                left // 8 + 1 : (left + scale) // 8 - 1,
            ] = 1.0
        return instance_image, mask

    def __getprompt__(self, instance_prompt: str, instance_target: str):
        if self.concept_type == "style":
            r = np.random.choice([0, 1, 2])
            instance_prompt = (
                f"{instance_prompt}, in the style of {instance_target}"
                if r == 0
                else f"in {instance_target}'s style, {instance_prompt}"
                if r == 1
                else f"in {instance_target}'s style, {instance_prompt}"
            )
        elif self.concept_type in ["nudity", "violence"]:
            r = np.random.choice([0, 1, 2])
            instance_prompt = (
                f"{instance_target}, {instance_prompt}"
                if r == 0
                else f"in {instance_target}'s style, {instance_prompt}"
                if r == 1
                else f"in {instance_target}'s style, {instance_prompt}"
            )
        elif self.concept_type == "object":
            anchor, target = instance_target.split("+")
            instance_prompt = instance_prompt.replace(anchor, target)
        elif self.concept_type == "memorization":
            instance_prompt = instance_target.split("+")[1]
        return instance_prompt

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        example = {}
        instance_image, instance_prompt, instance_target = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)
        # modify instance prompt according to the concept_type to include target concept
        # multiple style/object fine-tuning
        if ";" in instance_target:
            instance_target = instance_target.split(";")
            instance_target = instance_target[index % len(instance_target)]

        instance_anchor_prompt = instance_prompt
        instance_prompt = self.__getprompt__(instance_prompt, instance_target)
        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = (np.random.choice(["a far away ", "very small "]) + instance_prompt)
        elif random_scale > self.size:
            instance_prompt = (np.random.choice(["zoomed in ", "close up "]) + instance_prompt)

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["instance_anchor_prompt_ids"] = self.tokenizer(
            instance_anchor_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example

class DocoPromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt[index % len(self.prompt)]
        example["index"] = index
        return example
