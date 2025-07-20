import os
import random
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import PIL
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

from train_methods.train_utils import prompt_augmentation, tokenize
from train_methods.templates import imagenet_style_templates_small, imagenet_templates_small, person_templates_small

PIL_INTERPOLATION = {
    "linear": PIL.Image.Resampling.BILINEAR,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
    "nearest": PIL.Image.Resampling.NEAREST,
}

class MACEDataset(Dataset):
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        size: int=512,
        multi_concept: Optional[list[str, str]]=None,
        mapping: Optional[list[str]]=None,
        batch_size: Optional[int]=None,
        train_seperate: bool=False,
        input_data_path: Optional[str]=None
    ):  
        self.size = size
        self.tokenizer = tokenizer
        self.batch_counter = 0
        self.batch_size = batch_size
        self.concept_number = 0
        self.train_seperate = train_seperate
        
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
            sampled_indices = random.sample(range(0, 30), 30)
            self.instance_prompt.append(prompt_augmentation(erased_concept, sampled_indices=sampled_indices, concept_type=t))
            self.target_prompt.append(prompt_augmentation(mapping_concept, sampled_indices=sampled_indices, concept_type=t))
                
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
        example["instance_prompt_ids"] = tokenize(instance_prompt, self.tokenizer).input_ids
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
    def __init__(
        self,
        concept_type: str,
        image_dir: str,
        prompt_path: str,
        tokenizer: CLIPTokenizer,
        concept: str,
        anchor_concept: Optional[str]=None,
        aug: bool=True
    ):
        
        self.size = 512
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
        self.flip = transforms.RandomHorizontalFlip(0.5)

        self.image_transforms = transforms.Compose([
            self.flip,
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(self.size),
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

        example = {
            "instance_images": torch.from_numpy(instance_image).permute(2, 0, 1),
            "mask": torch.from_numpy(mask),
            "instance_prompt_ids": tokenize(instance_prompt, self.tokenizer).input_ids,
            "instance_anchor_prompt_ids": tokenize(instance_anchor_prompt, self.tokenizer).input_ids
        }
        return example


class DocoDataset(Dataset):
    # 多分上と同じ
    def __init__(
        self,
        concepts_list: list[str],
        concept_type: str,
        tokenizer: CLIPTokenizer,
        center_crop=False,
        hflip=False,
        aug=True,
    ):
        self.size = 512
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
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size)
                if center_crop
                else transforms.RandomCrop(self.size),
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

        example = {
            "instance_images": torch.from_numpy(instance_image).permute(2, 0, 1),
            "mask": torch.from_numpy(mask),
            "instance_prompt_ids": tokenize(instance_prompt, self.tokenizer).input_ids,
            "instance_anchor_prompt_ids": tokenize(instance_anchor_prompt, self.tokenizer).input_ids
        }
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

class ForgetMeNotDataset(Dataset):
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        size: int=512,
        center_crop: bool=False,
        use_pooler: bool=False,
        multi_concept: Optional[list[str]]=None,
        data_dir: str="fmn-data"
    ):  
        self.use_pooler = use_pooler
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_images_path  = []
        self.instance_prompt  = []

        token_idx = 1
        for c, t, num_tok in multi_concept:
            p = Path(data_dir)
            if not p.exists():
                raise ValueError(f"Instance {p} images root doesn't exists.")                   
            
            image_paths = list(p.iterdir())
            self.instance_images_path += image_paths

            target_snippet = f"{''.join([ f'<s{token_idx + i}>' for i in range(num_tok)])}"
            if t == "object":
                self.instance_prompt += [(f"a photo of {target_snippet}", target_snippet)] * len(image_paths)
            elif t == "style":
                self.instance_prompt += [(f"a photo in the style of {target_snippet}", target_snippet)] * len(image_paths)
            else:
                raise ValueError("unknown concept type!")
            token_idx += num_tok
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images]).convert("RGB")
        instance_prompt, target_tokens = self.instance_prompt[index % self.num_instance_images]

        example["instance_prompt"] = instance_prompt
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = tokenize(instance_prompt, self.tokenizer).input_ids
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
            if self.use_pooler and tok_id == pooler_token_id:
                concept_positions[i] = 1
        example["concept_positions"] = torch.tensor(concept_positions)[None]               

        return example

class FMNPivotalTuningDataset(Dataset):
    def __init__(
        self,
        instance_data_root: str,
        tokenizer: CLIPTokenizer,
        token_map: Optional[dict]=None,
        use_template: Optional[str]=None,
        size: int=512,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.token_map = token_map
        self.use_template = use_template

        if use_template == "naked":
            self.templates = ["a photo of naked"]
        elif use_template == "style":
            self.templates = ["a photo in the style of {}"]
        else:
            self.templates = ["a photo of {}"]

        self._length = self.num_instance_images
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images]).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.use_template:
            assert self.token_map is not None
            input_tok = list(self.token_map.values())[0]
            text = random.choice(self.templates).format(input_tok)
        else:
            text = self.instance_images_path[index % self.num_instance_images].stem
            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text = text.replace(token, value)
        
        if random.random() > 0.5:
            hflip = transforms.RandomHorizontalFlip(p=1)
            example["instance_images"] = hflip(example["instance_images"])
            
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        
        example["uncond_prompt_ids"] = self.tokenizer(
            "",
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids        

        return example

class Imagenette(Dataset):
    def __init__(self, split: str, class_to_forget=None, transform=None):
        self.dataset = load_dataset("frgfm/imagenette", "160px")[split]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.dataset.features["label"].names)}
        self.file_to_class = {str(idx): self.dataset["label"][idx] for idx in range(len(self.dataset))}

        self.class_to_forget = class_to_forget
        self.num_classes = max(self.class_to_idx.values()) + 1
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if example["label"] == self.class_to_forget:
            label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)
        return image, label

class NSFW(Dataset):
    def __init__(self, transform=None):
        self.dataset = load_dataset("data/nsfw")["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> torch.Tensor:
        example = self.dataset[idx]
        return example["image"] if not self.transform else self.transform(example["image"])

class SalUnDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.dataset = load_dataset("imagefolder", data_dir=data_path, split="train")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> torch.Tensor:
        example = self.dataset[idx]
        return example["image"] if not self.transform else self.transform(example["image"])

class AnchorsDataset(Dataset):
    def __init__(self, prompt_path, concept):
        self.anchor_list = []
        with open(prompt_path, "r", encoding='utf-8') as f:
            for anchor in f.readlines():
                anchor_concept = anchor.strip('\n')
                if anchor_concept != concept:
                    self.anchor_list.append(anchor_concept)

    def __len__(self):
        return len(self.anchor_list)

    def __getitem__(self, index):
        anchor = self.anchor_list[index % len(self.anchor_list)]
        return anchor

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        iteration=None,       # New argument for the iteration
        num_iterations=None   # New argument for the number of images per subset
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.iteration = iteration
        self.num_iterations = num_iterations

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)

        # Dynamically calculate images_per_subset based on total images and number of iterations
        self.images_per_subset = max(1, self.num_images // self.num_iterations)

        # Partition image paths into non-overlapping subsets based on iteration and images_per_subset
        start_idx = (self.iteration * self.images_per_subset) % self.num_images
        end_idx = start_idx + self.images_per_subset
        if end_idx <= self.num_images:
            self.subset_image_paths = self.image_paths[start_idx:end_idx]
        else:
            # Wrap around if end_idx exceeds the number of images
            self.subset_image_paths = self.image_paths[start_idx:] + self.image_paths[:end_idx - self.num_images]

        self._length = len(self.subset_image_paths) * repeats if set == "train" else len(self.subset_image_paths)

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        if learnable_property == "object":
            self.templates = imagenet_templates_small
        elif learnable_property == "style":
            print(f"Using learnable property : {learnable_property}")
            self.templates = imagenet_style_templates_small
        elif learnable_property == "person":
            self.templates = person_templates_small

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        # Ensure the subset is non-overlapping by using subset_image_paths
        image_path = self.subset_image_paths[i % len(self.subset_image_paths)]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
