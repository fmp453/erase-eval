import os
import random
import regex as re

from pathlib import Path

import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def clean_prompt(class_prompt_collection):
    class_prompt_collection = [re.sub(r"[0-9]+", lambda num: '' * len(num.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [re.sub(r"^\.+", lambda dots: '' * len(dots.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    class_prompt_collection = [x.replace('"', '') for x in class_prompt_collection]
    return class_prompt_collection
