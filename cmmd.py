# using https://github.com/sayakpaul/cmmd-pytorch

# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""

from glob import glob

import numpy as np
import tqdm
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

def mmd(x, y):
    """Memory-efficient MMD

    This implements the minimum-variance/biased version of the estimator described in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the minimum-variance estimate for MMD are almost identical.

    Args:
        x: The first set of embeddings of shape (n, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
        The MMD distance between x and y embedding sets.
    """

    _SIGMA = 10
    _SCALE = 1000

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0))))
    k_xy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0))))
    k_yy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0))))

    return _SCALE * (k_xx + k_yy - 2 * k_xy)


def _resize_bicubic(images, size):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
    images = images.permute(0, 2, 3, 1).numpy()
    return images

class ClipEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self):
        _CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
        self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)
        self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        self._model = self._model.cuda()

        self.input_image_size = self.image_processor.crop_size["height"]

    @torch.no_grad()
    def embed(self, images):
        """Computes CLIP embeddings for the given images.

        Args:
            images: An image array of shape (batch_size, height, width, 3). Values are in range [0, 1].

        Returns:
            Embedding array of shape (batch_size, embedding_width).
        """

        images = _resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        image_embs = self._model(**inputs).image_embeds.cpu()
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs

class CMMDDataset(Dataset):
    def __init__(self, path, reshape_to, max_count=-1):
        self.path = path
        self.reshape_to = reshape_to

        self.max_count = max_count
        img_path_list = self._get_image_list()
        if max_count > 0:
            img_path_list = img_path_list[:max_count]
        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def _get_image_list(self):
        ext_list = ["png", "jpg", "jpeg"]
        image_list = []
        for ext in ext_list:
            image_list.extend(glob(f"{self.path}/*{ext}"))
            image_list.extend(glob(f"{self.path}/*.{ext.upper()}"))
        # Sort the list to ensure a deterministic output.
        image_list.sort()
        return image_list

    def _center_crop_and_resize(self, im: Image.Image, size: int):
        w, h = im.size
        l = min(w, h)
        top = (h - l) // 2
        left = (w - l) // 2
        box = (left, top, left + l, top + l)
        im = im.crop(box)
        # Note that the following performs anti-aliasing as well.
        return im.resize((size, size), resample=Image.Resampling.BICUBIC)  # pytype: disable=module-attr

    def _read_image(self, path, size):
        im = Image.open(path)
        if size > 0:
            im = self._center_crop_and_resize(im, size)
        return np.asarray(im).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]

        x = self._read_image(img_path, self.reshape_to)
        if x.ndim == 3:
            return x
        elif x.ndim == 2:
            # Convert grayscale to RGB by duplicating the channel dimension.
            return np.tile(x[Ellipsis, np.newaxis], (1, 1, 3))

def compute_embeddings_for_dir(
    img_dir,
    embedding_model: ClipEmbeddingModel,
    batch_size,
    max_count=-1,
):
    """Computes embeddings for the images in the given directory.

    This drops the remainder of the images after batching with the provided
    batch_size to enable efficient computation on TPUs. This usually does not
    affect results assuming we have a large number of images in the directory.

    Args:
        img_dir: Directory containing .jpg or .png image files.
        embedding_model: The embedding model to use.
        batch_size: Batch size for the embedding model inference.
        max_count: Max number of images in the directory to use.

    Returns:
        Computed embeddings of shape (num_images, embedding_dim).
    """
    dataset = CMMDDataset(img_dir, reshape_to=embedding_model.input_image_size, max_count=max_count)
    count = len(dataset)
    print(f"Calculating embeddings for {count} images from {img_dir}.")

    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_embs = []
    for batch in tqdm.tqdm(dataloader, total=count // batch_size):
        image_batch = batch.numpy()

        # Normalize to the [0, 1] range.
        image_batch = image_batch / 255.0

        if np.min(image_batch) < 0 or np.max(image_batch) > 1:
            raise ValueError("Image values are expected to be in [0, 1]. Found:" f" [{np.min(image_batch)}, {np.max(image_batch)}].")

        # Compute the embeddings using a pmapped function.
        embs = np.asarray(embedding_model.embed(image_batch))  # The output has shape (num_devices, batch_size, embedding_dim).
        all_embs.append(embs)

    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs

def compute_cmmd(ref_dir, eval_dir, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
        ref_dir: Path to the directory containing reference images.
        eval_dir: Path to the directory containing images to be evaluated.
        ref_embed_file: Path to the pre-computed embedding file for the reference images.
        batch_size: Batch size used in the CLIP embedding calculation.
        max_count: Maximum number of images to use from each directory. A non-positive value reads all images available except for the images dropped due to batching.

    Returns:
        The CMMD value between the image sets.
    """
    
    embedding_model = ClipEmbeddingModel()
    ref_embs = compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype("float32")
    eval_embs = compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    val = mmd(ref_embs, eval_embs)
    return val.numpy()


def main(dir1, dir2, batch_size=32, max_count=-1):
    print(
        "The CMMD value is: "
        f" {compute_cmmd(dir1, dir2, batch_size, max_count):.3f}"
    )


    