# --------------------------------------------------------
# Pipeline for running with GigaPath
# --------------------------------------------------------
import os
import timm
import torch
import shutil
import numpy as np
import pandas as pd
import gigapath.slide_encoder as slide_encoder

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from gigapath.preprocessing.data.create_tiles_dataset import process_slide, process_mask_slide
import torchvision.transforms.functional as TF
import random


def _to_long(pic):
    arr = np.array(pic)  # PIL -> numpy
    return torch.as_tensor(arr, dtype=torch.long)


class TileEncodingDataset(Dataset):
    """
    Do encoding for tiles

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
        Example: ['images/256x_256y.png', 'images/256x_512y.png']
    transform : torchvision.transforms.Compose
        Transform to apply to each image
    """

    def __init__(self, image_paths: List[str], transform=None, mask_dir: List[str] = None, mask_transform=None,
                 paired_transform=None):
        self.transform = transform
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.mask_transform = mask_transform
        self.paired_transform = paired_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # get x, y coordinates from the image name
        x, y = img_name.split('.png')[0].split('_')
        x, y = int(x.replace('x', '')), int(y.replace('y', ''))
        # load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        mask = None
        if self.mask_dir is not None:
            mask_path = self.mask_dir[idx]
            with open(mask_path, "rb") as f:
                mask = Image.open(f).convert("L")

            if self.paired_transform is not None:
                img, mask = self.paired_transform(img, mask)

        img = self.transform(img)

        if self.mask_transform is not None and mask is not None:
            mask = self.mask_transform(mask)

        sample = {
            "img": img,
            "coords": torch.from_numpy(np.array([x, y])).float(),
        }
        if mask is not None:
            sample["mask"] = mask

        return sample


def tile_one_slide(slide_file: str = '', save_dir: str = '', level: int = 0, tile_size: int = 256, img_size : str = '1000_1000'):
    """
    This function is used to tile a single slide and save the tiles to a directory.
    -------------------------------------------------------------------------------
    Warnings: pixman 0.38 has a known bug, which produces partial broken images.
    Make sure to use a different version of pixman.
    -------------------------------------------------------------------------------

    Arguments:
    ----------
    slide_file : str
        The path to the slide file.
    save_dir : str
        The directory to save the tiles.
    level : int
        The magnification level to use for tiling. level=0 is the highest magnification level.
    tile_size : int
        The size of the tiles.
    """
    slide_id = os.path.splitext(os.path.basename(slide_file))[0]
    slide_id = f"{slide_id}_{img_size}"

    # slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {'TP53': 1, 'Diagnosis': 'Lung Cancer'}}
    slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {}}

    save_dir = Path(save_dir)
    if save_dir.exists():
        print(f"Warning: Directory {save_dir} already exists. ")

    print(f"Processing slide {slide_file} at level {level} with tile size {tile_size}. Saving to {save_dir}.")

    slide_dir = process_slide(
        slide_sample,
        level=level,
        margin=0,
        tile_size=tile_size,
        foreground_threshold=256,
        occupancy_threshold=0,
        output_dir=save_dir / "output",
        thumbnail_dir=save_dir / "thumbnails",
        tile_progress=True,
    )

    dataset_csv_path = slide_dir / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_path)
    assert len(dataset_df) > 0
    failed_csv_path = slide_dir / "failed_tiles.csv"
    failed_df = pd.read_csv(failed_csv_path)
    assert len(failed_df) == 0

    print(f"Slide {slide_file} has been tiled. {len(dataset_df)} tiles saved to {slide_dir}.")


def load_tile_encoder_transforms() -> transforms.Compose:
    """Load the transforms for the tile encoder"""
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return transform


# def load_mask_transforms():
#     transform = transforms.Compose(
#         [
#             transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
#             # transforms.CenterCrop(256),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(degrees=90),
#             _to_long
#         ]
#     )
#     return transform


def load_tile_encoder_transforms_val() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def load_mask_transforms_val():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
        # transforms.CenterCrop(256),
        _to_long,
    ])


def load_tile_slide_encoder(local_tile_encoder_path: str = '',
                            local_slide_encoder_path: str = '',
                            global_pool=False) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the GigaPath tile and slide encoder models.
    Note: Older versions of timm have compatibility issues.
    Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
    """
    if local_tile_encoder_path:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False,
                                         checkpoint_path=local_tile_encoder_path)
    else:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    print("Tile encoder param #", sum(p.numel() for p in tile_encoder.parameters()))

    if local_slide_encoder_path:
        slide_encoder_model = slide_encoder.create_model(local_slide_encoder_path, "gigapath_slide_enc12l768d", 1536,
                                                         global_pool=global_pool)
    else:
        slide_encoder_model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath",
                                                         "gigapath_slide_enc12l768d", 1536, global_pool=global_pool)
    print("Slide encoder param #", sum(p.numel() for p in slide_encoder_model.parameters()))

    return tile_encoder, slide_encoder_model


@torch.no_grad()
def run_inference_with_tile_encoder(image_paths: List[str], tile_encoder: torch.nn.Module,
                                    batch_size: int = 128) -> dict:
    """
    Run inference with the tile encoder

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
    tile_encoder : torch.nn.Module
        Tile encoder model
    """
    tile_encoder = tile_encoder.cuda()
    # make the tile dataloader
    tile_dl = DataLoader(TileEncodingDataset(image_paths, transform=load_tile_encoder_transforms()),
                         batch_size=batch_size, shuffle=False)
    # run inference
    tile_encoder.eval()
    collated_outputs = {'tile_embeds': [], 'coords': []}
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(tile_dl, desc='Running inference with tile encoder'):
            collated_outputs['tile_embeds'].append(tile_encoder(batch['img'].cuda()).detach().cpu())
            collated_outputs['coords'].append(batch['coords'])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}


@torch.no_grad()
def run_inference_with_slide_encoder(tile_embeds: torch.Tensor, coords: torch.Tensor,
                                     slide_encoder_model: torch.nn.Module) -> torch.Tensor:
    """
    Run inference with the slide encoder

    Arguments:
    ----------
    tile_embeds : torch.Tensor
        Tile embeddings
    coords : torch.Tensor
        Coordinates of the tiles
    slide_encoder_model : torch.nn.Module
        Slide encoder model
    """
    if len(tile_embeds.shape) == 2:
        tile_embeds = tile_embeds.unsqueeze(0)
        coords = coords.unsqueeze(0)

    slide_encoder_model = slide_encoder_model.cuda()
    slide_encoder_model.eval()
    # run inference
    with torch.cuda.amp.autocast(dtype=torch.float16):
        slide_embeds = slide_encoder_model(tile_embeds.cuda(), coords.cuda(), all_layer_embed=True)
    outputs = {"layer_{}_embed".format(i): slide_embeds[i].cpu() for i in range(len(slide_embeds))}
    outputs["last_layer_embed"] = slide_embeds[-1]
    return outputs


class PairedGeomTransform:
    def __init__(self, size_resize=224, size_crop=224, p_hflip=0.5, p_vflip=0.5):
        self.size_resize = size_resize
        self.size_crop = size_crop
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, img: Image.Image, mask: Image.Image):
        # 1) 统一 Resize 到 256
        img = TF.resize(img, self.size_resize, interpolation=TF.InterpolationMode.BICUBIC)
        mask = TF.resize(mask, self.size_resize, interpolation=TF.InterpolationMode.NEAREST)

        # 2) 同步随机翻转
        if random.random() < self.p_hflip:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < self.p_vflip:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # 3) 同步随机旋转（离散角度，避免 mask 插值伪影）
        angle = random.choice([-90, 0, 90, 180])
        if angle != 0:
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BICUBIC, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

        # 4) CenterCrop 到 224
        # img  = TF.center_crop(img,  self.size_crop)
        # mask = TF.center_crop(mask, self.size_crop)

        return img, mask


def load_tile_img_post_transforms():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_mask_post_transforms():
    def _to_tensor_long(mask_pil: Image.Image):
        # PIL -> [H,W] uint8 Tensor
        m = transforms.PILToTensor()(mask_pil).squeeze(0)  # [H,W], uint8
        m = m.long()
        return m

    return _to_tensor_long


def build_paired_transforms():
    paired = PairedGeomTransform(
        size_resize=224,
        size_crop=224,
        p_hflip=0.5,
        p_vflip=0.5,
    )
    img_post = load_tile_img_post_transforms()
    mask_post = load_mask_post_transforms()
    return paired, img_post, mask_post