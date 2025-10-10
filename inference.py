import torch, timm
from torchvision import transforms
from models.tile_backbone import TileBackbone
from models.film_decoder import SegDecoderWithFiLM
from gigapath.pipeline import load_tile_slide_encoder
from gigapath.pipeline import TileEncodingDataset,build_paired_transforms, load_tile_encoder_transforms_val, load_mask_transforms_val
from torch.utils.data import Dataset, DataLoader
import os
from engines.train import CE_DiceLoss, train_one_epoch
from engines.val import val_one_epoch
from data.dataloader_ import SlideTilesDataset
import torch.nn as nn
from torch.cuda.amp import autocast
from gigapath.pipeline import run_inference_with_slide_encoder
from pathlib import Path
from PIL import Image
import numpy as np
import torch.nn.functional as F

# 1) load model
os.environ["HF_TOKEN"] = ""
device = "cuda" if torch.cuda.is_available() else "cpu"

tile_encoder, slide_encoder = load_tile_slide_encoder(global_pool=True)
tile_backbone = TileBackbone(tile_encoder).eval().to(device)
slide_encoder = slide_encoder.eval().to(device)

NUM_CLASSES = 2
decoder = SegDecoderWithFiLM(slide_dim=768, in_ch=1536, n_cls=NUM_CLASSES).to(device)

checkpoint = torch.load("/content/drive/MyDrive/prov-gigapath/best_model.pth", map_location=device)
decoder.load_state_dict(checkpoint["model_state_dict"])
decoder.eval()

# 2) load data
slide_dir_val = r"/content/drive/MyDrive/val_Lab/img"
out_root = "tiles_masks"
os.makedirs(out_root, exist_ok=True)
img_files = sorted(os.listdir(slide_dir_val))

level = 1
raw_tile_resolution = 256 * (level + 1)
img_size = (raw_tile_resolution,raw_tile_resolution)

def slide_collate(batch):
    return batch[0]

def collate_with_optional_mask(batch):
    imgs = torch.stack([b["img"] for b in batch])
    coords = torch.stack([b["coords"] for b in batch])
    have_mask = all(("mask" in b) for b in batch)
    masks = torch.stack([b["mask"] for b in batch]) if have_mask else None
    return {"img": imgs, "coords": coords, "mask": masks}


paired_transform, img_post, mask_post = build_paired_transforms()
val_ds = SlideTilesDataset(slide_dir=slide_dir_val,transform=load_tile_encoder_transforms_val(),mask_dir=None,mask_transform=None,paired_transform=None)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=slide_collate)

#name files
def pad5(n: int) -> str:
    n = int(n)
    if n < 0:
        return "-" + str(abs(n)).zfill(4)
    else:
        return str(n).zfill(5)

for slide_idx, tile_ds in enumerate(val_dl):
    # load tiles within this slide
    tile_loader = DataLoader(
        tile_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_with_optional_mask
    )

    # extract slide embedding
    collated_outputs = {'tile_embeds': [], 'coords': []}
    with autocast(dtype=torch.float16):
        for batch in tile_loader:
            cls_tok, _ = tile_backbone(batch['img'].cuda())
            collated_outputs['tile_embeds'].append(cls_tok.detach().cpu())
            collated_outputs['coords'].append(batch['coords'])
    tile_encoder_outputs = {k: torch.cat(v) for k, v in collated_outputs.items()}
    slide_embeds = run_inference_with_slide_encoder(
        slide_encoder_model=slide_encoder, **tile_encoder_outputs
    )

    slide_name = Path(img_files[slide_idx]).stem
    slide_out_dir = Path(out_root) / slide_name
    slide_out_dir.mkdir(parents=True, exist_ok=True)

    # evaluate decoder
    for tile_batch in tile_loader:
        coords = tile_batch["coords"].cpu().numpy()
        imgs = tile_batch['img'].cuda()

        with torch.no_grad():
            _, fmap = tile_backbone(imgs)

        with autocast(dtype=torch.float16):
            logits = decoder(fmap, slide_embeds["last_layer_embed"])  # [B, K, H, W]

        logits_256 = F.interpolate(logits, size=img_size, mode="bilinear", align_corners=False)
        pred = logits_256.float().argmax(1).cpu().numpy()

        for mask, (x, y) in zip(pred, coords):
            fname = f"{pad5(int(x))}x_{pad5(int(y))}y.png"
            out_p = slide_out_dir / fname

            mask = (mask * 255).astype(np.uint8)

            Image.fromarray(mask).save(out_p)
