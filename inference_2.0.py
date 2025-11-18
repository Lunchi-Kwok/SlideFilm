import torch, timm
from torchvision import transforms

from models.film_decoder_sc import SegDecoderWithFiLM
from models.tile_backbone_sc import TileBackbone

from gigapath.pipeline import load_tile_slide_encoder
from gigapath.pipeline import TileEncodingDataset, build_paired_transforms, load_tile_encoder_transforms_val, load_mask_transforms_val
from torch.utils.data import Dataset, DataLoader
import os

from engines.val import val_one_epoch
from data.dataloader_ import SlideTilesDataset
import torch.nn as nn
from torch.cuda.amp import autocast
from gigapath.pipeline import run_inference_with_slide_encoder
from pathlib import Path
from PIL import Image
import numpy as np
import torch.nn.functional as F
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--HF_TOKEN", type=str, default="", help="HuggingFace token")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation slide directory")
    parser.add_argument("--out_root", type=str, default="tiles_masks", help="Output directory for masks")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--level", type=int, default=1)
    return parser.parse_args()

def pad5(n: int) -> str:
    n = int(n)
    if n < 0:
        return "-" + str(abs(n)).zfill(4)
    else:
        return str(n).zfill(5)

def slide_collate(batch):
    return batch[0]

def collate_with_optional_mask(batch):
    imgs = torch.stack([b["img"] for b in batch])
    coords = torch.stack([b["coords"] for b in batch])
    have_mask = all(("mask" in b) for b in batch)
    masks = torch.stack([b["mask"] for b in batch]) if have_mask else None
    return {"img": imgs, "coords": coords, "mask": masks}


def main():
    args = parse_args()
    # 1) load model
    os.environ["HF_TOKEN"] = args.HF_TOKEN
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tile_encoder, slide_encoder = load_tile_slide_encoder(global_pool=True)
    tile_backbone = TileBackbone(tile_encoder, layers=[5, 11, 23, 39]).eval().to(device)
    slide_encoder = slide_encoder.eval().to(device)

    decoder = SegDecoderWithFiLM(slide_dim=768, in_ch=1536, chs=(768, 384, 192, 96), n_cls=args.NUM_CLASSES, skip_layers=(5, 11, 23, 39),
                                 skip_in_ch=1536).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    decoder.load_state_dict(checkpoint["model_state_dict"])
    decoder.eval()

    # 2) load data
    slide_dir_val = args.val_dir
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)
    img_files = sorted(os.listdir(slide_dir_val))

    raw_tile_resolution = 256 * (2 ** args.level)
    img_size = (raw_tile_resolution, raw_tile_resolution)


    paired_transform, img_post, mask_post = build_paired_transforms()
    val_ds = SlideTilesDataset(slide_dir=slide_dir_val, transform=load_tile_encoder_transforms_val(), mask_dir=None,
                               mask_transform=None, paired_transform=None)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=slide_collate)


    for slide_idx, tile_ds in enumerate(val_dl):
        # load tiles within this slide
        tile_loader = DataLoader(
            tile_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_with_optional_mask
        )

        # extract slide embedding
        collated_outputs = {'tile_embeds': [], 'coords': []}
        with autocast(dtype=torch.float16):
            for batch in tile_loader:
                # cls_tok, _ = tile_backbone(batch['img'].cuda())
                out = tile_backbone(batch['img'].cuda())
                cls_tok = out[0]
                collated_outputs['tile_embeds'].append(cls_tok.detach().cpu())
                collated_outputs['coords'].append(batch['coords'])
        tile_encoder_outputs = {k: torch.cat(v) for k, v in collated_outputs.items()}
        slide_embeds = run_inference_with_slide_encoder(
            slide_encoder_model=slide_encoder, **tile_encoder_outputs
        )

        slide_name = Path(img_files[slide_idx]).stem
        slide_out_dir = Path(out_root) / slide_name
        slide_out_dir.mkdir(parents=True, exist_ok=True)

        slide_vec = slide_embeds["last_layer_embed"].to(next(decoder.parameters()).device)

        # evaluate decoder
        for tile_batch in tile_loader:
            coords = tile_batch["coords"].cpu().numpy()
            imgs = tile_batch['img'].cuda()

            # with torch.no_grad():
            #     _, fmap = tile_backbone(imgs)

            out = tile_backbone(imgs)
            if len(out) == 2:
                _, fmap = out
                skip_fmaps = None
            else:
                _, fmap, _, fmap_dict = out
                skip_layers = getattr(decoder, "skip_layers", [])
                skip_fmaps = {li: fmap_dict[li] for li in skip_layers if li in fmap_dict}

            if slide_vec.dim() == 1:
                slide_vec_b = slide_vec.unsqueeze(0).expand(imgs.size(0), -1)
            elif slide_vec.size(0) == 1 and imgs.size(0) > 1:
                slide_vec_b = slide_vec.expand(imgs.size(0), -1)
            else:
                slide_vec_b = slide_vec

                # with autocast(dtype=torch.float16):
            #     logits = decoder(fmap, slide_embeds["last_layer_embed"])  # [B, K, H, W]

            with autocast(dtype=torch.float16):
                logits = decoder(fmap, slide_vec_b, skip_fmaps) if skip_fmaps is not None else decoder(fmap, slide_vec_b)

            logits_256 = F.interpolate(logits, size=img_size, mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits_256)  # [B,1,H,W]
            pred = (probs > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)  # [B,H,W], 0/1

            for mask, (x, y) in zip(pred, coords):
                fname = f"{pad5(int(x))}x_{pad5(int(y))}y.png"
                out_p = slide_out_dir / fname
                mask = (mask * 255).astype(np.uint8)
                Image.fromarray(mask).save(out_p)

if __name__ == "__main__":
    main()