#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.tile_backbone import TileBackbone
from models.film_decoder import SegDecoderWithFiLM
from gigapath.pipeline import (
    load_tile_slide_encoder,
    build_paired_transforms,
    load_tile_encoder_transforms_val,
    load_mask_transforms_val,
)
from data.dataloader_ import SlideTilesDataset
from engines.train import train_one_epoch, FocalLoss
from engines.val import val_one_epoch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--NUM_CLASSES", type=int, required=True)
    parser.add_argument("--slide_dir_train", type=str, required=True)
    parser.add_argument("--mask_dir_train", type=str, required=True)
    parser.add_argument("--slide_dir_val", type=str, required=True)
    parser.add_argument("--mask_dir_val", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained decoder checkpoint (.pth)")
    parser.add_argument("--HF_TOKEN", type=str, default="", help="Optional HF token env value")
    args = parser.parse_args()

    if args.HF_TOKEN:
        os.environ["HF_TOKEN"] = args.HF_TOKEN

    # --- 1) Load encoders (frozen, same as original) ---
    tile_encoder, slide_encoder = load_tile_slide_encoder(global_pool=True)
    tile_backbone = TileBackbone(tile_encoder).eval().to(device)
    for p in tile_backbone.parameters():
        p.requires_grad = False

    slide_encoder = slide_encoder.eval().to(device)
    for p in slide_encoder.parameters():
        p.requires_grad = False

    # --- 2) Build decoder and load checkpoint ---
    decoder = SegDecoderWithFiLM(slide_dim=768, in_ch=1536, n_cls=args.NUM_CLASSES).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # If NUM_CLASSES changed compared to pretraining, head shape may differ.
    # We allow non-strict loading in that case to reuse the shared layers.
    try:
        decoder.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        decoder.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Train decoder only
    for p in decoder.parameters():
        p.requires_grad = True

    # --- 3) Dataloaders (same transforms & collate style) ---
    def slide_collate(batch):
        # SlideTilesDataset yields one item per slide; with batch_size=1 we return that item directly
        return batch[0]

    paired_transform, img_post, mask_post = build_paired_transforms()

    train_ds = SlideTilesDataset(
        slide_dir=args.slide_dir_train,
        transform=img_post,
        mask_dir=args.mask_dir_train,
        mask_transform=mask_post,
        paired_transform=paired_transform,
    )
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=slide_collate)

    val_ds = SlideTilesDataset(
        slide_dir=args.slide_dir_val,
        transform=load_tile_encoder_transforms_val(),
        mask_dir=args.mask_dir_val,
        mask_transform=load_mask_transforms_val(),
        paired_transform=None,
    )
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=slide_collate)

    # --- 4) Optimizer, AMP scaler, and loss (FocalLoss only) ---
    optimizer = torch.optim.AdamW(list(decoder.parameters()), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # --- 5) Finetune loop (validate every 10 epochs; save best by F1) ---
    best_score = ckpt.get("best_F1", 0.0) if isinstance(ckpt, dict) else 0.0

    for epoch in range(args.epochs + 1):
        # Train one epoch (engine follows your original signature)
        tr_loss = train_one_epoch(
            train_dl, tile_backbone, decoder, optimizer, scaler, loss_fn, slide_encoder
        )

        # Validate every 10 epochs (same cadence as your training code)
        if (epoch + 1) % 10 == 0:
            val_stats = val_one_epoch(
                val_dl, tile_backbone, decoder, loss_fn, slide_encoder, args.NUM_CLASSES
            )
            print(
                f"[Epoch {epoch + 1:03d}] "
                f"train_loss={tr_loss:.4f} | "
                f"val_loss={val_stats['val_loss']:.4f} | "
                f"mIoU={val_stats['mIoU']:.4f} | "
                f"F1={val_stats['F1']:.4f} | "
                f"Acc={val_stats['Acc']:.4f}"
            )

            # Save best model by F1 (same logic)
            if val_stats["F1"] > best_score:
                best_score = val_stats["F1"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_F1": best_score,
                    },
                    "best_model_finetuned.pth",
                )
                print(f"✅ New best model saved at epoch {epoch + 1}, F1={best_score:.4f}")

        else:
            print(f"[Epoch {epoch + 1:03d}] train_loss={tr_loss:.4f}")

    print("Finetune complete.")


if __name__ == "__main__":
    main()
