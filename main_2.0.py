import torch, timm
from torchvision import transforms
from models.tile_backbone_sc import TileBackbone
from models.film_decoder_sc import SegDecoderWithFiLM
import huggingface_hub
from gigapath.pipeline import load_tile_slide_encoder
from gigapath.pipeline import TileEncodingDataset, build_paired_transforms, load_tile_encoder_transforms_val, load_mask_transforms_val
from torch.utils.data import Dataset, DataLoader
import os

from engines.train_sc import train_one_epoch, BinaryFocalDiceLoss
from engines.val_sc import val_one_epoch

from data.dataloader_ import SlideTilesDataset
import argparse
import torch.nn as nn
import math


def main():
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--HF_TOKEN", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--NUM_CLASSES", type=int, required=True)
    parser.add_argument("--slide_dir_train", type=str, required=True)
    parser.add_argument("--mask_dir_train", type=str, required=True)
    parser.add_argument("--slide_dir_val", type=str, required=True)
    parser.add_argument("--mask_dir_val", type=str, required=True)

    args = parser.parse_args()

    os.environ["HF_TOKEN"] = args.HF_TOKEN

    # --- 1) Tile backbone ---
    tile_encoder, slide_encoder = load_tile_slide_encoder(global_pool=True)
    tile_backbone = TileBackbone(tile_encoder, layers=[5, 11, 23, 39]).eval().to(device)
    for p in tile_backbone.parameters(): p.requires_grad = False

    # --- 2) Slide encoder (LongNet) ---
    for p in slide_encoder.parameters(): p.requires_grad = False

    # --- 3) Seg decoder + FiLM ---
    decoder = SegDecoderWithFiLM(
        slide_dim=768,
        in_ch=1536,
        chs=(768, 384, 192, 96),
        n_cls=args.NUM_CLASSES,
        skip_layers=(5, 11, 23, 39),
        skip_in_ch=1536
    ).to(device)

    # --- 4) Dataloaders ---
    def slide_collate(batch):
        return batch[0]

    def bg_ratio_cosine(epoch, T, start=0.6, end=0.1):
        t = min(max(epoch / max(T - 1, 1), 0.0), 1.0)
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))

    paired_transform, img_post, mask_post = build_paired_transforms()
    train_ds = SlideTilesDataset(slide_dir=args.slide_dir_train, transform=img_post, mask_dir=args.mask_dir_train,
                                 mask_transform=mask_post, paired_transform=paired_transform)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=slide_collate)

    val_ds = SlideTilesDataset(slide_dir=args.slide_dir_val, transform=load_tile_encoder_transforms_val(),
                               mask_dir=args.mask_dir_val, mask_transform=load_mask_transforms_val(),
                               paired_transform=None)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=slide_collate)

    # --- 6) train ---
    optimizer = torch.optim.AdamW(list(decoder.parameters()), lr=1e-5, weight_decay=1e-4)
    # optimizer = torch.optim.AdamW([
    #     {"params": decoder.parameters(), "lr": 1e-3},
    #     {"params": tile_backbone.parameters(), "lr": 1e-4},
    #     {"params": slide_encoder.parameters(), "lr": 5e-5},
    # ], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    # loss_fn = CE_DiceLoss()     #multi class mask
    # loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    # loss_fn = FocalDiceLoss(alpha=0.25, gamma=2.0, lambda_dice=1.0)
    loss_fn = BinaryFocalDiceLoss(alpha=0.25, gamma=2.0, lambda_dice=1.0)
    best_score = 0

    with open("train_log.txt", "w") as f:
        f.write("Epoch,Train_Loss,Val_Loss,mIoU,F1,Acc\n")
        for epoch in range(args.epochs):
            ratio = bg_ratio_cosine(epoch, args.epochs, start=0.6, end=0.1)
            tr_loss = train_one_epoch(train_dl, tile_backbone, decoder, optimizer, scaler, loss_fn, slide_encoder,
                                      ratio)
            if (epoch + 1) % 10 == 0:
                val_stats = val_one_epoch(val_dl, tile_backbone, decoder, loss_fn, slide_encoder)

                log_line = (f"{epoch + 1:03d},{tr_loss:.4f},"
                            f"{val_stats['val_loss']:.4f},"
                            f"{val_stats['IoU']:.4f},"
                            f"{val_stats['Dice']:.4f},"
                            f"{val_stats['Acc']:.4f}\n")
                print(log_line.strip())
                f.write(log_line)
                f.flush()

                if val_stats['Dice'] > best_score:
                    best_score = val_stats['Dice']
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_Dice": best_score,
                    }, "best_model.pth")
                    print(f"✅ New best model saved at epoch {epoch + 1}, Dice={best_score:.4f}")


if __name__ == "__main__":
    main()
