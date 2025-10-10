import torch, timm
from torchvision import transforms
from models.tile_backbone import TileBackbone
from models.film_decoder import SegDecoderWithFiLM
import huggingface_hub
from gigapath.pipeline import load_tile_slide_encoder
from gigapath.pipeline import TileEncodingDataset,build_paired_transforms, load_tile_encoder_transforms_val, load_mask_transforms_val
from torch.utils.data import Dataset, DataLoader
import os
from engines.train import CE_DiceLoss, train_one_epoch, FocalLoss
from engines.val import val_one_epoch
from data.dataloader_ import SlideTilesDataset
import argparse
import torch.nn as nn


def main():
    device = "cuda"
    os.environ["HF_TOKEN"] = ""
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--NUM_CLASSES", type=int, required=True)
    parser.add_argument("--slide_dir_train", type=str, required=True)
    parser.add_argument("--mask_dir_train", type=str, required=True)
    parser.add_argument("--slide_dir_val", type=str, required=True)
    parser.add_argument("--mask_dir_val", type=str, required=True)

    args = parser.parse_args()

    # --- 1) Tile backbone ---
    tile_encoder, slide_encoder = load_tile_slide_encoder(global_pool=True)
    tile_backbone = TileBackbone(tile_encoder).eval().to(device)  #extract patch tokens  -->  cls_token[B, C], fmap[B, C, H, W]   B:Num_Tiles
    for p in tile_backbone.parameters(): p.requires_grad = False

    # --- 2) Slide encoder (LongNet) ---
    for p in slide_encoder.parameters(): p.requires_grad = False

    # --- 3) Seg decoder + FiLM ---
    decoder = SegDecoderWithFiLM(slide_dim=768, in_ch=1536, n_cls=args.NUM_CLASSES).to(device) #slide_dim got from run_inference_with_slide_encoder, in_ch got from run_inference_with_tile_encoder, change them when necessary

    # --- 4) Dataloaders ---
    def slide_collate(batch):
        return batch[0]

    paired_transform, img_post, mask_post = build_paired_transforms()
    train_ds = SlideTilesDataset(slide_dir=args.slide_dir_train, transform=img_post,mask_dir=args.mask_dir_train,mask_transform=mask_post,paired_transform=paired_transform)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=slide_collate)

    val_ds = SlideTilesDataset(slide_dir=args.slide_dir_val,transform=load_tile_encoder_transforms_val(),mask_dir=args.mask_dir_val,mask_transform=load_mask_transforms_val(),paired_transform=None)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=slide_collate)


    # --- 6) train ---
    optimizer = torch.optim.AdamW(list(decoder.parameters()), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    #loss_fn = CE_DiceLoss()     #multi class mask
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    best_score = 0

    for epoch in range(args.epochs + 1):
        tr_loss = train_one_epoch(train_dl, tile_backbone, decoder, optimizer, scaler, loss_fn, slide_encoder)
        if (epoch + 1) % 10 == 0:
            val_stats = val_one_epoch(
                val_dl, tile_backbone, decoder, loss_fn, slide_encoder, args.NUM_CLASSES
            )
            print(f"[Epoch {epoch + 1:03d}] "
                  f"train_loss={tr_loss:.4f} | "
                  f"val_loss={val_stats['val_loss']:.4f} | "
                  f"mIoU={val_stats['mIoU']:.4f} | "
                  f"F1={val_stats['F1']:.4f} | "
                  f"Acc={val_stats['Acc']:.4f}")

            if val_stats['F1'] > best_score:  # save best model
                best_score = val_stats['F1']
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_F1": best_score,
                    },
                    "best_model.pth"
                )
                print(f"✅ New best model saved at epoch {epoch + 1}, mIoU={best_score:.4f}")

if __name__ == "__main__":
    main()
