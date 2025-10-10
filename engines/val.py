# engines/train_seg.py
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from gigapath.pipeline import run_inference_with_slide_encoder

@torch.no_grad()
def val_one_epoch(Dataloaders, tile_backbone, decoder, loss_fn, slide_encoder, num_classes):
    decoder.eval()
    total = 0
    loss_sum = 0
    # confusion matrix
    conf_total = torch.zeros(num_classes, num_classes, dtype=torch.int64, device="cpu")

    for batch_ in Dataloaders:
        # load tiles within this slide
        tile_loader = DataLoader(
            batch_, batch_size=8, shuffle=False, num_workers=0
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

        # evaluate decoder
        for tile_batch in tile_loader:
            imgs = tile_batch['img'].cuda()
            masks = tile_batch['mask'].cuda()
            with torch.no_grad():
                _, fmap = tile_backbone(imgs)

            with autocast(dtype=torch.float16):
                logits = decoder(fmap, slide_embeds["last_layer_embed"])  # [B, K, H, W]
            loss = loss_fn(logits.float(), masks)

            bs = imgs.size(0)
            total += bs
            loss_sum += loss.item() * bs

            preds = logits.argmax(1)  # [B,H,W]
            conf_total += fast_confmat(preds.cpu(), masks.cpu(), num_classes)

    metr = metrics_from_confmat(conf_total)
    metr["val_loss"] = loss_sum / max(total, 1)
    return metr


def fast_confmat(pred, target, num_classes):
    k = (target >= 0) & (target < num_classes)
    inds = num_classes * target[k].to(torch.int64) + pred[k]
    conf = torch.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf


def metrics_from_confmat(conf):
    tp = conf.diag()
    pos_gt = conf.sum(1)
    pos_pred = conf.sum(0)

    # IoU
    union = pos_gt + pos_pred - tp
    iou = tp.float() / union.clamp_min(1)
    miou = iou.mean().item()

    # F1
    f1_c = 2*tp.float() / (pos_gt + pos_pred).clamp_min(1)
    mf1 = f1_c.mean().item()

    # Pixel Acc
    pix_acc = (tp.sum().float() / conf.sum().clamp_min(1)).item()

    return {"mIoU": miou, "F1": mf1, "Acc": pix_acc}

