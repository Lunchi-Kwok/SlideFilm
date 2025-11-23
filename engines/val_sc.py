import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from gigapath.pipeline import run_inference_with_slide_encoder

@torch.no_grad()
def val_one_epoch(Dataloaders, tile_backbone, decoder, loss_fn, slide_encoder, threshold=0.5):
    """
    Validation loop for binary segmentation (1-channel logits + sigmoid).
    Computes tile-level average Dice / IoU / Accuracy and val_loss.
    """
    decoder.eval()
    tile_backbone.eval()
    slide_encoder.eval()

    total_samples = 0
    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    acc_sum = 0.0

    eps = 1e-6

    for batch_ in Dataloaders:
        tile_loader = DataLoader(batch_, batch_size=8, shuffle=False, num_workers=0)

        collated_outputs = {'tile_embeds': [], 'coords': []}
        with autocast(dtype=torch.float16):
            for batch in tile_loader:
                out = tile_backbone(batch['img'].cuda())
                cls_tok = out[0]   # [B, D]
                collated_outputs['tile_embeds'].append(cls_tok.detach().cpu())
                collated_outputs['coords'].append(batch['coords'])

        tile_encoder_outputs = {k: torch.cat(v) for k, v in collated_outputs.items()}
        slide_embeds = run_inference_with_slide_encoder(
            slide_encoder_model=slide_encoder, **tile_encoder_outputs
        )
        slide_vec = slide_embeds["last_layer_embed"].to(next(decoder.parameters()).device)

        for tile_batch in tile_loader:
            imgs  = tile_batch['img'].cuda()   # [B,3,H,W]
            masks = tile_batch['mask'].cuda()

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

            with autocast(dtype=torch.float16):
                logits = decoder(fmap, slide_vec_b, skip_fmaps) if skip_fmaps is not None else decoder(fmap, slide_vec_b)
                loss = loss_fn(logits.float(), masks)

            bs = imgs.size(0)
            total_samples += bs
            loss_sum += loss.item() * bs

            probs = torch.sigmoid(logits.float())      # [B,1,H,W]
            preds = (probs > threshold).float()        # [B,1,H,W]

            if masks.dim() == 3:
                masks_fg = masks.unsqueeze(1).float()  # [B,1,H,W]
            else:
                masks_fg = masks.float()               # [B,1,H,W]

            preds_flat = preds.view(bs, -1)
            masks_flat = masks_fg.view(bs, -1)

            intersection = (preds_flat * masks_flat).sum(dim=1)             # [B]
            # union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1)          # [B]
            pred_area = preds_flat.sum(dim=1)  # [B]
            gt_area = masks_flat.sum(dim=1)
            dice = (2 * intersection + eps) / (pred_area+ gt_area + eps)                # [B]

            union = pred_area + gt_area - intersection  # [B]
            iou = torch.zeros_like(intersection, dtype=torch.float)

            non_empty = union > 0
            iou[non_empty] = intersection[non_empty] / (union[non_empty] + eps)

            iou[~non_empty] = 1.0

            # Accuracy
            correct = (preds_flat == masks_flat).float().sum(dim=1)        # [B]
            total_pix = preds_flat.size(1)
            acc = correct / max(total_pix, 1)                              # [B]

            dice_sum += dice.mean().item() * bs
            iou_sum  += iou.mean().item() * bs

    metr = {}
    if total_samples > 0:
        metr["Dice"] = dice_sum / total_samples
        metr["IoU"] = iou_sum / total_samples
        metr["val_loss"] = loss_sum / total_samples
    else:
        metr["Dice"] = 0.0
        metr["IoU"] = 0.0
        metr["val_loss"] = 0.0

    return metr
