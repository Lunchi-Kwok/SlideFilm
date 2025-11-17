import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from gigapath.pipeline import run_inference_with_slide_encoder

@torch.no_grad()
def val_one_epoch(Dataloaders, tile_backbone, decoder, loss_fn, slide_encoder, num_classes):
    decoder.eval()
    # slide_encoder.train()
    # tile_backbone.train()
    total, loss_sum = 0, 0
    conf_total = torch.zeros(num_classes, num_classes, dtype=torch.int64, device="cpu")

    for batch_ in Dataloaders:
        tile_loader = DataLoader(batch_, batch_size=8, shuffle=False, num_workers=0)

        collated_outputs = {'tile_embeds': [], 'coords': []}
        with autocast(dtype=torch.float16):
            for batch in tile_loader:
                out = tile_backbone(batch['img'].cuda())
                cls_tok = out[0]  
                collated_outputs['tile_embeds'].append(cls_tok.detach().cpu())
                collated_outputs['coords'].append(batch['coords'])
        tile_encoder_outputs = {k: torch.cat(v) for k, v in collated_outputs.items()}
        slide_embeds = run_inference_with_slide_encoder(
            slide_encoder_model=slide_encoder, **tile_encoder_outputs
        )
        slide_vec = slide_embeds["last_layer_embed"].to(next(decoder.parameters()).device)

        for tile_batch in tile_loader:
            imgs  = tile_batch['img'].cuda()
            masks = tile_batch['mask'].cuda()
            
            # non_empty_idx = (masks.view(masks.size(0), -1).sum(dim=1) > 0)
            # if not non_empty_idx.any():
            #     continue 

            # imgs = imgs[non_empty_idx]
            # masks = masks[non_empty_idx]

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
            total += bs
            loss_sum += loss.item() * bs

            preds = logits.argmax(1)
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

    union = pos_gt + pos_pred - tp
    iou = tp.float() / union.clamp_min(1)
    f1 = 2 * tp.float() / (pos_gt + pos_pred).clamp_min(1)

    miou = iou.mean().item()
    mf1 = f1.mean().item()
    acc = (tp.sum().float() / conf.sum().clamp_min(1)).item()

    return {"mIoU": miou, "F1": mf1, "Acc": acc}

