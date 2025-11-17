# engines/train_seg.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from gigapath.pipeline import run_inference_with_slide_encoder
import torch.nn.functional as F

def bg_ratio_step(epoch, steps=(5, 15), values=(0.6, 0.3, 0.1)):
    if epoch < steps[0]: return values[0]
    if epoch < steps[1]: return values[1]
    return values[2]

def bg_ratio_cosine(epoch, T, start=0.6, end=0.1):
    t = min(max(epoch / max(T-1, 1), 0.0), 1.0)
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))

class CE_DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        # logits: [B, K, H, W], target: [B, H, W]
        ce = self.ce(logits, target)
        with torch.no_grad():
            onehot = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1)
        probs = logits.softmax(1)
        inter = (probs*onehot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + onehot.sum(dim=(2,3))
        dice = 1 - (2*inter+1)/(union+1)
        return ce + dice.mean()

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction="mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)[:, 1, ...]        # [B,H,W]
        target_fg = (target == 1).float()                      # [B,H,W]

        dims = (1, 2)  # H,W
        intersection = (probs * target_fg).sum(dim=dims)
        union = probs.sum(dim=dims) + target_fg.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score  # [B]

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        """
        logits: [B, C, H, W] (raw, no softmax)
        target: [B, H, W] (long, class index 0..C-1)
        """
        ce_loss = F.cross_entropy(logits, target, reduction="none")  # [B,H,W]
        pt = torch.exp(-ce_loss)  # = softmax prob of true class

        focal = (self.alpha * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_dice=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        self.dice  = BinaryDiceLoss(smooth=1.0, reduction="mean")
        self.lambda_dice = lambda_dice

    def forward(self, logits, target):
        focal_loss = self.focal(logits, target)
        dice_loss  = self.dice(logits, target)
        return focal_loss + self.lambda_dice * dice_loss

def freeze(module):
    for p in module.parameters(): p.requires_grad = False

def unfreeze(module):
    for p in module.parameters(): p.requires_grad = True

def train_one_epoch(Dataloaders, tile_backbone, decoder, optim, scaler, loss_fn, slide_encoder, bg_keep_ratio):
    decoder.train()
    # slide_encoder.train()
    # tile_backbone.train()
    total = 0; loss_sum = 0

    for batch_ in Dataloaders:
        tile_loader = DataLoader(batch_, batch_size=32, shuffle=False, num_workers=4)

        collated_outputs = {'tile_embeds': [], 'coords': []}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for batch in tile_loader:
                out = tile_backbone(batch['img'].cuda())
                cls_tok = out[0]                       # [B, C]
                collated_outputs['tile_embeds'].append(cls_tok.detach().cpu())
                collated_outputs['coords'].append(batch['coords'])
        tile_encoder_outputs = {k: torch.cat(v) for k, v in collated_outputs.items()}
        slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder, **tile_encoder_outputs)
        slide_vec = slide_embeds["last_layer_embed"].to(next(decoder.parameters()).device)  # [D]

        for tile_batch in tile_loader:
            imgs  = tile_batch['img'].cuda()     # [B,C,H,W]
            masks = tile_batch['mask'].cuda()    # [B,H,W]
            
            non_empty = (masks.view(masks.size(0), -1).sum(dim=1) > 0)
            empty = ~non_empty
            if empty.any():
                keep_empty = (torch.rand(empty.sum(), device=masks.device) < bg_keep_ratio)
                keep_mask = non_empty.clone()
                keep_mask[empty] = keep_empty
                if not keep_mask.any():
                    continue
                imgs = imgs[keep_mask]
                masks = masks[keep_mask]

            with torch.no_grad():
                out = tile_backbone(imgs)
                if len(out) == 2:
                    _, fmap = out
                    skip_fmaps = None
                else:
                    _, fmap, _, fmap_dict = out
                    skip_fmaps = {li: fmap_dict[li] for li in getattr(decoder, "skip_layers", []) if li in fmap_dict}

            if slide_vec.dim() == 1:
                slide_vec_b = slide_vec.unsqueeze(0).expand(imgs.size(0), -1)  # [B, D]
            else:
                slide_vec_b = slide_vec

            optim.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                logits = decoder(fmap, slide_vec_b, skip_fmaps) if skip_fmaps is not None else decoder(fmap, slide_vec_b)

            assert logits.shape[-2:] == masks.shape[-2:], f"HW: {logits.shape} vs {masks.shape}"
            assert masks.dtype == torch.long
            loss = loss_fn(logits.float(), masks)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total += imgs.size(0)
            loss_sum += loss.item() * imgs.size(0)

    return loss_sum / total
