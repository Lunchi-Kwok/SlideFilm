# engines/train_seg.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from gigapath.pipeline import run_inference_with_slide_encoder
import torch.nn.functional as F

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

def freeze(module):
    for p in module.parameters(): p.requires_grad = False

def unfreeze(module):
    for p in module.parameters(): p.requires_grad = True

def train_one_epoch(Dataloaders, tile_backbone, decoder, optim, scaler, loss_fn, slide_encoder):
    decoder.train()
    total = 0; loss_sum = 0
    #load slides(each loop load 1 slide)
    for batch_ in Dataloaders:
        # tile_dataset = batch[0]        # [B,3,224,224]

        #load tiles within this slide
        tile_loader = DataLoader(
            batch_, batch_size=32, shuffle=False, num_workers=4
        )

        #extract slide embedding
        collated_outputs = {'tile_embeds': [], 'coords': []}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for batch in tile_loader:
                cls_tok, _ = tile_backbone(batch['img'].cuda())
                collated_outputs['tile_embeds'].append(cls_tok.detach().cpu())
                collated_outputs['coords'].append(batch['coords'])
        tile_encoder_outputs = {k: torch.cat(v) for k, v in collated_outputs.items()}
        slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder, **tile_encoder_outputs)

        #train decoder
        for tile_batch in tile_loader:
            imgs = tile_batch['img'].cuda()  # [B, C, H, W]
            masks = tile_batch['mask'].cuda()
            with torch.no_grad():
                _, fmap = tile_backbone(imgs)

            optim.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                logits = decoder(fmap, slide_embeds["last_layer_embed"])
            assert logits.shape[-2:] == masks.shape[-2:], f"HW: {logits.shape} vs {masks.shape}"
            assert masks.dtype == torch.long
            loss = loss_fn(logits.float(), masks)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total += imgs.size(0)
            loss_sum += loss.item() * imgs.size(0)

    return loss_sum/total
