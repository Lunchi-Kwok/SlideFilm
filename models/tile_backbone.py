import math
import torch
import torch.nn as nn
from gigapath.pipeline import TileEncodingDataset

class TileBackbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        feats = {}

        def hook(module, inp, out):
            feats["tokens_pre_norm"] = out  # [B, 1+N, C]

        handle = self.base.blocks[-1].register_forward_hook(hook)
        with torch.no_grad():
            _ = self.base.forward_features(x)
        handle.remove()

        tokens = feats["tokens_pre_norm"]
        if hasattr(self.base, "norm") and self.base.norm is not None:
            tokens = self.base.norm(tokens)

        cls_token = tokens[:, 0]      # [B, C]
        patch_tokens = tokens[:, 1:]  # [B, N, C]

        B, N, C = patch_tokens.shape
        h = w = int(math.sqrt(N))

        fmap = patch_tokens.transpose(1, 2).reshape(B, C, h, w)

        return cls_token, fmap
