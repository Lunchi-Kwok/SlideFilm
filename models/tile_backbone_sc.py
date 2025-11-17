import math
import torch
import torch.nn as nn
from gigapath.pipeline import TileEncodingDataset

class TileBackbone(nn.Module):
    def __init__(self, base_model, layers=None):
        """
        Args:
            base_model: DINOv2-ViT 模型
            layers: 可选，想要额外提取的 block 索引列表/元组。
                    - None: 与原版一致，只取最后一层
                    - [5,11,23,39] 等：返回这些层的 fmap/cls 字典
        """
        super().__init__()
        self.base = base_model
        if layers is not None:
            if isinstance(layers, int):
                layers = [layers]
            self.layers = sorted(set(layers))
        else:
            self.layers = None

    def forward(self, x):
        feats = {}
        handles = []

        last_idx = len(self.base.blocks) - 1

        # 注册 hook
        if self.layers is None:
            def hook(module, inp, out):
                feats[last_idx] = out  # [B, 1+N, C]
            handles.append(self.base.blocks[last_idx].register_forward_hook(hook))
        else:
            for li in self.layers:
                def _make_hook(i):
                    def h(module, inp, out):
                        feats[i] = out
                    return h
                handles.append(self.base.blocks[li].register_forward_hook(_make_hook(li)))
            # 确保最后一层也取到（保持与原返回兼容）
            if last_idx not in feats:
                def hook_last(module, inp, out):
                    feats[last_idx] = out
                handles.append(self.base.blocks[last_idx].register_forward_hook(hook_last))

        with torch.no_grad():
            _ = self.base.forward_features(x)

        for h in handles:
            h.remove()

        # 只取最后一层（与原版一致）
        tokens_last = feats[last_idx]
        if hasattr(self.base, "norm") and self.base.norm is not None:
            tokens_last = self.base.norm(tokens_last)

        cls_token = tokens_last[:, 0]       # [B, C]
        patch_tokens = tokens_last[:, 1:]   # [B, N, C]
        B, N, C = patch_tokens.shape
        h = w = int(math.sqrt(N))
        fmap = patch_tokens.transpose(1, 2).reshape(B, C, h, w)  # [B, C, H, W]

        # 如果没有指定 layers，直接保持原有返回
        if self.layers is None:
            return cls_token, fmap

        # 否则，额外组装各层字典
        cls_dict, fmap_dict = {}, {}
        for li in self.layers:
            tokens = feats[li]                      # [B, 1+N, C]
            # 只有最后一层过 norm（与 ViT 默认一致）
            if li == last_idx and hasattr(self.base, "norm") and self.base.norm is not None:
                tokens = self.base.norm(tokens)

            cls_i = tokens[:, 0]                   # [B, C]
            p_i = tokens[:, 1:]                    # [B, N, C]
            B_i, N_i, C_i = p_i.shape
            s = int(math.sqrt(N_i))
            fmap_i = p_i.transpose(1, 2).reshape(B_i, C_i, s, s)

            cls_dict[li] = cls_i
            fmap_dict[li] = fmap_i

        # 返回：最后一层（保持兼容）+ 各层字典（用于 skip）
        return cls_token, fmap, cls_dict, fmap_dict
