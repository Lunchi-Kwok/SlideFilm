import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, d_in, c_out, limit=0.5, use_ln=True):
        super().__init__()
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(d_in) if use_ln else nn.Identity()
        self.fc1 = nn.Linear(d_in, c_out)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c_out, 2 * c_out)
        self.limit = limit

        # Let initial behavior ≈ identity
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cond):        # cond: [B, d_in]
        cond = self.ln(cond)
        gb = self.fc2(self.act(self.fc1(cond)))     # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)
        # Limit to avoid releasing features at the beginning
        gamma = torch.tanh(gamma) * self.limit      # [-limit, limit]
        beta  = torch.tanh(beta)  * self.limit
        return gamma, beta

def apply_film(f, gamma, beta):
    # f: [B,C,H,W]; gamma/beta: [B,C]
    return (1 + gamma).unsqueeze(-1).unsqueeze(-2) * f + beta.unsqueeze(-1).unsqueeze(-2)

# Upsample spacial resolution and reduce channel
class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)

class _Fuse(nn.Module):
    """拼接后降维/融合： [B, c_main+c_skip, H, W] -> [B, c_out, H, W]"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out), nn.GELU(),
        )
    def forward(self, x):
        return self.fuse(x)

class SegDecoderWithFiLM(nn.Module):
    """
    ViT 多层 skip（patch fmap） + FiLM 条件化解码器
    - slide_dim: slide 条件向量维度（喂给 FiLM）
    - in_ch: 主干输入通道（ViT hidden dim，如 1536）
    - chs: 四个上采样阶段的输出通道
    - n_cls: 类别数
    - skip_layers: 期望使用的 ViT 层索引（长度=4 对齐 up1..up4）
    - skip_in_ch: 每个 skip fmap 的通道数（ViT-G 为 1536；若换 backbone 需对应修改）
    """
    def __init__(self,
                 slide_dim=768,
                 in_ch=1536,
                 chs=(768, 384, 192, 96),
                 n_cls=None,
                 skip_layers=(5, 11, 23, 39),
                 skip_in_ch=1536):
        super().__init__()
        assert len(skip_layers) == 4
        self.skip_layers = tuple(skip_layers)

        # FiLM：输入一层 + 四个阶段各一层（共5个）
        self.films = nn.ModuleList([FiLM(slide_dim, c) for c in (in_ch,)+tuple(chs)])

        # 主干上采样（不改你的 Up）
        self.up1 = Up(in_ch,   chs[0])  # 1536 -> 768
        self.up2 = Up(chs[0],  chs[1])  # 768  -> 384
        self.up3 = Up(chs[1],  chs[2])  # 384  -> 192
        self.up4 = Up(chs[2],  chs[3])  # 192  -> 96

        # 将 ViT 的 skip fmap（1536）投影到各阶段通道
        self.skip_proj = nn.ModuleList([
            nn.Conv2d(skip_in_ch, chs[0], 1, bias=False),  # for up1
            nn.Conv2d(skip_in_ch, chs[1], 1, bias=False),  # for up2
            nn.Conv2d(skip_in_ch, chs[2], 1, bias=False),  # for up3
            nn.Conv2d(skip_in_ch, chs[3], 1, bias=False),  # for up4
        ])

        # 拼接后的融合（把 [stage, skip] 两路合成 stage 通道）
        self.fuse1 = _Fuse(chs[0] + chs[0], chs[0])
        self.fuse2 = _Fuse(chs[1] + chs[1], chs[1])
        self.fuse3 = _Fuse(chs[2] + chs[2], chs[2])
        self.fuse4 = _Fuse(chs[3] + chs[3], chs[3])

        self.head = nn.Conv2d(chs[3], n_cls, 1)

    def _get_skip(self, idx, skip_fmaps):
        """取出并投影第 idx 个阶段的 skip；若缺失则返回 None"""
        if skip_fmaps is None:
            return None
        key = self.skip_layers[idx]
        if key not in skip_fmaps:
            return None
        s = self.skip_proj[idx](skip_fmaps[key])  # 通道对齐
        return s

    def _fuse(self, x, s, fuse_block):
        if s is None:
            return x
        if s.shape[-2:] != x.shape[-2:]:
            s = F.interpolate(s, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return fuse_block(torch.cat([x, s], dim=1))

    def forward(self, fmap, slide_vec, skip_fmaps=None):
        """
        Args:
            fmap: [B, in_ch, h, w]      # 最深层 ViT fmap（如最后一层）
            slide_vec: [B, slide_dim]   # slide 级条件向量
            skip_fmaps: dict[int -> Tensor]，键为 ViT 层号，值形状 [B, skip_in_ch, h, w]
        """
        f = fmap

        # L0：输入先做一次 FiLM
        g, b = self.films[0](slide_vec); f = apply_film(f, g, b)

        # Stage 1
        f = self.up1(f)
        s1 = self._get_skip(0, skip_fmaps)
        f = self._fuse(f, s1, self.fuse1)
        g, b = self.films[1](slide_vec); f = apply_film(f, g, b)

        # Stage 2
        f = self.up2(f)
        s2 = self._get_skip(1, skip_fmaps)
        f = self._fuse(f, s2, self.fuse2)
        g, b = self.films[2](slide_vec); f = apply_film(f, g, b)

        # Stage 3
        f = self.up3(f)
        s3 = self._get_skip(2, skip_fmaps)
        f = self._fuse(f, s3, self.fuse3)
        g, b = self.films[3](slide_vec); f = apply_film(f, g, b)

        # Stage 4
        f = self.up4(f)
        s4 = self._get_skip(3, skip_fmaps)
        f = self._fuse(f, s4, self.fuse4)
        g, b = self.films[4](slide_vec); f = apply_film(f, g, b)

        return self.head(f)  # [B, n_cls, H_out, W_out]
