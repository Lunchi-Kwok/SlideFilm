# models/film_decoder.py
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

class SegDecoderWithFiLM(nn.Module):
    def __init__(self, slide_dim=768, in_ch=1536, chs=(768, 384, 192, 96), n_cls=None):
        super().__init__()
        self.films = nn.ModuleList([FiLM(slide_dim, c) for c in (in_ch,)+tuple(chs)])

        self.up1 = Up(in_ch, chs[0])  # 1536 -> 768
        self.up2 = Up(chs[0], chs[1])  # 768  -> 384
        self.up3 = Up(chs[1], chs[2])  # 384  -> 192
        self.up4 = Up(chs[2], chs[3])  # 192  -> 96
        self.head = nn.Conv2d(chs[3], n_cls, 1)

    def forward(self, fmap, slide_vec):
        # fmap: [B, C, h, w]
        f = fmap
        # L0
        g,b = self.films[0](slide_vec); f = apply_film(f, g, b)

        # L1
        f = self.up1(f)
        g,b = self.films[1](slide_vec); f = apply_film(f, g, b)

        # L2
        f = self.up2(f)
        g,b = self.films[2](slide_vec); f = apply_film(f, g, b)

        # L3
        f = self.up3(f)
        g,b = self.films[3](slide_vec); f = apply_film(f, g, b)

        f = self.up4(f)
        g, b = self.films[4](slide_vec);f = apply_film(f, g, b)

        return self.head(f)  # [B, K, H, W]
