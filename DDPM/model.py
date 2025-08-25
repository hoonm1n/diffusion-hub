import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_ch)
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # timestep embedding: MLP output (B, out_ch) → (B, out_ch, 1, 1)
        t_emb_out = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb_out
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.residual_conv(x)



class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, t_emb_dim=512):
        super().__init__()
        self.timestep_embedding = SinusoidalPosEmb(t_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim)
        )

        # encoder
        self.conv_in = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.res1 = ResBlock(base_channels, base_channels, t_emb_dim)
        self.down1 = Downsample(base_channels)
        self.res2 = ResBlock(base_channels, base_channels*2, t_emb_dim)
        self.down2 = Downsample(base_channels*2)
        self.res3 = ResBlock(base_channels*2, base_channels*4, t_emb_dim)
        self.down3 = Downsample(base_channels*4)     
        self.res4 = ResBlock(base_channels*4, base_channels*8, t_emb_dim)  

        # decoder
        self.up1 = Upsample(base_channels*8)  
        self.res5 = ResBlock(base_channels*8, base_channels*4, t_emb_dim)
        self.up2 = Upsample(base_channels*4)
        self.res6 = ResBlock(base_channels * 4, base_channels * 2, t_emb_dim)
        self.up3 = Upsample(base_channels * 2)
        self.res7 = ResBlock(base_channels * 2, base_channels, t_emb_dim)

        self.skip_conv1 = nn.Conv2d(256, 512, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(128, 256, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(512, 1024, kernel_size=1)



        self.conv_out = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        # timestep embedding
        t_emb = self.timestep_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # encoder
        x1 = self.conv_in(x)
        x1 = self.res1(x1, t_emb)
        x2 = self.down1(x1)
        x2 = self.res2(x2, t_emb)
        x3 = self.down2(x2)
        x3 = self.res3(x3, t_emb)
        x4 = self.down3(x3)
        x4 = self.res4(x4, t_emb)

        # decoder
        x5 = self.up1(x4)
        x5 = self.res5(x5 + self.skip_conv3(x3), t_emb)  # skip connection 
        x6 = self.up2(x5)
        x6 = self.res6(x6 + self.skip_conv1(x2), t_emb)  # skip connection 
        x7 = self.up3(x6)
        x7 = self.res7(x7 + self.skip_conv2(x1), t_emb)  # skip connection 

        out = self.conv_out(x7)
        return out