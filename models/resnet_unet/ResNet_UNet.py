# models/resnet_unet.py
import torch
import torch.nn as nn
import torchvision.models as models
from models.base_model import BaseColorizationModel


class ResNet_UNet(BaseColorizationModel):
    def __init__(self, in_ch: int = 1, resnet_weights=None):
        """
        in_ch: number of input channels for the U-Net path (1 for L, 3 for L+hints or L+reference)
        resnet_weights: passed to torchvision.models.resnet34 weights arg (use None to avoid downloads)
        """
        super().__init__()

        # ResNet backbone (encoder)
        resnet = models.resnet34(weights=resnet_weights if resnet_weights is not None else models.ResNet34_Weights.DEFAULT)
        self.resnet_conv = nn.Sequential(*list(resnet.children())[:-2])  # (B,512,H/32,W/32)
        for p in self.resnet_conv.parameters():
            p.requires_grad = False

        # U-Net encoder (first block takes in_ch channels)
        self.enc1 = self.block(in_ch,   32)
        self.enc2 = self.block(32,  64)
        self.enc3 = self.block(64,  128)
        self.enc4 = self.block(128, 256)
        self.enc5 = self.block(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Fusion
        self.fuse = nn.Conv2d(512 + 512, 512, kernel_size=1)

        # Decoder
        self.up5 = self.up_block(512, 512)
        self.up4 = self.up_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.up2 = self.up_block(128, 64)
        self.up1 = self.up_block(64,  32)

        self.out_conv = nn.Conv2d(32, 2, kernel_size=1)

    # ------------------------------------------------------
    # Blocks
    # ------------------------------------------------------
    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x):
        # U-Net encoder
        # If input contains hints/reference channels (3), the U-Net encoder consumes them directly.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.pool(e5)

        # ResNet encoder on fake RGB
        # The ResNet backbone expects an RGB-like tensor derived from the L channel only.
        # If x has 3 channels (L, a_hint, b_hint) we extract the L channel.
        L_channel = x[:, 0:1, ...]
        x_res = self.l_to_rgb(L_channel)
        res = self.resnet_conv(x_res)  # (B,512,H/32,W/32)

        # Fusion
        fused = torch.cat([e6, res], dim=1)
        bottleneck = self.fuse(fused)

        # Decoder
        d5 = self.up5(bottleneck) + e5
        d4 = self.up4(d5)        + e4
        d3 = self.up3(d4)        + e3
        d2 = self.up2(d3)        + e2
        d1 = self.up1(d2)        + e1

        out = self.out_conv(d1)        # (B,2,H,W)
        out = torch.tanh(out)          # enforce [-1,1]

        # Validate shapes (optional, can remove once stable)
        self.validate_forward(x, out)

        return out
    
    def l_to_rgb(self, L):
        """
        Convert a single-channel L tensor in [0,1] to an RGB tensor in [0,1]
        by assuming zero chroma (a=b=0) and inverting the Lab → sRGB transform.
        """

        # Scale L back to [0,100]
        L_scaled = L * 100.0
        fy = (L_scaled + 16.0) / 116.0
        fx = fy
        fz = fy

        delta = 6.0 / 29.0
        delta_sq = delta * delta

        def f_inv(t):
            return torch.where(t > delta, t ** 3, 3 * delta_sq * (t - 4.0 / 29.0))

        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        X = Xn * f_inv(fx)
        Y = Yn * f_inv(fy)
        Z = Zn * f_inv(fz)

        r = 3.2406 * X + -1.5372 * Y + -0.4986 * Z
        g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        b = 0.0557 * X + -0.2040 * Y + 1.0570 * Z

        # r,g,b currently have shape (B,1,H,W) because input L has a channel dim.
        # Remove that singleton channel so stacking produces (B,3,H,W).
        if r.dim() == 4 and r.size(1) == 1:
            r = r.squeeze(1)
            g = g.squeeze(1)
            b = b.squeeze(1)

        rgb = torch.stack([r, g, b], dim=1)
        rgb = torch.clamp(rgb, min=0.0)
        rgb = torch.where(
            rgb > 0.0031308, 1.055 * torch.pow(rgb, 1.0 / 2.4) - 0.055, 12.92 * rgb
        )
        return torch.clamp(rgb, 0.0, 1.0)
