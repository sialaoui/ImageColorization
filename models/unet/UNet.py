# models/unet/UNet.py
import torch
import torch.nn as nn
from models.base_model import BaseColorizationModel


class UNet(BaseColorizationModel):
    """
    Standard U-Net adapted for grayscale → ab colorization.
    Accepts input channels = 1 (L) or 3 (L + hints/reference a/b).
    Output: (B,2,H,W) with tanh activation in [-1,1]
    """

    def __init__(self, in_ch: int = 1):
        super().__init__()

        # -----------------------
        # Encoder
        # -----------------------
        # allow in_ch to be 1 (default) or 3 (hints/reference)
        self.enc1 = self.block(in_ch,   32)   # H
        self.enc2 = self.block(32,  64)   # H/2
        self.enc3 = self.block(64,  128)  # H/4
        self.enc4 = self.block(128, 256)  # H/8
        self.enc5 = self.block(256, 512)  # H/16
        self.pool = nn.MaxPool2d(2)

        # -----------------------
        # Bottleneck
        # -----------------------
        self.bottleneck = self.block(512, 512)  # H/32

        # -----------------------
        # Decoder
        # -----------------------
        self.up5 = self.up_block(512, 512)   # H/16
        self.up4 = self.up_block(512, 256)   # H/8
        self.up3 = self.up_block(256, 128)   # H/4
        self.up2 = self.up_block(128, 64)    # H/2
        self.up1 = self.up_block(64,  32)    # H

        # Output layer: (a,b)
        self.out_conv = nn.Conv2d(32, 2, kernel_size=1)

    # ------------------------------------------------------------------------
    # U-Net Blocks
    # ------------------------------------------------------------------------
    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------------
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # H
        e2 = self.enc2(self.pool(e1))  # H/2
        e3 = self.enc3(self.pool(e2))  # H/4
        e4 = self.enc4(self.pool(e3))  # H/8
        e5 = self.enc5(self.pool(e4))  # H/16
        e6 = self.pool(e5)             # H/32

        # Bottleneck
        b = self.bottleneck(e6)

        # Decoder with skip connections
        d5 = self.up5(b)   + e5   # H/16
        d4 = self.up4(d5)  + e4   # H/8
        d3 = self.up3(d4)  + e3   # H/4
        d2 = self.up2(d3)  + e2   # H/2
        d1 = self.up1(d2)  + e1   # H

        # Output chrominance
        out = self.out_conv(d1)
        out = torch.tanh(out)     # [-1,1]

        # Validate shapes (optional debugging)
        self.validate_forward(x, out)

        return out