import torch.nn as nn
import torch
import torch.nn.functional as F


class PixelWiseGatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        # feat1, feat2: [B, C, H, W]
        fused = torch.cat([feat1, feat2], dim=1)  
        alpha = self.gate_conv(fused)             
        return alpha * feat1 + (1 - alpha) * feat2

