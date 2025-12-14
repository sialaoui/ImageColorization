# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Perceptual Color Loss
# ------------------------------------------------------------
class MagicalLoss(nn.Module):
    """
    Combines:
    1) SmoothL1 reconstruction on (a,b)
    2) L1 on magnitude (saturation)
    3) L1 on variance of chrominance channels
    """
    def __init__(self, w_recon=1.0, w_mag=0.3, w_var=0.1):
        super().__init__()
        self.w_recon = w_recon
        self.w_mag = w_mag
        self.w_var = w_var

    def forward(self, pred_ab, target_ab):
        # 1) Reconstruction
        recon = F.smooth_l1_loss(pred_ab, target_ab)

        # 2) Magnitude (chroma intensity)
        pred_mag = torch.sqrt(pred_ab[:, 0:1]**2 + pred_ab[:, 1:2]**2 + 1e-6)
        tgt_mag  = torch.sqrt(target_ab[:, 0:1]**2 + target_ab[:, 1:2]**2 + 1e-6)
        mag_loss = F.l1_loss(pred_mag, tgt_mag)

        # 3) Variance alignment
        pred_flat = pred_ab.view(pred_ab.size(0), 2, -1)
        tgt_flat  = target_ab.view(target_ab.size(0), 2, -1)
        pred_std  = pred_flat.std(dim=2)
        tgt_std   = tgt_flat.std(dim=2)
        var_loss  = F.l1_loss(pred_std, tgt_std)

        total = (
            self.w_recon * recon +
            self.w_mag   * mag_loss +
            self.w_var   * var_loss
        )

        details = {
            "recon": recon.item(),
            "mag":   mag_loss.item(),
            "var":   var_loss.item(),
        }
        return total, details
    
################### colorLoss ###############################################
class ColorLoss(nn.Module):
    """
    Loss pour colorisation:
    1) reconstruction (SmoothL1) sur (a,b)
    2) alignement de la magnitude (saturation) |ab|
    3) Total Variation (TV) pour lisser les couleurs
    """
    def __init__(self, w_recon=1.0, w_mag=0.15, w_tv=0.01):
        super().__init__()
        self.w_recon = w_recon
        self.w_mag   = w_mag
        self.w_tv    = w_tv

    def tv_loss(self, ab):
        """
        Total Variation loss sur les canaux ab pour réduire le bruit.
        ab : (B,2,H,W)
        """
        diff_h = ab[:, :, 1:, :] - ab[:, :, :-1, :]
        diff_w = ab[:, :, :, 1:] - ab[:, :, :, :-1]
        return diff_h.abs().mean() + diff_w.abs().mean()

    def forward(self, pred_ab, target_ab):
        # 1) reconstruction
        recon = F.smooth_l1_loss(pred_ab, target_ab)

        # 2) magnitude (= saturation)
        pred_mag = torch.sqrt(pred_ab[:, 0:1]**2 + pred_ab[:, 1:2]**2 + 1e-6)
        tgt_mag  = torch.sqrt(target_ab[:, 0:1]**2 + target_ab[:, 1:2]**2 + 1e-6)
        mag_loss = F.l1_loss(pred_mag, tgt_mag)

        # 3) TV loss pour lisser les couleurs
        tv = self.tv_loss(pred_ab)

        total = (
            self.w_recon * recon +
            self.w_mag   * mag_loss +
            self.w_tv    * tv
        )

        details = {
            "recon": recon.item(),
            "mag":   mag_loss.item(),
            "tv":    tv.item(),
        }
        return total, details


# ------------------------------------------------------------
# Basic Losses (simple PyTorch wrappers)
# ------------------------------------------------------------
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target), {}   # keep same API as MagicalLoss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target), {}


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target), {}