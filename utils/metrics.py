# utils/metrics.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log10

# SSIM
from torchmetrics.functional import structural_similarity_index_measure as tm_ssim

# LPIPS
import lpips

# FID
from torchmetrics.image.fid import FrechetInceptionDistance


# ---------------------------------------------------------
# Basic Error Metrics
# ---------------------------------------------------------
def l1_metric(pred, target):
    """ Mean absolute error """
    return torch.mean(torch.abs(pred - target)).item()


def l2_metric(pred, target):
    """ Mean squared error (L2) """
    return torch.mean((pred - target) ** 2).item()


# ---------------------------------------------------------
# PSNR
# ---------------------------------------------------------
def psnr_metric(pred, target, max_val=1.0):
    """
    pred, target ∈ [0,1], shape (B,3,H,W)
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * log10(max_val ** 2 / mse)


# ---------------------------------------------------------
# SSIM (TorchMetrics implementation)
# ---------------------------------------------------------
def ssim_metric(pred, target):
    """
    pred, target ∈ [0,1], shape (B,3,H,W)
    Returns a scalar (mean SSIM across batch)
    """
    ssim = tm_ssim(pred, target, data_range=1.0)
    return float(ssim.item())


# ---------------------------------------------------------
# LPIPS (Perceptual similarity)
# ---------------------------------------------------------
class LPIPSMetric:
    """
    Wrapper around lpips.LPIPS
    """
    def __init__(self, net="alex", device="cpu"):
        self.lpips_fn = lpips.LPIPS(net=net).to(device)
        self.device = device

    def __call__(self, pred, target):
        """
        pred, target ∈ [0,1], shape (B,3,H,W)
        LPIPS expects [-1,1]
        """
        pred = (pred * 2 - 1).to(self.device)
        target = (target * 2 - 1).to(self.device)
        score = self.lpips_fn(pred, target)
        return float(score.mean().item())


# ---------------------------------------------------------
# FID (Frechet Inception Distance)
# ---------------------------------------------------------
class FIDMetric:
    """
    Computes FID incrementally as you feed batches.

    Usage:
        fid = FIDMetric(device)
        fid.update(pred_rgb, real_rgb)
        final_score = fid.compute()
    """
    def __init__(self, device="cpu"):
        self.metric = FrechetInceptionDistance(feature=2048).to(device)
        self.device = device

    def update(self, pred, target):
        """
        pred, target ∈ [0,1], shape (B,3,H,W)
        Convert to uint8 [0,255]
        """
        pred_uint8 = (pred * 255).byte().to(self.device)
        tgt_uint8  = (target * 255).byte().to(self.device)

        self.metric.update(tgt_uint8, real=True)
        self.metric.update(pred_uint8, real=False)

    def compute(self):
        return float(self.metric.compute().item())


# ---------------------------------------------------------
# High-Level evaluation wrapper for a single model
# ---------------------------------------------------------
def evaluate_model_on_images(rgb_pred, rgb_target, lpips_fn=None, fid_obj=None):
    """
    rgb_pred, rgb_target: (B,3,H,W) in [0,1]

    lpips_fn: LPIPSMetric() or None
    fid_obj : FIDMetric() or None
    """
    results = {
        "L1": l1_metric(rgb_pred, rgb_target),
        "L2": l2_metric(rgb_pred, rgb_target),
        "PSNR": psnr_metric(rgb_pred, rgb_target),
        "SSIM": ssim_metric(rgb_pred, rgb_target),
    }

    if lpips_fn is not None:
        results["LPIPS"] = lpips_fn(rgb_pred, rgb_target)

    if fid_obj is not None:
        fid_obj.update(rgb_pred, rgb_target)

    return results