# test.py
import os
import json
import torch
import argparse
from tqdm import tqdm

from utils.data import get_dataloaders, lab_to_rgb_batch
from utils.metrics import (
    l1_metric, l2_metric, psnr_metric, ssim_metric,
    LPIPSMetric, FIDMetric
)

# Import model registry or specific models
from models.unet import UNet
from models.resnet_unet import ResNet_UNet



# ------------------------------------------------------------
# Model registry for easy loading
# ------------------------------------------------------------
MODEL_REGISTRY = {
    "unet": UNet,
    "resnet_unet": ResNet_UNet,
}

# ------------------------------------------------------------
# Main test function
# ------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    _, _, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    if args.arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model = MODEL_REGISTRY[args.arch]().to(device)

    # Load best checkpoint by default
    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            "experiments", args.arch, "checkpoints", "best_model_" + args.loss + ".pth"
        )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # --------------------------------------------------------
    # Metrics setup
    # --------------------------------------------------------
    slow_lpips = None
    slow_fid = None

    if args.use_slow_metrics:
        slow_lpips = LPIPSMetric(device=device)
        slow_fid = FIDMetric(device=device)

    # Aggregation
    agg = {
        "L1": [],
        "L2": [],
        "PSNR": [],
        "SSIM": [],
    }
    if args.use_slow_metrics:
        agg["LPIPS"] = []
        # FID is computed at the end (single value)

    # Create results dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Optionally save predictions
    save_pred_dir = None
    if args.save_images:
        save_pred_dir = os.path.join(args.output_dir, "predictions")
        os.makedirs(save_pred_dir, exist_ok=True)

    # --------------------------------------------------------
    # Loop on test set
    # --------------------------------------------------------
    with torch.no_grad():
        for idx, (L, ab_gt) in enumerate(tqdm(test_loader)):

            L = L.to(device)
            ab_gt = ab_gt.to(device)

            ab_pred = model(L)

            # Convert to RGB for evaluation
            rgb_pred = lab_to_rgb_batch(L, ab_pred).to(device)
            rgb_gt   = lab_to_rgb_batch(L, ab_gt).to(device)

            # Fast metrics
            agg["L1"].append(l1_metric(rgb_pred, rgb_gt))
            agg["L2"].append(l2_metric(rgb_pred, rgb_gt))
            agg["PSNR"].append(psnr_metric(rgb_pred, rgb_gt))
            agg["SSIM"].append(ssim_metric(rgb_pred, rgb_gt))

            # Slow metrics
            if args.use_slow_metrics:
                agg["LPIPS"].append(slow_lpips(rgb_pred.float(), rgb_gt.float()))
                slow_fid.update(rgb_pred, rgb_gt)

            # Save predictions as images
            if args.save_images:
                for b in range(rgb_pred.size(0)):
                    out_path = os.path.join(save_pred_dir, f"sample_{idx}_{b}.png")
                    img = rgb_pred[b].cpu().permute(1,2,0).numpy()
                    img = (img * 255).astype("uint8")
                    from PIL import Image
                    Image.fromarray(img).save(out_path)

    # --------------------------------------------------------
    # Final aggregation
    # --------------------------------------------------------
    final_metrics = {
        "L1": float(sum(agg["L1"]) / len(agg["L1"])),
        "L2": float(sum(agg["L2"]) / len(agg["L2"])),
        "PSNR": float(sum(agg["PSNR"]) / len(agg["PSNR"])),
        "SSIM": float(sum(agg["SSIM"]) / len(agg["SSIM"])),
    }

    if args.use_slow_metrics:
        final_metrics["LPIPS"] = float(sum(agg["LPIPS"]) / len(agg["LPIPS"]))
        final_metrics["FID"]   = slow_fid.compute()

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------
    out_json = os.path.join(args.output_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("\nSaved results to:", out_json)
    print("\nFinal Metrics:")
    print(json.dumps(final_metrics, indent=4))


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)

    parser.add_argument(
        "--arch", type=str, default="unet",
        help="Model architecture"
    )

    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a custom checkpoint (default: best model)"
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument(
        "--use_slow_metrics",
        action="store_true",
        help="Enable LPIPS and FID (slow)"
    )

    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save predicted RGB images"
    )

    parser.add_argument(
        "--output_dir", type=str,
        default="experiments/results_test",
        help="Where to save metrics & predictions"
    )

    args = parser.parse_args()
    main(args)