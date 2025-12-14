# visualize.py
import os
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from utils.data import RGB2LabDataset, lab_to_rgb
from models.unet import UNet
from models.resnet_unet import ResNet_UNet


# ---------------------------------------------------------
# Model registry
# ---------------------------------------------------------
MODEL_REGISTRY = {
    "unet": UNet,
    "resnet_unet": ResNet_UNet,
}

# ---------------------------------------------------------
# Combine L + Pred + GT into a single image
# ---------------------------------------------------------
def make_triplet_image(L, pred_rgb, gt_rgb):
    # --- Fix shapes ---------------------------------------------------------
    L = np.squeeze(L)            # remove batch/channel dims
    pred_rgb = np.squeeze(pred_rgb)
    gt_rgb  = np.squeeze(gt_rgb)

    # --- Fix L channel ------------------------------------------------------
    if L.ndim == 2:              # correct
        L_img = (L * 255).astype("uint8")
    else:
        raise ValueError(f"L has wrong shape: {L.shape}")

    L_img = np.repeat(L_img[..., None], 3, axis=2)

    # --- Fix RGB types ------------------------------------------------------
    if pred_rgb.dtype != np.uint8:
        pred_rgb = np.clip(pred_rgb, 0, 1)
        pred_rgb = (pred_rgb * 255).astype(np.uint8)

    if gt_rgb.dtype != np.uint8:
        gt_rgb = np.clip(gt_rgb, 0, 1)
        gt_rgb = (gt_rgb * 255).astype(np.uint8)

    # --- Final concatenation ------------------------------------------------
    triplet = np.concatenate([L_img, pred_rgb, gt_rgb], axis=1)

    # ---- Sanity checks -----------------------------------------------------
    if triplet.dtype != np.uint8:
        raise ValueError("Triplet must be uint8")
    if triplet.ndim != 3 or triplet.shape[2] != 3:
        raise ValueError(f"Triplet has wrong shape: {triplet.shape}")

    return Image.fromarray(triplet)



# ---------------------------------------------------------
# Main visualization script
# ---------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Load dataset split
    # ---------------------------
    data_dir = os.path.join(args.data_root, args.split)
    dataset = RGB2LabDataset(data_dir, image_size=args.image_size)

    # Random sample indices
    indices = random.sample(range(len(dataset)), args.num_images)

    # ---------------------------
    # Load model
    # ---------------------------
    if args.arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model = MODEL_REGISTRY[args.arch]().to(device)

    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            "experiments", args.arch, "checkpoints", "best_model.pth"
        )

    print("Loading checkpoint:", args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------
    # Generate visualizations
    # ---------------------------
    for i, idx in enumerate(tqdm(indices, desc="Generating samples")):
        L, ab_gt = dataset[idx]
        L = L.unsqueeze(0).to(device)
        ab_gt = ab_gt.unsqueeze(0).to(device)

        with torch.no_grad():
            ab_pred = model(L)

        # Convert all to RGB
        L_np = L[0, 0].cpu().numpy()
        pred_rgb_np = lab_to_rgb(L[0], ab_pred[0])
        gt_rgb_np   = lab_to_rgb(L[0], ab_gt[0])

        # Create triplet image
        out_img = make_triplet_image(L_np, pred_rgb_np, gt_rgb_np)

        out_path = os.path.join(args.output_dir, f"sample_{i}.png")
        out_img.save(out_path)

    print("\nSaved visualizations in:", args.output_dir)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--arch", type=str, default="unet")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument(
        "--output_dir", type=str,
        default="experiments/visualizations",
        help="Directory where visualizations will be saved"
    )

    args = parser.parse_args()
    main(args)