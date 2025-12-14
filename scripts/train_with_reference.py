import os
import numpy as np
from PIL import Image

import random

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T

from models.resnet_unet import ResNet_UNet
from skimage.color import rgb2lab

import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import glob

from utils.data import lab_to_rgb
from utils.losses import MagicalLoss


# -------------------------------------------------------------------
# CONFIG GLOBALE
# -------------------------------------------------------------------
IMAGE_SIZE = 256

BATCH_SIZE   = 128
NUM_EPOCHS   = 80
LR           = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS  = 0

def build_pairs(file_list):
    class_to_images = {}

    # Regrouper les images par classe
    for path in file_list:
        label = os.path.basename(os.path.dirname(path))
        class_to_images.setdefault(label, []).append(path)

    pairs = []

    for label, imgs in class_to_images.items():
        # tri pour un ordre stable
        imgs = sorted(imgs)

        # on parcourt 2 par 2
        for i in range(0, len(imgs) - 1, 2):
            a = imgs[i]
            b = imgs[i+1]

            # ajouter les deux sens
            pairs.append((a, b))
            pairs.append((b, a))

    return pairs


# -------------------------------------------------------------------
# DATASET : RGB -> Lab + HINTS SUR ab
# -------------------------------------------------------------------
class RGB2LabPairDataset(Dataset):
    """
    Dataset pour pairs (target, reference).

    Retourne :
      x_input : (3,H,W) = [ L_target , a_ref , b_ref ]
      ab_target : (2,H,W) = ground truth target
      ref_rgb : (H,W,3) image référence RGB (pour visualisation)
    """
    def __init__(self, pairs, image_size=256):
        """
        pairs = liste [ (target_path, reference_path), ... ]
        """
        self.pairs = pairs
        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target_path, ref_path = self.pairs[idx]

        # Load target
        img_t = Image.open(target_path).convert("RGB")
        x_t = self.transform(img_t)
        x_t_np = x_t.permute(1,2,0).numpy().astype("float32")

        # Load reference
        img_r = Image.open(ref_path).convert("RGB")
        x_r = self.transform(img_r)
        x_r_np = x_r.permute(1,2,0).numpy().astype("float32")

        # LAB
        lab_t = rgb2lab(x_t_np).astype("float32")
        lab_r = rgb2lab(x_r_np).astype("float32")

        # Target LAB
        L_t = lab_t[...,0] / 100.0
        ab_t = lab_t[...,1:] / 128.0

        # Reference AB
        ab_r = lab_r[...,1:] / 128.0

        # Tensor
        L_t  = torch.from_numpy(L_t).unsqueeze(0)             # (1,H,W)
        ab_t = torch.from_numpy(ab_t).permute(2,0,1)          # (2,H,W)
        ab_r = torch.from_numpy(ab_r).permute(2,0,1)          # (2,H,W)

        # Input réseau
        x_input = torch.cat([L_t, ab_r], dim=0)               # (3,H,W)

        # ref_rgb pour affichage
        ref_rgb = x_r_np.copy()

        return x_input, ab_t, ref_rgb

# -------------------------------------------------------------------
# FONCTIONS TRAIN / EVAL
# -------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, log_details=False):
    model.train()
    running_loss = 0.0
    details_acc = {"recon": 0.0, "mag": 0.0, "var": 0.0}

    for x_in, ab, _ in loader:
        x_in = x_in.to(device)   # (B,3,H,W)
        ab   = ab.to(device)     # (B,2,H,W)

        optimizer.zero_grad()
        pred_ab = model(x_in)

        loss, loss_details = criterion(pred_ab, ab)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if log_details:
            for k in details_acc:
                details_acc[k] += loss_details[k]

    avg_loss = running_loss / len(loader)
    if log_details:
        for k in details_acc:
            details_acc[k] /= len(loader)
        return avg_loss, details_acc
    else:
        return avg_loss


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x_in, ab, _  in loader:
            x_in = x_in.to(device)
            ab   = ab.to(device)
            pred_ab = model(x_in)

            loss, _ = criterion(pred_ab, ab)
            running_loss += loss.item()

    return running_loss / len(loader)


# -------------------------------------------------------------------
# SAUVEGARDE D'EXEMPLES (L / GT / PRED / PRED_enh)
# -------------------------------------------------------------------
def save_examples(model, loader, device, out_dir,
                  split_name="train", max_images=20):

    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for batch in loader:

            # batch contient : (x_input, ab_target, ref_rgb)
            x_batch, ab_batch, ref_rgb_batch = batch

            x_batch  = x_batch.to(device)
            ab_batch = ab_batch.to(device)

            pred_ab = model(x_batch)

            for i in range(x_batch.size(0)):
                if saved >= max_images:
                    return

                # ------------- INPUTS -------------
                x_in   = x_batch[i]     # (3,H,W)
                L      = x_in[0:1]      # (1,H,W)
                ab_gt  = ab_batch[i]
                ab_pred = pred_ab[i]

                # ------------- REFERENCE RGB -------------
                ref_rgb = ref_rgb_batch[i].numpy()  # (H,W,3)

                # Convert ref RGB → LAB
                lab_ref = rgb2lab(ref_rgb).astype("float32")
                L_r  = torch.from_numpy(lab_ref[...,0] / 100.0).unsqueeze(0)
                ab_r = torch.from_numpy(lab_ref[...,1:] / 128.0).permute(2,0,1)

                # ------------- CONVERSIONS -------------
                gray     = L[0].cpu().numpy()
                rgb_ref  = lab_to_rgb(L_r, ab_r)
                rgb_gt   = lab_to_rgb(L, ab_gt)
                rgb_pred = lab_to_rgb(L, ab_pred)

                # ------------- PLOT -------------
                fig, axes = plt.subplots(1, 4, figsize=(12, 3))

                axes[0].imshow(gray, cmap="gray")
                axes[0].set_title("Input L")
                axes[0].axis("off")

                axes[1].imshow(rgb_ref)
                axes[1].set_title("Reference RGB")
                axes[1].axis("off")

                axes[2].imshow(rgb_gt)
                axes[2].set_title("GT")
                axes[2].axis("off")

                axes[3].imshow(rgb_pred)
                axes[3].set_title("Prediction")
                axes[3].axis("off")

                fig.tight_layout()
                fname = os.path.join(out_dir, f"{split_name}_{saved:03d}.png")
                fig.savefig(fname, dpi=150)
                plt.close(fig)

                saved += 1

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    
    dataset_root = "data/imagenet_labeled"
    train_files = []
    test_files = []

    for label in os.listdir(dataset_root):
        class_dir = os.path.join(dataset_root, label)
        if not os.path.isdir(class_dir):
            continue

        # Support common image extensions (JPEG, jpg, png, etc.)
        images = []
        for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.JPG", "*.png", "*.PNG", "*.bmp", "*.webp"):
            images.extend(glob.glob(os.path.join(class_dir, ext)))

        # Skip classes with too few images (need at least 2 for a valid split)
        if len(images) < 2:
            continue

        # Split balanced class by class
        train, test = train_test_split(
            images,
            test_size=0.1,
            random_state=42,
            shuffle=True
        )

        train_files.extend(train)
        test_files.extend(test)
        
    print("Train =", len(train_files))
    print("Test  =", len(test_files))

    train_pairs = build_pairs(train_files)
    test_pairs  = build_pairs(test_files)
    val_pairs, _ = train_test_split(
        train_pairs,
        test_size=0.25,
        random_state=42,
        shuffle=True
    )
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    print(len(train_pairs))
    print(len(val_pairs))
    train_dataset = RGB2LabPairDataset(
        train_pairs,
        image_size=IMAGE_SIZE
    )
    eval_dataset = RGB2LabPairDataset(
        val_pairs,
        image_size=IMAGE_SIZE,
    )
    test_dataset = RGB2LabPairDataset(
        val_pairs,
        image_size = IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    eval_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Train size: {len(train_dataset)}  |  Eval size: {len(eval_dataset)}")

    # ModÃ¨le + loss + optim
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Inputs are [L_target, a_ref, b_ref] -> 3 channels. Instantiate accordingly.
    # Use resnet_weights=None to avoid automatic download of pretrained weights by default.
    model = ResNet_UNet(in_ch=3, resnet_weights=None).to(device)
    criterion = MagicalLoss(
        w_recon=1.0,
        w_mag=0.3,
        w_var=0.1
    )
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Scheduler: on divise le LR par 2 toutes les 40 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    try:
        from torchsummary import summary
        summary(model, input_size=(3, IMAGE_SIZE, IMAGE_SIZE), device=str(device))
    except Exception as e:
        print(f"(torchsummary non disponible ou erreur ignorÃ©e: {e})")

    # Entraînement
    print("Début de l'entraînement...")
    print("=" * 70)

    for epoch in range(NUM_EPOCHS):
        train_res = train_epoch(model, train_loader, criterion, optimizer, device, log_details=True)
        eval_loss = eval_epoch(model, eval_loader, criterion, device)

        train_loss, details = train_res
        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train: {train_loss:.4f} "
            f"(recon: {details['recon']:.4f}, "
            f"mag: {details['mag']:.4f}, "
            f"var: {details['var']:.4f}) | "
            f"Eval: {eval_loss:.4f}"
        )

        scheduler.step()

    print("=" * 70)
    print("Entraînement terminé")

    # Sauvegarde de quelques exemples
    print("Sauvegarde d'exemples...")
    save_examples(
        model, train_loader, device,
        out_dir="visuals/reference/train",
        split_name="train",
        max_images=24,
    )
    save_examples(
        model, eval_loader, device,
        out_dir="visuals/reference/val",
        split_name="val",
        max_images=24,
    )
    print("Images sauvegardées dans 'visuals/reference/train' et 'visuals/reference/val'")
    # -----------------------
    # SAUVEGARDE DU MODÈLE
    # -----------------------
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/color_by_ref_model.pth")
    print("Modèle sauvegardé dans 'checkpoints/color_by_ref_model.pth'")