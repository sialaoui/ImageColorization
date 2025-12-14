import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T
from torchvision import models

from skimage.color import rgb2lab, lab2rgb

import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.unet import UNet 
from models.resnet_unet import ResNet_UNet
from utils.data import lab_to_rgb
from utils.losses import MagicalLoss
# -------------------------------------------------------------------
# CONFIG GLOBALE
# -------------------------------------------------------------------
TRAIN_DIR = "data/COCO/train/"  # dossier train
VAL_DIR   = "data/COCO/val/"    # dossier val
IMAGE_SIZE = 256

BATCH_SIZE   = 128
NUM_EPOCHS   = 80
LR           = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS  = 0

# pourcentage de pixels où on garde la vraie couleur comme "hint"
HINT_RATIO_TRAIN = 0.01   # 1% des pixels
HINT_RATIO_VAL   = 0.01   # tu peux mettre 0.0 si tu veux évaluer sans hints

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")





# -------------------------------------------------------------------
# DATASET : RGB -> Lab + HINTS SUR ab
# -------------------------------------------------------------------
class RGB2LabDataset(Dataset):
    """
    Retourne :
      x_input : (3, H, W) = [ L , hinted_a , hinted_b ]
      ab      : (2, H, W) = ground truth complet
    où hinted_a, hinted_b sont zéro partout sauf sur un pourcentage HINT_RATIO
    des pixels, où on met la vraie couleur GT.
    """
    def __init__(self, image_dir, image_size=256, hint_ratio=0.01,
                 extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ]
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

        self.image_size = image_size
        self.hint_ratio = hint_ratio

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(img)                  # (3,H,W)
        x_np = x.permute(1, 2, 0).numpy().astype(np.float32)  # (H,W,3)

        lab = rgb2lab(x_np).astype("float32")    # L: [0,100], a/b: [-128,128]

        L = lab[..., 0] / 100.0                  # [0,1]
        ab = lab[..., 1:] / 128.0                # [-1,1]

        L_tensor = torch.from_numpy(L).unsqueeze(0)          # (1,H,W)
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1)    # (2,H,W)

        H, W = L_tensor.shape[1], L_tensor.shape[2]

        # ----------------------------------------------------
        # Construction des hints : même shape que ab_tensor
        # ----------------------------------------------------
        hint_ab = torch.zeros_like(ab_tensor)   # (2,H,W)

        if self.hint_ratio > 0.0:
            num_pixels = H * W
            k = int(self.hint_ratio * num_pixels)
            k = max(k, 1)  # au moins 1 pixel si ratio > 0

            # positions aléatoires SANS (trop) de répétitions
            idxs = torch.randperm(num_pixels)[:k]
            ys = (idxs // W).long()
            xs = (idxs % W).long()

            hint_ab[:, ys, xs] = ab_tensor[:, ys, xs]

        # hinted_a, hinted_b en (1,H,W)
        hinted_a = hint_ab[0:1]
        hinted_b = hint_ab[1:2]

        # input final : 3 canaux [L, hinted_a, hinted_b]
        x_input = torch.cat([L_tensor, hinted_a, hinted_b], dim=0)  # (3,H,W)

        return x_input, ab_tensor


# -------------------------------------------------------------------
# FONCTIONS TRAIN / EVAL
# -------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, log_details=False):
    model.train()
    running_loss = 0.0
    details_acc = {"recon": 0.0, "mag": 0.0, "var": 0.0}

    for x_in, ab in loader:
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
        for x_in, ab in loader:
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
    """
    Sauvegarde quelques exemples dans out_dir :

    Colonne 1 : Input (L)
    Colonne 2 : Ground Truth
    Colonne 3 : Prediction brute
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for b_idx, (x_batch, ab_batch) in enumerate(loader):
            x_batch  = x_batch.to(device)   # (B,3,H,W)
            ab_batch = ab_batch.to(device) # (B,2,H,W)
            pred_ab  = model(x_batch)

            for i in range(x_batch.size(0)):
                if saved >= max_images:
                    return

                x_in   = x_batch[i]      # (3,H,W)
                L      = x_in[0:1, ...]  # (1,H,W)
                ab_gt  = ab_batch[i]
                ab_pred = pred_ab[i]

                gray     = L[0].cpu().numpy()
                rgb_gt   = lab_to_rgb(L, ab_gt)          # vérité terrain
                rgb_pred = lab_to_rgb(L, ab_pred)        # prédiction brute

                cols = 3
                fig, axes = plt.subplots(1, cols, figsize=(3*cols, 3))

                # 1. L
                axes[0].imshow(gray, cmap="gray")
                axes[0].set_title("Input (L)")
                axes[0].axis("off")

                # 2. GT
                axes[1].imshow(rgb_gt)
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                # 3. Pred brute
                axes[2].imshow(rgb_pred)
                axes[2].set_title("Prediction (raw)")
                axes[2].axis("off")

                fig.tight_layout()
                fname = os.path.join(out_dir, f"{split_name}_{saved:03d}.png")
                fig.savefig(fname, dpi=150)
                plt.close(fig)

                saved += 1


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Dataset & DataLoaders
    train_dataset = RGB2LabDataset(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        hint_ratio=HINT_RATIO_TRAIN
    )
    eval_dataset = RGB2LabDataset(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        hint_ratio=HINT_RATIO_VAL
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Train size: {len(train_dataset)}  |  Eval size: {len(eval_dataset)}")

    # Modèle + loss + optim
    # Use in_ch=3 because inputs are [L, a_hint, b_hint].
    # Pass resnet_weights=None to avoid automatic download of pretrained weights in environments without network.
    model = ResNet_UNet(in_ch=3, resnet_weights=None).to(device)
    criterion = MagicalLoss(
        w_recon=1.0,
        w_mag=0.3,
        w_var=0.1
    )
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Scheduler: on divise le LR par 2 toutes les 40 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # Optionnel : résumé du modèle
    try:
        from torchsummary import summary
        summary(model, input_size=(3, IMAGE_SIZE, IMAGE_SIZE), device=str(device))
    except Exception as e:
        print(f"(torchsummary non disponible ou erreur ignorée: {e})")

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
        out_dir="visuals/hint/train",
        split_name="train",
        max_images=24
    )
    save_examples(
        model, eval_loader, device,
        out_dir="visuals/hint/val",
        split_name="val",
        max_images=24
    )
    print("Images sauvegardées dans 'viza/train' et 'viza/val'")
    # -----------------------
    # SAUVEGARDE DU MODÈLE
    # -----------------------
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/color_onepercent_model.pth")
    print("Modèle sauvegardé dans 'checkpoints/color_onepercent_model.pth'")