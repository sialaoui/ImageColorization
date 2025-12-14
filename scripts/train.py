# train.py
import argparse
import os
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from utils.data import get_dataloaders
from models.unet import UNet
from models.resnet_unet import ResNet_UNet
from utils.losses import L1Loss, SmoothL1Loss, MSELoss, MagicalLoss, ColorLoss

# ------------------------------------------
# Training / evaluation loops
# ------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0

    for L, ab in loader:
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        out = model(L)
        loss, _ = criterion(out, ab)
        loss.backward()
        optimizer.step()
        running += loss.item()

    return running / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for L, ab in loader:
            L, ab = L.to(device), ab.to(device)
            out = model(L)
            loss, _ = criterion(out, ab)
            running += loss.item()
    return running / len(loader)


# ------------------------------------------
# Main
# ------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    # Model
    if args.arch == "unet":
        model = UNet()
    elif args.arch == "resnet_unet":
        model = ResNet_UNet()
    else:
        model =  None  # extend with your registry
    model = model.to(device)

    # Loss
    if args.loss == "l1":
        criterion = L1Loss()
    elif args.loss == "smoothl1":
        criterion = SmoothL1Loss()
    elif args.loss == "mse":
        criterion = MSELoss()
    elif args.loss == "magicalloss":
        criterion = MagicalLoss()
    elif args.loss == "colorloss":
        criterion = ColorLoss()
    else:
        raise ValueError(f"Unknown loss {args.loss}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_epoch = -1
    train_history = []
    val_history = []

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"[Epoch {epoch}/{args.epochs}] Train={train_loss:.4f} | Val={val_loss:.4f}")

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)

        # Early stopping + best model
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, args.out))
        else:
            patience_counter += 1

        if patience_counter > args.patience:
            print("Early stopping triggered.")
            break

    # ------------------------------------------
    # Optional: save loss curves
    # ------------------------------------------
    if args.save_plots:
        plt.plot(train_history, label="Train")
        plt.plot(val_history, label="Val")
        plt.legend()
        plt.title("Loss curves")
        plt.savefig(os.path.join(args.checkpoint_dir, "training_curve.png"))

    print(f"Best model saved at epoch {best_epoch} with val loss {best_val:.4f}")


# ------------------------------------------
# CLI
# ------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True, help="root folder containing train/val/test/images")
    parser.add_argument("--arch", type=str, default="unet", help="architecture name")
    parser.add_argument("--loss",
                        type=str,
                        default="l1",
                        choices=["l1", "smoothl1", "mse", "magicalloss","colorloss"],
                        help="Loss function to use"
                        )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--save_plots", action="store_true", default=True)
    parser.add_argument("--plots_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join("checkpoints", args.arch)

    if args.plots_dir is None:
        args.plots_dir = os.path.join("experiments", args.arch, "plots")

    if args.out is None:
        args.out = "best_model_" + args.arch + "_" + args.loss + ".pth"


    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    main(args)