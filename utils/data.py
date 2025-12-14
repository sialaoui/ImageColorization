# utils/data.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torchvision.transforms as T


# ----------------------------------------------------------
# LAB → RGB for visualization
# ----------------------------------------------------------
def lab_to_rgb(L, ab):
    """
    L:  (1,H,W) in [0,1]
    ab: (2,H,W) in [-1,1]

    returns RGB image (H,W,3) in [0,1]
    """
    L = L[0].cpu().numpy()                # (H,W)
    ab = ab.cpu().numpy().transpose(1,2,0)  # (H,W,2)

    # Undo normalization
    L = L * 100
    ab = ab * 128

    lab = np.concatenate([L[..., None], ab], axis=2)
    rgb = np.clip(lab2rgb(lab.astype(np.float64)), 0, 1)
    return rgb

def lab_to_rgb_batch(L, ab):
    """
    L:  (B,1,H,W)
    ab: (B,2,H,W)
    Return RGB in shape (B,3,H,W) normalized to [0,1].
    """
    rgbs = []
    for i in range(L.size(0)):
        rgb = lab_to_rgb(L[i], ab[i])  # (H,W,3)
        rgb = torch.from_numpy(rgb).permute(2,0,1)  # (3,H,W)
        rgbs.append(rgb)
    return torch.stack(rgbs, dim=0)

# ----------------------------------------------------------
# Dataset that outputs L (input) and ab (target)
# ----------------------------------------------------------
class RGB2LabDataset(Dataset):
    def __init__(self, image_dir, image_size=256, extensions=('.jpg','.jpeg','.png','.bmp','.webp')):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ]
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(img)
        x_np = x.permute(1,2,0).numpy().astype(np.float32)

        lab = rgb2lab(x_np).astype("float32")
        L  = lab[...,0] / 100.0
        ab = lab[...,1:] / 128.0

        L  = torch.from_numpy(L).unsqueeze(0)
        ab = torch.from_numpy(ab).permute(2,0,1)

        return L, ab


# ----------------------------------------------------------
# Build train/val/test dataloaders
# ----------------------------------------------------------
def get_dataloaders(root, batch_size=16, image_size=256, num_workers=4):
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val") 
    test_dir  = os.path.join(root, "test")

    train_set = RGB2LabDataset(train_dir, image_size=image_size)
    val_set   = RGB2LabDataset(val_dir, image_size=image_size)
    test_set  = RGB2LabDataset(test_dir, image_size=image_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
