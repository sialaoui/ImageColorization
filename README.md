# Colors project

This repository hosts neural colorization utilities organized around a handful of reusable modules:

- **`models/`** — PyTorch architectures, including U-Net variants, that map luminance inputs to chrominance predictions.
- **`utils/`** — Data pipelines, loss functions, and evaluation metrics shared across training and testing.
- **`scripts/`** — Command-line entry points for training, evaluation, visualization, and reference-based workflows.
    
## Automatic colorization
1. Install Python 3.10+ with PyTorch and torchvision. A typical setup:
   ```bash
   pip install -r requirements.txt  
   ```
2. Prepare a dataset directory with `train/`, `val/`, and `test/` splits of RGB images. The loaders convert them to CIELAB internally.
3. Train a model (default U-Net):
    example for the COCO data and a save in unet in the checkpoint directory.
   ```bash
   python -m scripts.train --data_root data/COCO --checkpoint_dir checkpoints/unet
   ```
   using the resnet_unet architecture 
   ```bash
   python -m scripts.train --data_root data/COCO --checkpoint_dir checkpoints/resnet_unet --arch resnet_unet
   ```
4. Evaluate a saved checkpoint:
example using oco data and the architecture resnet_unet . we can change the architecture, checkpoints, data in arguments and with --loss we choose the loss;by default it's l1
   ```bash
    python -m scripts.test --data_root data/COCO --checkpoint checkpoints/resnet_unet/best_model_resnet_unet_l1.pth --arch resnet_unet

   ```
5. Visualize predictions from a folder of images:
here we visualized with the same data with --split of validation and with the default argument --num_images of 2 .
   ```bash
   python -m scripts.visualize --data_root data/COCO/ --checkpoint checkpoints/unet/best_model_unet_l1.pth --split val
   ```
## hint-based colorization

Hint-based colorization
------------------------
`scripts/train_with_hints.py` implements a training variant where sparse ground-truth chrominance "hints" (a,b) are injected into the model input. The dataset returns a 3-channel input formed as `[L, a_hint, b_hint]` where `a_hint`/`b_hint` are mostly zeros with a small fraction of pixels filled with the true chrominance.

How to run
```bash
# Runs the script using constants defined inside the file
python -m scripts.train_with_hints
```

Configurable constants inside `scripts/train_with_hints.py`
- `TRAIN_DIR`, `VAL_DIR` — paths to your train/val image folders.
- `IMAGE_SIZE` — resize size (default 256).
- `BATCH_SIZE`, `NUM_EPOCHS`, `LR`, `WEIGHT_DECAY`, `NUM_WORKERS` — training hyperparameters.
- `HINT_RATIO_TRAIN`, `HINT_RATIO_VAL` — fraction of pixels that are revealed as hints (default 0.01 = 1%).

Notes
- The script currently uses file-level constants (not CLI args). If you want CLI flexibility, see the suggestion below to add `argparse` support.

Reference-based colorization
----------------------------
`scripts/train_with_reference.py` implements reference-based training: for each target image, a reference image (from the same class) provides chrominance channels (`a_ref`, `b_ref`). The network input is `[L_target, a_ref, b_ref]` and the target output is the true `ab` of the target image.

How the script constructs pairs
- The script expects a root dataset directory where subfolders correspond to classes (example: `imagenet_labeled/<class_id>/*.JPEG`). It pairs images inside each class (adjacent pairs in sorted order) and produces both directions (A→B and B→A) for training.

How to run
```bash
python -m scripts.train_with_reference
```

Configurable constants inside `scripts/train_with_reference.py`
- `dataset_root` — root folder containing per-class subfolders (default `imagenet_labeled` in the script).
- `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `LR`, `WEIGHT_DECAY`, `NUM_WORKERS` — training hyperparameters.

Outputs produced by hint/reference scripts
- Checkpoints are saved under `checkpoints/` (file names match those used in each script: e.g. `color_onepercent_model.pth`, `color_by_ref_model.pth`).
- Visual examples are saved to `visuals/hint/` and `visuals/reference/` respectively.

Attention-based reference colorization (notebook)
-------------------------------------------------
`scripts/attentionMapRefColorisation.ipynb` is an interactive Jupyter notebook that demonstrates a more advanced **attention-based** reference colorization approach.

**How it works**
1. **Dual U-Net encoders** — separate encoders for the target luminance (`TL`) and the reference luminance (`RL`).
2. **VGG19 feature extractor** — extracts multi-scale color features (`phi1`, `phi2`, `phi3`) from the RGB reference image using a frozen pretrained VGG19.
3. **Grid-based attention** — at each scale, the notebook computes a soft attention map between target and reference feature grids, then transfers the corresponding color features to guide the decoder.
4. **U-Net decoder** — fuses luminance skip connections with the transferred color features to predict the ab chrominance of the target.

**Dataset expected**
The notebook uses a custom `ReferencePairDataset` that expects:
- A source folder (low-resolution target images) and a reference folder (color references).
- Image pairs are matched by a common ID prefix (e.g., `000000000036` → `000000000036xSomeRef.png`).

You can adapt these paths inside the notebook:
- `train_src`, `valid_src` — source image folders.
- `train_ref`, `valid_ref` — reference image folders.
- `image_size`, `batch_size`, `epochs`, `lr` — training hyperparameters.

**How to run**
1. Open the notebook in VS Code or Jupyter:
   ```bash
   code scripts/attentionMapRefColorisation.ipynb
   # or
   jupyter notebook scripts/attentionMapRefColorisation.ipynb
   ```
2. Run cells sequentially — the notebook trains the model, saves a checkpoint to `checkpoints/attention_map_reference_model.pth`, and displays sample colorizations.
3. After training, the final cells compute evaluation metrics (L1, L2, PSNR, SSIM, LPIPS, FID) on the test set.

**Output**
- Checkpoint: `checkpoints/attention_map_reference_model.pth`
- Visual comparison plots showing: grayscale input, reference image, ground-truth color, and predicted colorization.

