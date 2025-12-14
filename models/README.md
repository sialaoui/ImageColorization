# Model architectures

This folder hosts the colorization networks used by the training and evaluation scripts. All models map a single-channel L input to two-channel **ab** chrominance predictions and inherit common utilities from `base_model.py` (shape validation, normalization helpers).

## UNet (`models/unet/UNet.py`)
- **Purpose:** Baseline encoder–decoder with skip connections for reconstructing spatial detail.
- **Encoder:** Five convolutional blocks with max pooling, expanding channels 1 → 512.
- **Bottleneck:** Additional 512-channel block at 1/32 resolution.
- **Decoder:** Five transposed-convolution blocks mirrored to the encoder, with additive skip connections.
- **Output:** 1×1 convolution to 2 channels followed by `tanh` to keep **ab** within [-1, 1].
- **Usage:** Loaded when `--arch unet` is passed to `scripts/train.py` or `scripts/test.py`.

## ResNet-UNet (`models/resnet_unet/ResNet_UNet.py`)
- **Purpose:** Hybrid model that fuses a frozen ImageNet-pretrained ResNet-34 encoder with a U-Net pathway to improve semantic color hints.
- **Backbone:** `torchvision.models.resnet34` truncated before the classifier; parameters are frozen to stabilize training.
- **Parallel encoder:** A U-Net style encoder on the L channel (1 → 512 channels) runs alongside the ResNet branch.
- **Fusion:** Concatenates the deepest U-Net feature map with the ResNet output, then projects to 512 channels via a 1×1 convolution.
- **Decoder:** U-Net decoder with transposed convolutions and skip connections identical to the baseline UNet.
- **Output:** 2-channel `tanh` head producing normalized **ab** predictions.
- **Usage:** Select with `--arch resnet_unet` when running testing; add to the training registry as needed.

## Extending
To register a new architecture:
1. Implement a subclass of `BaseColorizationModel` in its own module.
2. Export it from the package initializer (`models/<arch>/__init__.py`) for clean imports.
3. Add the class to the registries used in `scripts/train.py` and `scripts/test.py`.
