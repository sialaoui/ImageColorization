# models/base_model.py
import abc
import torch.nn as nn


class BaseColorizationModel(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for all colorization models.

    All models must:
      - take input L  : (B,1,H,W) in [0,1]
      - output ab     : (B,2,H,W) in [-1,1]

    This ensures all models have consistent signatures.
    """

    def __init__(self):
        super().__init__()

    # -----------------------------------------
    # Child models MUST implement forward()
    # -----------------------------------------
    @abc.abstractmethod
    def forward(self, L):
        """
        Args:
            L: Tensor (B,1,H,W) Lightness channel in [0,1]

        Returns:
            ab: Tensor (B,2,H,W) chroma channels in [-1,1]
        """
        raise NotImplementedError

    # -----------------------------------------
    # Utility: weight initialization
    # -----------------------------------------
    def init_weights(self, init_type="kaiming"):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------
    # Freeze / unfreeze
    # -----------------------------------------
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    # -----------------------------------------
    # Count parameters
    # -----------------------------------------
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -----------------------------------------
    # Shape Validator (optional but useful)
    # -----------------------------------------
    def validate_forward(self, L, out):
        if L.dim() != 4 or L.size(1) not in (1, 3):
            raise ValueError("Input must be (B,1,H,W) or (B,3,H,W).")
        if out.size(1) != 2:
            raise ValueError("Model output must be (B,2,H,W).")
        return True