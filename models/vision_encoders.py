"""Unified wrapper for frozen vision encoders: ViT, CLIP, I-JEPA."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
    ViTModel,
    ViTImageProcessor,
)


# Maps encoder_type -> (hidden_dim, num_patches)
ENCODER_META: Dict[str, Tuple[int, int]] = {
    "vit": (1024, 196),      # ViT-L/16: (224/16)^2 = 196
    "clip": (1024, 256),     # CLIP ViT-L/14: (224/14)^2 = 256
    "ijepa": (1280, 256),   # I-JEPA ViT-H/14: (224/14)^2 = 256
}


class VisionEncoderWrapper(nn.Module):
    """Unified interface around frozen ViT / CLIP / I-JEPA encoders.

    Always returns **patch tokens only** (no CLS), with shape
    ``(batch, num_patches, hidden_dim)``.
    """

    def __init__(self, model_id: str, encoder_type: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()

        if encoder_type not in ENCODER_META:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. "
                f"Choose from {list(ENCODER_META.keys())}."
            )

        self.encoder_type = encoder_type
        self.hidden_dim, self.num_patches = ENCODER_META[encoder_type]

        # --- Load model & processor ---
        if encoder_type == "vit":
            self.model = ViTModel.from_pretrained(model_id, dtype=dtype)
            self.processor = ViTImageProcessor.from_pretrained(model_id)

        elif encoder_type == "clip":
            self.model = CLIPVisionModel.from_pretrained(model_id, dtype=dtype)
            self.processor = CLIPImageProcessor.from_pretrained(model_id)

        elif encoder_type == "ijepa":
            self.model = AutoModel.from_pretrained(model_id, dtype=dtype)
            self.processor = AutoProcessor.from_pretrained(model_id)

        # --- Freeze everything ---
        self.model.requires_grad_(False)
        self.model.eval()

    def preprocess(self, images: list) -> Dict[str, torch.Tensor]:
        """Run the encoder-specific image processor.

        Args:
            images: list of PIL.Image objects.

        Returns:
            Dict with ``pixel_values`` tensor ready for the encoder.
        """
        return self.processor(images=images, return_tensors="pt")

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from images.

        Args:
            pixel_values: preprocessed image tensor
                          ``(batch, channels, height, width)``.

        Returns:
            Patch token embeddings ``(batch, num_patches, hidden_dim)``.
        """
        outputs = self.model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # (B, seq_len, D)

        if self.encoder_type in ("vit", "clip"):
            # Drop the leading CLS token -> (B, num_patches, D)
            hidden = hidden[:, 1:, :]
        # I-JEPA has no CLS; last_hidden_state is already pure patches.

        return hidden
