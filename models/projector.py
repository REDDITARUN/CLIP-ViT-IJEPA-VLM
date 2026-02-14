"""Trainable MLP projector (bridge) from vision space to LLM space."""

import torch
import torch.nn as nn


class Projector(nn.Module):
    """Two-layer MLP that maps vision encoder patch tokens into the LLM
    embedding space.

    Architecture:
        LayerNorm(vision_dim)
        -> Linear(vision_dim, llm_dim)
        -> GELU
        -> Linear(llm_dim, llm_dim)

    Applied independently to each patch token:
        (B, N, vision_dim) -> (B, N, llm_dim)
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision tokens to LLM space.

        Args:
            x: vision patch tokens ``(batch, num_patches, vision_dim)``.

        Returns:
            Projected tokens ``(batch, num_patches, llm_dim)``.
        """
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
