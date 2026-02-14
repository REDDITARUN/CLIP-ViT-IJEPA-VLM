"""Loss tracking and convergence plotting utilities."""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


class LossTracker:
    """Accumulates per-step losses and provides smoothed statistics.

    Usage::

        tracker = LossTracker("clip")
        for step in range(num_steps):
            ...
            tracker.update(step, loss_value)
        tracker.summary()
    """

    def __init__(self, name: str):
        self.name = name
        self.steps: List[int] = []
        self.losses: List[float] = []

    def update(self, step: int, loss: float) -> None:
        self.steps.append(step)
        self.losses.append(loss)

    def running_average(self, window: int = 20) -> List[float]:
        """Compute a simple moving average over the loss history."""
        smoothed = []
        for i in range(len(self.losses)):
            start = max(0, i - window + 1)
            smoothed.append(sum(self.losses[start : i + 1]) / (i - start + 1))
        return smoothed

    def summary(self) -> dict:
        """Return a dict summary of the tracker."""
        if not self.losses:
            return {"name": self.name, "steps": 0}
        return {
            "name": self.name,
            "total_steps": len(self.losses),
            "final_loss": self.losses[-1],
            "min_loss": min(self.losses),
            "avg_loss_last_50": (
                sum(self.losses[-50:]) / min(50, len(self.losses))
            ),
        }

    def save(self, path: str) -> None:
        """Save loss history to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "name": self.name,
            "steps": self.steps,
            "losses": self.losses,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LossTracker":
        """Load a tracker from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        tracker = cls(data["name"])
        tracker.steps = data["steps"]
        tracker.losses = data["losses"]
        return tracker


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #


def plot_convergence(
    trackers: Dict[str, LossTracker],
    title: str = "Convergence: Vision Encoder Comparison",
    smoothing_window: int = 20,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot training loss curves for multiple encoders on one figure.

    Args:
        trackers: mapping ``encoder_name -> LossTracker``.
        title: plot title.
        smoothing_window: moving-average window size.
        save_path: if given, save the figure to this path.
        figsize: matplotlib figure size.

    Returns:
        The matplotlib Figure object.
    """
    fig, (ax_raw, ax_smooth) = plt.subplots(1, 2, figsize=figsize)

    colors = {"vit": "#e74c3c", "clip": "#2ecc71", "ijepa": "#3498db"}

    for name, tracker in trackers.items():
        color = colors.get(name, None)

        # Raw loss
        ax_raw.plot(
            tracker.steps,
            tracker.losses,
            alpha=0.35,
            color=color,
            linewidth=0.8,
        )

        # Smoothed loss
        smoothed = tracker.running_average(smoothing_window)
        ax_smooth.plot(
            tracker.steps,
            smoothed,
            label=name.upper(),
            color=color,
            linewidth=2,
        )

    ax_raw.set_title("Raw Loss")
    ax_raw.set_xlabel("Step")
    ax_raw.set_ylabel("Loss")
    ax_raw.grid(True, alpha=0.3)

    ax_smooth.set_title(f"Smoothed Loss (window={smoothing_window})")
    ax_smooth.set_xlabel("Step")
    ax_smooth.set_ylabel("Loss")
    ax_smooth.legend(fontsize=12)
    ax_smooth.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved convergence plot to {save_path}")

    return fig
