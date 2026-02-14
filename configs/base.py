"""Experiment configuration for VLM embedding stitching benchmark."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Full configuration for a single encoder benchmark run."""

    # --- Encoder ---
    encoder_name: str = "clip"
    encoder_model_id: str = "openai/clip-vit-large-patch14"
    encoder_type: str = "clip"  # one of: "vit", "clip", "ijepa"

    # --- LLM ---
    llm_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # --- LoRA ---
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # --- Dataset ---
    dataset_name: str = "HuggingFaceM4/the_cauldron"
    dataset_subset: str = "clevr"
    max_samples: Optional[int] = None  # None = use all

    # --- Training ---
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    num_steps: int = 2000
    max_grad_norm: float = 1.0

    # --- Eval / Logging ---
    eval_every: int = 50
    log_every: int = 10
    save_dir: str = "outputs"

    # --- Misc ---
    seed: int = 42
    dtype: str = "bfloat16"  # "float16" or "bfloat16"
    max_seq_len: int = 256


def get_encoder_configs() -> List[ExperimentConfig]:
    """Return pre-filled configs for all three vision encoders."""

    vit_config = ExperimentConfig(
        encoder_name="vit",
        encoder_model_id="google/vit-large-patch16-224",
        encoder_type="vit",
    )

    clip_config = ExperimentConfig(
        encoder_name="clip",
        encoder_model_id="openai/clip-vit-large-patch14",
        encoder_type="clip",
    )

    ijepa_config = ExperimentConfig(
        encoder_name="ijepa",
        encoder_model_id="facebook/ijepa_vith14_1k",
        encoder_type="ijepa",
    )

    return [vit_config, clip_config, ijepa_config]
