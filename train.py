"""Core training loop for the VLM embedding stitching benchmark.

Can be imported from a notebook (Colab) or run as a standalone script::

    python train.py                        # runs all 3 encoders
    python train.py --encoder clip         # run a single encoder
"""

import gc
import os
import time
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
try:
    # Use notebook widget if available, otherwise plain text bars
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from configs import ExperimentConfig, get_encoder_configs
from data import CauldronDataset, VLMCollator
from models import StitchedVLM
from utils import LossTracker, plot_convergence


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Resolve dtype string, forcing float32 on MPS (bfloat16 is unstable)."""
    if device.type == "mps":
        print("  [INFO] MPS detected -- forcing float32 (bfloat16 is unstable on MPS)")
        return torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_lr_lambda(warmup_steps: int):
    """Linear warmup then constant."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(float(step) / max(warmup_steps, 1), 1e-2)
        return 1.0
    return lr_lambda


# --------------------------------------------------------------------------- #
#  Single experiment
# --------------------------------------------------------------------------- #


def run_experiment(
    config: ExperimentConfig,
    device: Optional[torch.device] = None,
) -> LossTracker:
    """Train one encoder configuration and return its LossTracker.

    Args:
        config: the experiment config for this run.
        device: torch device; auto-detected if None.

    Returns:
        A ``LossTracker`` with per-step losses.
    """
    device = device or _get_device()
    dtype = _resolve_dtype(config.dtype, device)

    print("=" * 70, flush=True)
    print(f"  EXPERIMENT: {config.encoder_name.upper()}", flush=True)
    print(f"  Encoder:    {config.encoder_model_id}", flush=True)
    print(f"  LLM:        {config.llm_model_id}", flush=True)
    print(f"  Device:     {device}   Dtype: {dtype}", flush=True)
    print("=" * 70, flush=True)

    # ---- Build model ----
    model = StitchedVLM(
        encoder_model_id=config.encoder_model_id,
        encoder_type=config.encoder_type,
        llm_model_id=config.llm_model_id,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        dtype=dtype,
    )
    model = model.to(device)

    trainable = model.count_trainable_parameters()
    total = model.count_total_parameters()
    print(f"  Trainable params: {trainable:,}  ({trainable / total * 100:.2f}%)", flush=True)
    print(f"  Total params:     {total:,}", flush=True)

    # ---- Dataset & DataLoader ----
    dataset = CauldronDataset(
        dataset_name=config.dataset_name,
        subset=config.dataset_subset,
        max_samples=config.max_samples,
    )
    collator = VLMCollator(
        image_processor=model.encoder.processor,
        tokenizer=model.tokenizer,
        max_seq_len=config.max_seq_len,
    )
    # MPS doesn't play well with multiprocessing data loaders
    num_workers = 0 if device.type == "mps" else 2
    pin_memory = device.type == "cuda"

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    data_iter = iter(dataloader)

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(
        model.get_trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _get_lr_lambda(config.warmup_steps)
    )

    # ---- Training loop ----
    tracker = LossTracker(config.encoder_name)
    model.train()
    optimizer.zero_grad()

    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    t_start = time.time()

    total_micro_steps = config.num_steps * config.gradient_accumulation_steps
    print(f"\n  Starting training: {config.num_steps} optimizer steps "
          f"({total_micro_steps} micro-steps, "
          f"accum={config.gradient_accumulation_steps})\n", flush=True)

    while global_step < config.num_steps:
        # Fetch next batch (restart dataloader if exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward
        outputs = model(**batch)
        loss = outputs["loss"] / config.gradient_accumulation_steps
        loss.backward()
        accum_loss += loss.item()
        micro_step += 1

        # Log every micro-step so the user can see activity
        micro_loss = loss.item() * config.gradient_accumulation_steps
        print(f"  micro {micro_step:>6d} / {total_micro_steps}  |  "
              f"accum {((micro_step - 1) % config.gradient_accumulation_steps) + 1}"
              f"/{config.gradient_accumulation_steps}  |  "
              f"micro_loss={micro_loss:.4f}", flush=True)

        # Optimizer step after accumulating enough micro-batches
        if micro_step % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.get_trainable_parameters()),
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            step_loss = accum_loss
            accum_loss = 0.0

            tracker.update(global_step, step_loss)

            elapsed = time.time() - t_start
            lr_now = scheduler.get_last_lr()[0]
            steps_per_sec = global_step / elapsed if elapsed > 0 else 0
            eta = (config.num_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0

            print(f"  >>> STEP {global_step:>5d} / {config.num_steps}  |  "
                  f"loss={step_loss:.4f}  |  "
                  f"lr={lr_now:.2e}  |  "
                  f"elapsed={elapsed:.0f}s  |  "
                  f"speed={steps_per_sec:.2f} steps/s  |  "
                  f"ETA={eta:.0f}s", flush=True)
            print("-" * 80, flush=True)

    print("", flush=True)

    elapsed_total = time.time() - t_start
    summary = tracker.summary()
    print(f"\n  Done in {elapsed_total:.1f}s  |  "
          f"Final loss: {summary.get('final_loss', 'N/A'):.4f}  |  "
          f"Min loss: {summary.get('min_loss', 'N/A'):.4f}\n", flush=True)

    # ---- Save tracker ----
    save_dir = os.path.join(config.save_dir, config.encoder_name)
    tracker.save(os.path.join(save_dir, "loss_history.json"))

    return tracker


# --------------------------------------------------------------------------- #
#  Run all experiments
# --------------------------------------------------------------------------- #


def run_all_experiments(
    configs: Optional[list] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, LossTracker]:
    """Run all encoder experiments sequentially and return trackers.

    Args:
        configs: list of ExperimentConfig; defaults to all three encoders.
        device: torch device; auto-detected if None.

    Returns:
        dict mapping encoder_name -> LossTracker.
    """
    if configs is None:
        configs = get_encoder_configs()

    device = device or _get_device()
    trackers: Dict[str, LossTracker] = {}

    for cfg in configs:
        tracker = run_experiment(cfg, device=device)
        trackers[cfg.encoder_name] = tracker

        # Aggressively free GPU memory between runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Plot comparison ----
    save_path = os.path.join(configs[0].save_dir, "convergence.png")
    plot_convergence(trackers, save_path=save_path)

    # ---- Print summary ----
    print("\n" + "=" * 70, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for name, trk in trackers.items():
        s = trk.summary()
        print(f"  {name.upper():8s} | final={s.get('final_loss', 0):.4f}  "
              f"min={s.get('min_loss', 0):.4f}  "
              f"avg_last50={s.get('avg_loss_last_50', 0):.4f}", flush=True)
    print("=" * 70, flush=True)

    return trackers


# --------------------------------------------------------------------------- #
#  CLI entry point
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM Embedding Benchmark")
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        choices=["vit", "clip", "ijepa"],
        help="Run a single encoder (default: run all three).",
    )
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_subset", type=str, default=None)
    args = parser.parse_args()

    all_configs = get_encoder_configs()

    # Filter to single encoder if requested
    if args.encoder:
        all_configs = [c for c in all_configs if c.encoder_name == args.encoder]

    # Apply CLI overrides
    for cfg in all_configs:
        if args.num_steps is not None:
            cfg.num_steps = args.num_steps
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.max_samples is not None:
            cfg.max_samples = args.max_samples
        if args.dataset_subset is not None:
            cfg.dataset_subset = args.dataset_subset

    results = run_all_experiments(all_configs)
