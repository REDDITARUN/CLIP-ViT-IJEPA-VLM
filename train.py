"""Core training loop for the VLM embedding stitching benchmark.

Can be imported or run as a standalone script::

    python train.py                        # runs all 3 encoders
    python train.py --encoder clip         # run a single encoder
    python train.py --no-wandb             # disable wandb
    python train.py --no-push              # disable HF Hub push
"""

import gc
import json
import os
import time
from dataclasses import asdict
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import ExperimentConfig, get_encoder_configs
from data import CauldronDataset, VLMCollator
from models import StitchedVLM
from utils import LossTracker, plot_convergence


# --------------------------------------------------------------------------- #
#  Environment / tokens
# --------------------------------------------------------------------------- #


def _load_env():
    """Load .env file if it exists (HF_TOKEN, WANDB_API_KEY, HF_REPO_ID)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed -- rely on env vars directly


def _get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN")


def _get_wandb_key() -> Optional[str]:
    return os.environ.get("WANDB_API_KEY")


def _get_hf_repo() -> Optional[str]:
    return os.environ.get("HF_REPO_ID")


# --------------------------------------------------------------------------- #
#  Weights & Biases
# --------------------------------------------------------------------------- #


def _init_wandb(config: ExperimentConfig, enabled: bool = True):
    """Initialize a wandb run for this experiment. Returns the run or None."""
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("  [WARN] wandb not installed, skipping logging.", flush=True)
        return None

    key = _get_wandb_key()
    if key:
        wandb.login(key=key, relogin=True)

    run = wandb.init(
        project="rlj-vlm-benchmark",
        name=f"{config.encoder_name}-{config.llm_model_id.split('/')[-1]}",
        group=config.dataset_subset,
        tags=[config.encoder_name, config.encoder_type],
        config=asdict(config),
        reinit=True,
    )
    return run


def _log_wandb(run, step: int, metrics: dict):
    """Log metrics to wandb if run is active."""
    if run is not None:
        run.log(metrics, step=step)


def _finish_wandb(run):
    """Finish a wandb run if active."""
    if run is not None:
        run.finish()


# --------------------------------------------------------------------------- #
#  HuggingFace Hub
# --------------------------------------------------------------------------- #


def _push_to_hub(config: ExperimentConfig, save_dir: str, enabled: bool = True):
    """Push trained projector + LoRA + config to HuggingFace Hub."""
    if not enabled:
        return

    repo_id = _get_hf_repo()
    token = _get_hf_token()

    if not repo_id:
        print("  [SKIP] HF_REPO_ID not set, skipping Hub push.", flush=True)
        return
    if not token:
        print("  [SKIP] HF_TOKEN not set, skipping Hub push.", flush=True)
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  [WARN] huggingface_hub not installed, skipping push.", flush=True)
        return

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id, exist_ok=True, private=True)

    # Upload the encoder's output folder
    folder_path = os.path.join(save_dir, config.encoder_name)
    path_in_repo = config.encoder_name

    print(f"  Pushing {folder_path}/ -> hf://{repo_id}/{path_in_repo}/", flush=True)

    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        path_in_repo=path_in_repo,
        commit_message=f"Upload {config.encoder_name} projector + LoRA",
    )
    print(f"  [OK] Pushed to https://huggingface.co/{repo_id}", flush=True)


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
    use_wandb: bool = True,
    push_to_hub: bool = True,
) -> LossTracker:
    """Train one encoder configuration and return its LossTracker.

    Args:
        config: the experiment config for this run.
        device: torch device; auto-detected if None.
        use_wandb: whether to log to Weights & Biases.
        push_to_hub: whether to push results to HuggingFace Hub.

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

    # ---- Wandb ----
    wandb_run = _init_wandb(config, enabled=use_wandb)

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

    # ---- Auto-calculate num_steps if not set ----
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    if config.num_steps is None:
        steps_per_epoch = max(1, len(dataset) // effective_batch)
        config.num_steps = steps_per_epoch * config.num_epochs
    approx_epochs = config.num_steps * effective_batch / max(len(dataset), 1)
    print(f"  Dataset:   {len(dataset):,} samples", flush=True)
    print(f"  Eff batch: {effective_batch}  |  Steps: {config.num_steps}  "
          f"(~{approx_epochs:.1f} epochs)", flush=True)

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
    micro_step_total = 0
    accum_loss = 0.0
    t_start = time.time()
    accum = config.gradient_accumulation_steps

    # Outer bar: optimizer steps
    pbar_step = tqdm(
        total=config.num_steps,
        desc=f"{config.encoder_name.upper():6s} steps",
        position=0,
        bar_format="{l_bar}{bar:30}{r_bar}",
        leave=True,
    )
    # Inner bar: micro-batches within current optimizer step
    pbar_micro = tqdm(
        total=accum,
        desc=f"{'':6s} micro",
        position=1,
        bar_format="{l_bar}{bar:15}{r_bar}",
        leave=True,
    )

    while global_step < config.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs["loss"] / accum
        loss.backward()
        accum_loss += loss.item()
        micro_step_total += 1

        # Update inner bar
        pbar_micro.update(1)
        pbar_micro.set_postfix(loss=f"{loss.item() * accum:.4f}")

        if micro_step_total % accum == 0:
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

            pbar_step.set_postfix(
                loss=f"{step_loss:.4f}",
                lr=f"{lr_now:.2e}",
                t=f"{elapsed:.0f}s",
            )
            pbar_step.update(1)

            # Reset inner bar for next optimizer step
            pbar_micro.reset()

            # ---- Log to wandb ----
            _log_wandb(wandb_run, global_step, {
                "train/loss": step_loss,
                "train/lr": lr_now,
                "train/elapsed_s": elapsed,
                "train/steps_per_sec": global_step / elapsed if elapsed > 0 else 0,
            })

    pbar_micro.close()
    pbar_step.close()

    elapsed_total = time.time() - t_start
    summary = tracker.summary()
    print(f"\n  Done in {elapsed_total:.1f}s  |  "
          f"Final loss: {summary.get('final_loss', 'N/A'):.4f}  |  "
          f"Min loss: {summary.get('min_loss', 'N/A'):.4f}\n", flush=True)

    # ---- Save outputs ----
    save_dir = os.path.join(config.save_dir, config.encoder_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Loss history (JSON)
    tracker.save(os.path.join(save_dir, "loss_history.json"))

    # 2. Projector weights
    proj_path = os.path.join(save_dir, "projector.pt")
    torch.save(model.projector.state_dict(), proj_path)
    print(f"  Saved projector  -> {proj_path}", flush=True)

    # 3. LoRA adapter (if used)
    if config.use_lora:
        lora_path = os.path.join(save_dir, "lora_adapter")
        model.llm.save_pretrained(lora_path)
        print(f"  Saved LoRA       -> {lora_path}/", flush=True)

    # 4. Training config (for reproducibility)
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"  Saved config     -> {cfg_path}", flush=True)

    # ---- Push to HuggingFace Hub ----
    _push_to_hub(config, config.save_dir, enabled=push_to_hub)

    # ---- Finish wandb ----
    _finish_wandb(wandb_run)

    return tracker


# --------------------------------------------------------------------------- #
#  Run all experiments
# --------------------------------------------------------------------------- #


def run_all_experiments(
    configs: Optional[list] = None,
    device: Optional[torch.device] = None,
    use_wandb: bool = True,
    push_to_hub: bool = True,
) -> Dict[str, LossTracker]:
    """Run all encoder experiments sequentially and return trackers.

    Args:
        configs: list of ExperimentConfig; defaults to all three encoders.
        device: torch device; auto-detected if None.
        use_wandb: whether to log to Weights & Biases.
        push_to_hub: whether to push results to HuggingFace Hub.

    Returns:
        dict mapping encoder_name -> LossTracker.
    """
    if configs is None:
        configs = get_encoder_configs()

    device = device or _get_device()
    trackers: Dict[str, LossTracker] = {}

    for cfg in configs:
        tracker = run_experiment(
            cfg, device=device, use_wandb=use_wandb, push_to_hub=push_to_hub,
        )
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

    _load_env()

    parser = argparse.ArgumentParser(description="VLM Embedding Benchmark")
    parser.add_argument(
        "--encoder", type=str, default=None,
        choices=["vit", "clip", "ijepa"],
        help="Run a single encoder (default: run all three).",
    )
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_subset", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--no-push", action="store_true", help="Disable HF Hub push.")
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

    results = run_all_experiments(
        all_configs,
        use_wandb=not args.no_wandb,
        push_to_hub=not args.no_push,
    )
