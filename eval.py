"""Evaluate trained VLM models on held-out benchmarks.

Computes exact-match accuracy on VQA datasets not seen during training.

Usage:
    python eval.py                                    # eval all 3 encoders on all benchmarks
    python eval.py --encoder clip                     # eval one encoder
    python eval.py --encoder clip --benchmark okvqa   # specific benchmark
    python eval.py --max_samples 500                  # quick eval on subset
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from inference import load_from_local, TrainedVLM
from data import CauldronDataset


# --------------------------------------------------------------------------- #
#  Benchmarks: datasets NOT used in training
# --------------------------------------------------------------------------- #

BENCHMARKS = {
    "okvqa": {
        "subset": "okvqa",
        "description": "OK-VQA: requires external knowledge (9k samples)",
    },
    "aokvqa": {
        "subset": "aokvqa",
        "description": "A-OKVQA: augmented knowledge VQA (17k samples)",
    },
    "cocoqa": {
        "subset": "cocoqa",
        "description": "COCO-QA: basic visual QA (46k samples)",
    },
    "visual7w": {
        "subset": "visual7w",
        "description": "Visual7W: who/what/where/when/why/how/which (70k samples)",
    },
    "tallyqa": {
        "subset": "tallyqa",
        "description": "TallyQA: counting objects (184k samples)",
    },
}


# --------------------------------------------------------------------------- #
#  Answer normalization (standard VQA eval)
# --------------------------------------------------------------------------- #


def _normalize_answer(answer: str) -> str:
    """Normalize an answer for exact-match comparison.

    Follows standard VQA evaluation:
      - lowercase
      - strip punctuation and articles
      - strip whitespace
    """
    import re
    import string

    answer = answer.lower().strip()

    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))

    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)

    # Collapse whitespace
    answer = " ".join(answer.split())

    return answer


def _check_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth (normalized exact match)."""
    pred = _normalize_answer(predicted)
    gt = _normalize_answer(ground_truth)

    # Exact match
    if pred == gt:
        return True

    # Check if ground truth is contained in prediction (for short answers)
    if gt in pred:
        return True

    return False


# --------------------------------------------------------------------------- #
#  Evaluation loop
# --------------------------------------------------------------------------- #


def evaluate_model(
    model: TrainedVLM,
    benchmark_name: str,
    max_samples: int = 1000,
) -> dict:
    """Run evaluation on a single benchmark.

    Args:
        model: loaded TrainedVLM instance.
        benchmark_name: key from BENCHMARKS dict.
        max_samples: max samples to evaluate on.

    Returns:
        dict with accuracy, total, correct, and per-sample results.
    """
    bench = BENCHMARKS[benchmark_name]

    print(f"\n  Evaluating on: {bench['description']}", flush=True)

    # Load benchmark dataset
    dataset = CauldronDataset(
        dataset_name="HuggingFaceM4/the_cauldron",
        subset=bench["subset"],
        max_samples=max_samples,
    )

    correct = 0
    total = 0
    results = []

    pbar = tqdm(range(len(dataset)), desc=f"  {benchmark_name}")

    for idx in pbar:
        sample = dataset[idx]
        image = sample["image"]
        question = sample["instruction"].replace("<image>\n", "")
        ground_truth = sample["response"]

        # Generate answer
        predicted = model.generate(
            image,
            question=question,
            max_new_tokens=20,
            do_sample=False,
        )

        match = _check_match(predicted, ground_truth)
        if match:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": match,
        })

        # Update progress
        acc = correct / total * 100
        pbar.set_postfix(acc=f"{acc:.1f}%", correct=correct, total=total)

    accuracy = correct / total * 100 if total > 0 else 0

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def evaluate_encoder(
    encoder_name: str,
    benchmarks: Optional[List[str]] = None,
    save_dir: str = "outputs",
    max_samples: int = 1000,
) -> Dict[str, dict]:
    """Evaluate a single encoder on multiple benchmarks.

    Args:
        encoder_name: one of "vit", "clip", "ijepa".
        benchmarks: list of benchmark names (default: all).
        save_dir: where trained weights are stored.
        max_samples: max samples per benchmark.

    Returns:
        dict mapping benchmark_name -> results.
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    print(f"\n{'=' * 60}", flush=True)
    print(f"  EVALUATING: {encoder_name.upper()}", flush=True)
    print(f"{'=' * 60}", flush=True)

    model = load_from_local(encoder_name=encoder_name, save_dir=save_dir)

    all_results = {}
    for bench_name in benchmarks:
        result = evaluate_model(model, bench_name, max_samples=max_samples)
        all_results[bench_name] = result

    # Free memory
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


def run_full_eval(
    encoder_names: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
    save_dir: str = "outputs",
    max_samples: int = 1000,
) -> dict:
    """Run evaluation for all encoders on all benchmarks.

    Returns:
        Nested dict: encoder_name -> benchmark_name -> results.
    """
    if encoder_names is None:
        encoder_names = ["vit", "clip", "ijepa"]

    all_results = {}
    for enc in encoder_names:
        enc_dir = os.path.join(save_dir, enc)
        if not os.path.isdir(enc_dir):
            print(f"\n  [SKIP] {enc}: no trained weights at {enc_dir}/", flush=True)
            continue
        all_results[enc] = evaluate_encoder(
            enc, benchmarks=benchmarks, save_dir=save_dir, max_samples=max_samples,
        )

    # ---- Print summary table ----
    _print_summary(all_results)

    # ---- Save results ----
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Save full results (without per-sample for size)
    summary = {}
    for enc, benches in all_results.items():
        summary[enc] = {}
        for bench_name, res in benches.items():
            summary[enc][bench_name] = {
                "accuracy": res["accuracy"],
                "correct": res["correct"],
                "total": res["total"],
            }

    summary_path = os.path.join(eval_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved eval summary -> {summary_path}", flush=True)

    return all_results


def _print_summary(all_results: dict):
    """Print a comparison table."""
    if not all_results:
        return

    # Collect all benchmark names
    bench_names = set()
    for benches in all_results.values():
        bench_names.update(benches.keys())
    bench_names = sorted(bench_names)

    print(f"\n{'=' * 70}", flush=True)
    print(f"  EVALUATION SUMMARY (Accuracy %)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Header
    header = f"  {'Encoder':<10s}"
    for b in bench_names:
        header += f" | {b:>10s}"
    header += f" | {'Average':>8s}"
    print(header, flush=True)
    print(f"  {'-' * (len(header) - 2)}", flush=True)

    # Rows
    for enc in sorted(all_results.keys()):
        row = f"  {enc.upper():<10s}"
        accs = []
        for b in bench_names:
            if b in all_results[enc]:
                acc = all_results[enc][b]["accuracy"]
                row += f" | {acc:>9.1f}%"
                accs.append(acc)
            else:
                row += f" | {'N/A':>10s}"
        avg = sum(accs) / len(accs) if accs else 0
        row += f" | {avg:>7.1f}%"
        print(row, flush=True)

    print(f"{'=' * 70}", flush=True)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import argparse

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Evaluate trained VLM models")
    parser.add_argument(
        "--encoder", type=str, default=None,
        choices=["vit", "clip", "ijepa"],
        help="Evaluate a single encoder (default: all trained).",
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        choices=list(BENCHMARKS.keys()),
        help="Evaluate on a single benchmark (default: all).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=1000,
        help="Max samples per benchmark (default: 1000).",
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs",
        help="Directory with trained model weights.",
    )
    args = parser.parse_args()

    encoders = [args.encoder] if args.encoder else None
    benchmarks = [args.benchmark] if args.benchmark else None

    results = run_full_eval(
        encoder_names=encoders,
        benchmarks=benchmarks,
        save_dir=args.save_dir,
        max_samples=args.max_samples,
    )
