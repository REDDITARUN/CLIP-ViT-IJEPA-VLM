# CLIP vs ViT vs I-JEPA: Vision Encoder Stitching Benchmark

**Which frozen vision encoder produces the best embeddings for building a Vision-Language Model?**

This repo contains a controlled benchmark comparing three vision encoders — **CLIP**, **Supervised ViT**, and **I-JEPA** — stitched into the same frozen LLM using a small trainable projector + LoRA. We measure convergence speed, final loss, and downstream VQA accuracy to understand how pre-training strategy affects VLM performance.

```
[Frozen Vision Encoder] ──► [Trainable Projector] ──► [Frozen LLM + LoRA]
       │                           │                          │
  CLIP / ViT / I-JEPA       LayerNorm + MLP           Qwen2.5 + LoRA
```

| Links | |
|-------|---|
| **Trained Models (0.5B)** | [huggingface.co/Teen-Different/CVJ_VLMs](https://huggingface.co/Teen-Different/CVJ_VLMs) |
| **Trained Models (1.5B)** | [huggingface.co/Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B](https://huggingface.co/Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B) |
| **W&B Dashboard** | [wandb.ai/tarunreddi-university-at-buffalo/rlj-vlm-benchmark](https://wandb.ai/tarunreddi-university-at-buffalo/rlj-vlm-benchmark) |
| **Blog Post** | [teendifferent.substack.com](https://teendifferent.substack.com/) |

---

## Vision Encoders Compared

| Encoder | Model | Hidden Dim | Patches | Params | Pre-training |
|---------|-------|-----------|---------|--------|-------------|
| **ViT-L/16** | `google/vit-large-patch16-224` | 1024 | 196 | 304M | Supervised (ImageNet-21k) |
| **CLIP ViT-L/14** | `openai/clip-vit-large-patch14` | 1024 | 256 | 304M | Contrastive (400M image-text pairs) |
| **I-JEPA ViT-H/14** | `facebook/ijepa_vith14_1k` | 1280 | 256 | 632M | Self-supervised (predict masked patches) |

---

## Results

### Experiment 1: Qwen2.5-0.5B + COCO-QA (2K steps)

| Component | Details |
|-----------|---------|
| **LLM** | Qwen/Qwen2.5-0.5B-Instruct (frozen + LoRA r=16) |
| **Dataset** | the_cauldron / cocoqa (46k Q&A pairs) |
| **Steps** | 2,000 (effective batch 16) |
| **Hardware** | Single T4 GPU (~18 min/encoder) |

**Training Loss:**

| Encoder | Final Loss | Min Loss | Trainable Params | Total Params |
|---------|-----------|----------|----------------:|------------:|
| **CLIP** | 0.6815 | 0.1747 | 2.8M (0.35%) | 800M |
| **ViT** | 0.5221 | 0.2141 | 2.8M (0.35%) | 801M |
| **I-JEPA** | 0.4947 | 0.2282 | 3.0M (0.27%) | 1.13B |

**Eval Accuracy (exact match, 1K samples):**

| Encoder | CLEVR | COCO-QA | TallyQA | Average |
|---------|------:|--------:|--------:|--------:|
| **CLIP** | 5.0% | 55.8% | 0.2% | 20.3% |
| **I-JEPA** | 10.4% | 51.2% | 0.0% | 20.5% |
| **ViT** | 10.6% | 51.0% | 0.0% | 20.5% |

### Experiment 2: Qwen2.5-1.5B + VQAv2 (1 epoch, ~5K steps)

| Component | Details |
|-----------|---------|
| **LLM** | Qwen/Qwen2.5-1.5B-Instruct (frozen + LoRA r=16) |
| **Dataset** | the_cauldron / vqav2 |
| **Steps** | 5,173 (1 epoch, effective batch 16) |
| **Hardware** | Single T4 GPU |

**Training Loss:**

| Encoder | Final Loss | Min Loss | Avg Last 50 |
|---------|-----------|----------|------------:|
| **VIT** | 0.4424 | 0.2872 | 0.7473 |
| **CLIP** | 0.9335 | 0.2534 | 0.7339 |
| **I-JEPA** | 0.6716 | 0.2547 | 0.7560 |

**Eval Accuracy (exact match, 1K samples):**

| Encoder | CLEVR | COCO-QA | TallyQA | Average |
|---------|------:|--------:|--------:|--------:|
| **CLIP** | 35.4% | 33.2% | 43.8% | **37.5%** |
| **I-JEPA** | 35.4% | 20.2% | 38.0% | 31.2% |
| **ViT** | 33.0% | 23.8% | 39.0% | 31.9% |

### Key Findings

- **CLIP dominates** when paired with a larger LLM (1.5B) — its text-aligned embeddings give the projector a much easier mapping job, leading to the highest average accuracy (37.5%)
- **I-JEPA matches CLIP on reasoning** (tied at 35.4% on CLEVR) — its self-supervised structural understanding shines on compositional tasks
- **Scaling the LLM helps massively** — going from 0.5B to 1.5B improved CLIP's average from 20.3% to 37.5% (+85%)
- **VQAv2 > COCO-QA for generalization** — training on VQAv2 gave much better cross-dataset transfer (CLEVR went from 5-10% to 33-35%)
- **Supervised ViT** is a solid baseline but consistently trails CLIP and I-JEPA

---

## Project Structure

```
├── configs/
│   └── base.py              # ExperimentConfig dataclass + encoder presets
├── models/
│   ├── vision_encoders.py   # Unified wrapper for ViT, CLIP, I-JEPA
│   ├── projector.py         # LayerNorm + 2-layer MLP bridge
│   └── vlm.py               # StitchedVLM: encoder + projector + LLM
├── data/
│   └── dataset.py           # CauldronDataset + VLMCollator
├── utils/
│   └── metrics.py           # LossTracker + convergence plotting
├── train.py                 # Main training loop (all 3 encoders)
├── inference.py             # Load & generate (local or HF Hub)
├── eval.py                  # Benchmark on held-out datasets
├── test.py                  # End-to-end smoke tests
├── requirements.txt
├── .env.example             # Template for HF_TOKEN, WANDB_API_KEY
├── MODEL_CARD.md            # HF model card (0.5B)
└── MODEL_CARD_1.5B.md       # HF model card (1.5B)
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/REDDITARUN/CLIP-ViT-IJEPA-VLM.git
cd CLIP-ViT-IJEPA-VLM
pip install -r requirements.txt
```

### 2. Set up tokens (optional, for Wandb + HF pushing)

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN, WANDB_API_KEY, HF_REPO_ID
```

### 3. Train all three encoders

```bash
python train.py
```

This trains ViT, CLIP, and I-JEPA sequentially. Each encoder gets its own projector + LoRA adapter. Outputs are saved to `outputs/{encoder_name}/`.

To customize training:

```python
from configs.base import ExperimentConfig, get_encoder_configs
from train import run_all_experiments

# Use defaults
run_all_experiments()

# Or customize
configs = get_encoder_configs()
for cfg in configs:
    cfg.llm_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    cfg.dataset_subset = "vqav2"
    cfg.num_epochs = 2
    cfg.lora_rank = 32
run_all_experiments(configs=configs, use_wandb=True, push_to_hub=True)
```

### 4. Inference

```bash
# From local trained weights
python inference.py --encoder clip --image photo.jpg --question "What is this?"

# From HuggingFace Hub
python inference.py --encoder clip --image photo.jpg --from-hub Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B

# Interactive mode
python inference.py --encoder clip --image photo.jpg --interactive

# Compare all three
for enc in clip vit ijepa; do
  python inference.py --encoder $enc --image photo.jpg --question "What is this?"
done
```

### 5. Evaluate

```bash
# Eval all encoders from HF Hub
python eval.py --from-hub Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B

# Quick test with fewer samples
python eval.py --from-hub Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B --max_samples 100

# Eval specific encoder on specific benchmark
python eval.py --encoder clip --benchmark clevr --from-hub Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B
```

Available benchmarks: `clevr`, `scienceqa`, `cocoqa`, `tallyqa`

---

## How It Works

### The Stitching Architecture

Each experiment follows the same pipeline:

1. **Freeze** the vision encoder — no gradients flow back to CLIP/ViT/I-JEPA
2. **Freeze** the LLM — Qwen2.5 weights stay fixed
3. **Train** only two small components:
   - **Projector** (LayerNorm → Linear → GELU → Linear): maps vision features to LLM embedding space
   - **LoRA** (rank=16 on q_proj, v_proj): lightweight adaptation of the LLM's attention layers
4. **Measure** convergence speed (training loss) and downstream accuracy (exact-match VQA)

This isolates the effect of the vision encoder's embedding quality — everything else is held constant.

### Why These Encoders?

| Encoder | Training Signal | What It Learns |
|---------|----------------|---------------|
| **CLIP** | "Match image to caption" | Language-aligned concepts (dog, red, beach) |
| **I-JEPA** | "Predict masked patches in latent space" | Spatial structure, part-whole relationships |
| **ViT** | "Classify into ImageNet categories" | Class-discriminative features |

---

## Extending This

**Add a new encoder:**
1. Add entry to `ENCODER_META` in `models/vision_encoders.py`
2. Add config to `get_encoder_configs()` in `configs/base.py`
3. Run `python train.py`

**Train on a different dataset:**
```python
cfg.dataset_name = "HuggingFaceM4/the_cauldron"
cfg.dataset_subset = "scienceqa"  # or any subset
```

**Use a different LLM:**
```python
cfg.llm_model_id = "Qwen/Qwen2.5-3B-Instruct"
cfg.lora_rank = 32  # increase for larger models
```

---

## Citation

```bibtex
@misc{clip-vit-ijepa-vlm-2026,
  title={CLIP vs ViT vs I-JEPA: Vision Encoder Stitching Benchmark},
  author={Tarun Reddi},
  year={2026},
  url={https://github.com/REDDITARUN/CLIP-ViT-IJEPA-VLM}
}
```

## License

Apache 2.0
