---
library_name: peft
base_model:
  - Qwen/Qwen2.5-1.5B-Instruct
  - openai/clip-vit-large-patch14
  - google/vit-large-patch16-224
  - facebook/ijepa_vith14_1k
datasets:
  - HuggingFaceM4/the_cauldron
tags:
  - vision-language
  - vlm
  - model-stitching
  - clip
  - ijepa
  - vit
  - lora
  - benchmark
  - embedding-comparison
language:
  - en
pipeline_tag: image-text-to-text
license: apache-2.0
---

# CLIP-ViT-IJEPA-VLMs-1.5B — Vision Encoder Stitching Benchmark (1.5B)

**Which frozen vision encoder produces the best embeddings for a VLM?**

This repo contains trained **projector weights + LoRA adapters** from a controlled experiment comparing three vision encoders stitched into **Qwen2.5-1.5B-Instruct**. Trained on **VQAv2** for 1 full epoch (~5,173 steps).

> **See also:** [0.5B version (COCO-QA training)](https://huggingface.co/Teen-Different/CVJ_VLMs)

| Links | |
|-------|---|
| **Training Code** | [github.com/REDDITARUN/CLIP-ViT-IJEPA-VLM](https://github.com/REDDITARUN/CLIP-ViT-IJEPA-VLM) |
| **0.5B Version** | [Teen-Different/CVJ_VLMs](https://huggingface.co/Teen-Different/CVJ_VLMs) |
| **W&B Dashboard** | [wandb.ai/tarunreddi-university-at-buffalo/rlj-vlm-benchmark](https://wandb.ai/tarunreddi-university-at-buffalo/rlj-vlm-benchmark) |
| **Blog Post** | [teendifferent.substack.com](https://teendifferent.substack.com/) |

---

## The Experiment

```
[Frozen Vision Encoder] ──► [Trainable Projector] ──► [Qwen-1.5B + LoRA]
         │                          │                         │
    CLIP / ViT / I-JEPA      LayerNorm+MLP           rank=16, q_proj+v_proj
```

### Vision Encoders Compared

| Encoder | Model | Hidden Dim | Patches | Params | Pre-training |
|---------|-------|-----------|---------|--------|-------------|
| **ViT-L/16** | `google/vit-large-patch16-224` | 1024 | 196 | 304M | Supervised (ImageNet-21k) |
| **CLIP ViT-L/14** | `openai/clip-vit-large-patch14` | 1024 | 256 | 304M | Contrastive (400M image-text pairs) |
| **I-JEPA ViT-H/14** | `facebook/ijepa_vith14_1k` | 1280 | 256 | 632M | Self-supervised (predict masked patches) |

### Training Setup

| Component | Details |
|-----------|---------|
| **LLM** | Qwen/Qwen2.5-1.5B-Instruct (frozen + LoRA rank=16) |
| **Projector** | LayerNorm → Linear → GELU → Linear |
| **Dataset** | HuggingFaceM4/the_cauldron (vqav2 subset) |
| **Optimizer** | AdamW, lr=1e-4, linear warmup 100 steps |
| **Steps** | 5,173 (1 full epoch, effective batch size 16) |
| **Hardware** | Single T4 GPU |

---

## Results

### Training Loss

| Encoder | Final Loss | Min Loss | Avg Last 50 | Trainable Params |
|---------|-----------|----------|------------:|-----------------:|
| **VIT** | 0.4424 | 0.2872 | 0.7473 | ~3M |
| **CLIP** | 0.9335 | 0.2534 | 0.7339 | ~3M |
| **I-JEPA** | 0.6716 | 0.2547 | 0.7560 | ~3M |

### Evaluation Accuracy (exact match, 1K samples per benchmark)

| Encoder | CLEVR | COCO-QA | TallyQA | Average |
|---------|------:|--------:|--------:|--------:|
| **CLIP** | 35.4% | **33.2%** | **43.8%** | **37.5%** |
| **I-JEPA** | **35.4%** | 20.2% | 38.0% | 31.2% |
| **ViT** | 33.0% | 23.8% | 39.0% | 31.9% |

### Comparison: 0.5B vs 1.5B

| Encoder | 0.5B Avg | 1.5B Avg | Improvement |
|---------|:--------:|:--------:|:-----------:|
| **CLIP** | 20.3% | **37.5%** | +85% |
| **I-JEPA** | 20.5% | 31.2% | +52% |
| **ViT** | 20.5% | 31.9% | +56% |

### Key Findings

- **CLIP dominates** with the 1.5B LLM — highest average accuracy (37.5%), strongest on COCO-QA and TallyQA
- **I-JEPA ties CLIP on CLEVR** (both 35.4%) — its structural embeddings shine on compositional reasoning
- **Scaling the LLM is the biggest lever** — CLIP improved +85% going from 0.5B to 1.5B
- **VQAv2 training transfers well** — much better cross-dataset generalization than COCO-QA training
- **CLIP's text-alignment advantage grows with scale** — the gap between CLIP and others widens with a larger LLM

---

## Repo Structure

```
├── clip/
│   ├── projector.pt           # Trained MLP bridge weights
│   ├── lora_adapter/          # LoRA adapter (loadable via peft)
│   ├── loss_history.json      # Per-step training loss
│   └── config.json            # Full experiment config
├── vit/
│   └── ... (same structure)
└── ijepa/
    └── ... (same structure)
```

---

## Quick Start

### Install

```bash
pip install torch transformers peft Pillow huggingface-hub
```

### Inference (standalone)

```python
import json, torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B"
ENCODER = "clip"  # or "vit" or "ijepa"

# Download weights
cfg = json.load(open(hf_hub_download(REPO, f"{ENCODER}/config.json")))
proj_path = hf_hub_download(REPO, f"{ENCODER}/projector.pt")
lora_dir = snapshot_download(REPO, allow_patterns=f"{ENCODER}/lora_adapter/*")
lora_path = f"{lora_dir}/{ENCODER}/lora_adapter"

# Load encoder (example for CLIP)
from transformers import CLIPVisionModel, CLIPImageProcessor
encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14",
                                           torch_dtype=torch.bfloat16).cuda().eval()
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load projector
import torch.nn as nn
class Projector(nn.Module):
    def __init__(self, vd, ld):
        super().__init__()
        self.norm = nn.LayerNorm(vd)
        self.fc1 = nn.Linear(vd, ld)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ld, ld)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))

llm_dim = AutoConfig.from_pretrained(cfg["llm_model_id"]).hidden_size
projector = Projector(1024, llm_dim).to(torch.bfloat16).cuda()
projector.load_state_dict(torch.load(proj_path, map_location="cuda", weights_only=True))
projector.eval()

# Load LLM + LoRA
tokenizer = AutoTokenizer.from_pretrained(cfg["llm_model_id"])
llm = AutoModelForCausalLM.from_pretrained(cfg["llm_model_id"],
                                            torch_dtype=torch.bfloat16).cuda()
llm = PeftModel.from_pretrained(llm, lora_path).cuda().eval()

# Run inference
image = Image.open("photo.jpg").convert("RGB")
pixels = processor(images=[image], return_tensors="pt")["pixel_values"].cuda()

with torch.no_grad():
    vis = encoder(pixel_values=pixels).last_hidden_state[:, 1:, :]  # drop CLS
    vis = projector(vis)
    tok = tokenizer("<image>\nWhat is in this image?", return_tensors="pt").to("cuda")
    txt = llm.model.model.embed_tokens(tok["input_ids"])
    embeds = torch.cat([vis, txt], dim=1)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device="cuda")
    out = llm.generate(inputs_embeds=embeds, attention_mask=mask,
                       max_new_tokens=50, min_new_tokens=3, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Using the CLI (from source repo)

```bash
git clone https://github.com/REDDITARUN/CLIP-ViT-IJEPA-VLM.git && cd CLIP-ViT-IJEPA-VLM
pip install -r requirements.txt

# Inference
python inference.py --encoder clip --image photo.jpg --from-hub Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B

# Evaluate
python eval.py --from-hub Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B

# Interactive
python inference.py --encoder clip --image photo.jpg --from-hub Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B --interactive

# Compare all three
for enc in clip vit ijepa; do
  python inference.py --encoder $enc --image photo.jpg --question "What is this?" \
    --from-hub Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B
done
```

---

## What Makes Each Encoder Different?

| Encoder | What the embeddings encode | Trained on | Best for |
|---------|---------------------------|-----------|----------|
| **CLIP** | Language-aligned concepts ("dog", "red", "beach") | 400M image-text pairs | Fast convergence, concept questions, counting |
| **I-JEPA** | Spatial structure, part-whole relations, layout | ImageNet (self-supervised) | Spatial reasoning, compositional tasks |
| **ViT** | ImageNet class-discriminative features | ImageNet-21k (supervised) | Classification-style tasks |

---

## Limitations

- **1.5B LLM** — still a small language model; larger LLMs would likely improve all results
- **LoRA rank 16** — low-rank adaptation may be insufficient for the 1.5B model; rank 32-64 could help
- **Single epoch** — only 1 pass over VQAv2; more epochs would improve convergence
- **LoRA on q/v only** — targeting more modules (k_proj, o_proj, gate_proj) could help
- **Exact match eval** — strict metric; softer metrics (F1, BERTScore) would show higher accuracy

---

## Citation

```bibtex
@misc{cvj-vlms-1.5b-2026,
  title={CLIP vs ViT vs I-JEPA: Vision Encoder Stitching Benchmark (1.5B)},
  author={Tarun Reddi},
  year={2026},
  url={https://huggingface.co/Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B}
}
```
