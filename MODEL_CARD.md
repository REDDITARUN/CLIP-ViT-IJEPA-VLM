---
library_name: peft
base_model:
  - Qwen/Qwen2.5-0.5B-Instruct
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

# CVJ_VLMs — Vision Encoder Stitching Benchmark (0.5B)

**Which frozen vision encoder produces the best embeddings for a VLM?**

This repo contains trained **projector weights + LoRA adapters** from a controlled experiment comparing three vision encoders stitched into **Qwen2.5-0.5B-Instruct**. Trained on **COCO-QA** (46k Q&A pairs) for 2,000 steps.

> **See also:** [1.5B version (VQAv2 training)](https://huggingface.co/Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B)

| Links | |
|-------|---|
| **Training Code** | [github.com/REDDITARUN/CLIP-ViT-IJEPA-VLM](https://github.com/REDDITARUN/CLIP-ViT-IJEPA-VLM) |
| **1.5B Version** | [Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B](https://huggingface.co/Teen-Different/CLIP-ViT-IJEPA-VLMs-1.5B) |
| **W&B Dashboard** | [wandb.ai/tarunreddi-university-at-buffalo/rlj-vlm-benchmark](https://wandb.ai/tarunreddi-university-at-buffalo/rlj-vlm-benchmark) |
| **Blog Post** | [teendifferent.substack.com](https://teendifferent.substack.com/) |

---

## The Experiment

```
[Frozen Vision Encoder] ──► [Trainable Projector] ──► [Qwen-0.5B + LoRA]
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
| **LLM** | Qwen/Qwen2.5-0.5B-Instruct (frozen + LoRA rank=16) |
| **Projector** | LayerNorm → Linear → GELU → Linear |
| **Dataset** | HuggingFaceM4/the_cauldron (cocoqa subset, 46k Q&A pairs) |
| **Optimizer** | AdamW, lr=1e-4, linear warmup 100 steps |
| **Steps** | 2,000 optimizer steps, effective batch size 16 |
| **Hardware** | Single T4 GPU (~18 min per encoder) |

---

## Results

### Training Loss

| Encoder | Final Loss | Min Loss | Trainable Params | Total Params |
|---------|-----------|----------|----------------:|------------:|
| **CLIP** | 0.6815 | 0.1747 | 2,805,504 (0.35%) | 800,018,048 |
| **ViT** | 0.5221 | 0.2141 | 2,805,504 (0.35%) | 801,189,504 |
| **I-JEPA** | 0.4947 | 0.2282 | 3,035,392 (0.27%) | 1,127,830,400 |

### Evaluation Accuracy (exact match, 1K samples per benchmark)

| Encoder | CLEVR | COCO-QA | TallyQA | Average |
|---------|------:|--------:|--------:|--------:|
| **CLIP** | 5.0% | **55.8%** | 0.2% | 20.3% |
| **I-JEPA** | **10.4%** | 51.2% | 0.0% | **20.5%** |
| **ViT** | 10.6% | 51.0% | 0.0% | 20.5% |

### Sample Inference (Golden Retriever Image)

| Question | CLIP | ViT | I-JEPA |
|----------|------|-----|--------|
| "Describe the image?" | Dog | Dog | Dog |
| "Is the dog's tongue out?" | Yes | Yes | Yes |
| "What color is the fur?" | Dog | Yellow | Brown |
| "What color is the dog?" | Fur | Brown | Brown |
| "What animal is this?" | Dog | Dog | Dog |

### Key Findings (0.5B)

- **CLIP excels on COCO-QA** (55.8%) — its text-aligned embeddings match the Q&A format well
- **I-JEPA and ViT lead on CLEVR** (~10%) — better at compositional/spatial reasoning
- All three struggle on **TallyQA** (counting) with only 0.5B LLM and short training
- Answers are short (1 word) because COCO-QA uses short Q&A format

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

### Inference (standalone — no repo clone needed)

```python
import json, torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "Teen-Different/CVJ_VLMs"
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
python inference.py --encoder clip --image photo.jpg --from-hub Teen-Different/CVJ_VLMs

# Evaluate
python eval.py --from-hub Teen-Different/CVJ_VLMs

# Interactive
python inference.py --encoder clip --image photo.jpg --from-hub Teen-Different/CVJ_VLMs --interactive
```

---

## What Makes Each Encoder Different?

| Encoder | What the embeddings encode | Trained on | Best for |
|---------|---------------------------|-----------|----------|
| **CLIP** | Language-aligned concepts ("dog", "red", "beach") | 400M image-text pairs | Fast convergence, concept questions |
| **I-JEPA** | Spatial structure, part-whole relations, layout | ImageNet (self-supervised) | Spatial reasoning, compositional tasks |
| **ViT** | ImageNet class-discriminative features | ImageNet-21k (supervised) | Classification-style tasks |

---

## Limitations

- **0.5B LLM** — small language model limits answer quality and length
- **Short answers** — trained on COCO-QA which has 1-3 word answers
- **2,000 steps** — relatively short training; longer training would improve all encoders
- **Single dataset** — trained only on COCO-QA; multi-dataset training would improve generalization

---

## Citation

```bibtex
@misc{cvj-vlms-0.5b-2026,
  title={CLIP vs ViT vs I-JEPA: Vision Encoder Stitching Benchmark (0.5B)},
  author={Tarun Reddi},
  year={2026},
  url={https://huggingface.co/Teen-Different/CVJ_VLMs}
}
```
