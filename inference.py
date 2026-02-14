"""Inference script: load trained projector + LoRA and generate answers from images.

Supports loading from:
  1. Local folder  (e.g. outputs/clip/)
  2. HuggingFace Hub repo  (e.g. your-username/rlj-vlm-benchmark)

Usage:
    # Local -- single image
    python inference.py --encoder clip --image photo.jpg --question "What is in this image?"

    # Local -- interactive mode (keep asking questions)
    python inference.py --encoder clip --image photo.jpg --interactive

    # From HuggingFace Hub
    python inference.py --encoder clip --image photo.jpg --from-hub your-username/rlj-vlm-benchmark

    # Use a URL instead of a local file
    python inference.py --encoder clip --image "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
"""

import json
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Optional

import torch
from PIL import Image

from models.vision_encoders import VisionEncoderWrapper, ENCODER_META
from models.projector import Projector


# --------------------------------------------------------------------------- #
#  Silence noisy library logs during loading
# --------------------------------------------------------------------------- #


@contextmanager
def _quiet_load(verbose: bool = False):
    """Suppress transformers/torch loading noise unless verbose=True."""
    if verbose:
        yield
        return

    # Suppress transformers, huggingface_hub, and torch warnings
    prev_verbosity = None
    try:
        import transformers
        prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    try:
        yield
    finally:
        if prev_verbosity is not None:
            transformers.logging.set_verbosity(prev_verbosity)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        warnings.resetwarnings()


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float32
    return torch.bfloat16


def _load_image(source: str) -> Image.Image:
    """Load an image from a local path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        import requests
        from io import BytesIO
        response = requests.get(source, timeout=15)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(source).convert("RGB")


# --------------------------------------------------------------------------- #
#  Model loader
# --------------------------------------------------------------------------- #


class TrainedVLM:
    """Loads a trained projector + LoRA adapter and runs inference.

    This is a lightweight inference wrapper -- it does NOT load the full
    StitchedVLM training class. Instead it loads the three components
    independently:
      1. Frozen vision encoder (from HuggingFace)
      2. Trained projector (from local/hub)
      3. LLM + trained LoRA adapter (from local/hub)
    """

    def __init__(
        self,
        encoder_type: str,
        encoder_model_id: str,
        llm_model_id: str,
        projector_path: str,
        lora_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
    ):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        self.device = device or _get_device()
        self.dtype = dtype or _resolve_dtype(self.device)

        if verbose:
            print(f"Loading on {self.device} with {self.dtype}", flush=True)

        with _quiet_load(verbose=verbose):
            # 1. Vision encoder (frozen)
            self.encoder = VisionEncoderWrapper(
                model_id=encoder_model_id,
                encoder_type=encoder_type,
                dtype=self.dtype,
            ).to(self.device)

            # 2. Projector (trained)
            vision_dim = self.encoder.hidden_dim
            llm_config = AutoConfig.from_pretrained(llm_model_id)
            llm_dim = llm_config.hidden_size

            self.projector = Projector(vision_dim=vision_dim, llm_dim=llm_dim).to(self.dtype)
            state_dict = torch.load(projector_path, map_location="cpu", weights_only=True)
            self.projector.load_state_dict(state_dict)
            self.projector = self.projector.to(self.device)
            self.projector.eval()

            # 3. LLM + LoRA
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_id, dtype=self.dtype,
            ).to(self.device)

            if lora_path and os.path.isdir(lora_path):
                self.llm = PeftModel.from_pretrained(self.llm, lora_path).to(self.device)

            self.llm.eval()

            # Resolve embed_tokens layer
            if isinstance(self.llm, PeftModel):
                self._embed_layer = self.llm.model.model.embed_tokens
            else:
                self._embed_layer = self.llm.model.embed_tokens

        print(f"  Model loaded ({encoder_type}).", flush=True)

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        question: str = "Describe the image.",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Generate a text response given an image and a question.

        Args:
            image: PIL Image.
            question: text prompt / question.
            max_new_tokens: max tokens to generate.
            temperature: sampling temperature (ignored if do_sample=False).
            do_sample: whether to sample or use greedy decoding.

        Returns:
            Generated text string.
        """
        # Encode image
        pixel_values = self.encoder.preprocess([image])["pixel_values"].to(self.device)
        vision_tokens = self.encoder(pixel_values)
        visual_embeds = self.projector(vision_tokens.to(self.dtype))

        # Tokenize question -- must match training format exactly
        prompt = f"<image>\n{question}"
        tok = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        text_embeds = self._embed_layer(tok["input_ids"])

        # Concatenate [visual | text]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        # Generate
        # Qwen2.5 models have *two* EOS ids (<|endoftext|> + <|im_end|>).
        # The instruct-tuned 1.5B sometimes fires <|im_end|> immediately
        # when the prompt doesn't look like a chat template.  Force at
        # least a few tokens so the trained LoRA has a chance to speak.
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=3,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        output_ids = self.llm.generate(**gen_kwargs)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if not text:
            # Fallback: decode WITHOUT stripping special tokens so we can
            # see what the model actually produced (helps debug).
            raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
            print(f"  [DEBUG] Empty output. Raw tokens ({output_ids.shape[-1]}): {raw!r}",
                  flush=True)

        return text


# --------------------------------------------------------------------------- #
#  Load from local folder
# --------------------------------------------------------------------------- #


def load_from_local(
    encoder_name: str,
    save_dir: str = "outputs",
    verbose: bool = False,
) -> TrainedVLM:
    """Load a trained VLM from a local output directory.

    Expects the following structure::

        outputs/clip/
        ├── config.json
        ├── projector.pt
        └── lora_adapter/

    Args:
        encoder_name: one of "vit", "clip", "ijepa".
        save_dir: root outputs directory.
        verbose: show detailed loading logs.

    Returns:
        A ready-to-use TrainedVLM instance.
    """
    folder = os.path.join(save_dir, encoder_name)

    cfg_path = os.path.join(folder, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    projector_path = os.path.join(folder, "projector.pt")
    lora_path = os.path.join(folder, "lora_adapter")
    if not os.path.isdir(lora_path):
        lora_path = None

    return TrainedVLM(
        encoder_type=cfg["encoder_type"],
        encoder_model_id=cfg["encoder_model_id"],
        llm_model_id=cfg["llm_model_id"],
        projector_path=projector_path,
        lora_path=lora_path,
        verbose=verbose,
    )


# --------------------------------------------------------------------------- #
#  Load from HuggingFace Hub
# --------------------------------------------------------------------------- #


def load_from_hub(
    repo_id: str,
    encoder_name: str,
    verbose: bool = False,
) -> TrainedVLM:
    """Download and load a trained VLM from HuggingFace Hub.

    Expects the repo to have::

        clip/config.json
        clip/projector.pt
        clip/lora_adapter/...

    Args:
        repo_id: HuggingFace repo id (e.g. "your-username/rlj-vlm-benchmark").
        encoder_name: one of "vit", "clip", "ijepa".

    Returns:
        A ready-to-use TrainedVLM instance.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN")

    # Download config first
    cfg_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{encoder_name}/config.json",
        token=token,
    )
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Download projector
    projector_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{encoder_name}/projector.pt",
        token=token,
    )

    # Download LoRA adapter folder
    lora_path = None
    try:
        lora_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{encoder_name}/lora_adapter/*",
            token=token,
        )
        candidate = os.path.join(lora_dir, encoder_name, "lora_adapter")
        if os.path.isdir(candidate):
            lora_path = candidate
    except Exception:
        print("  [WARN] Could not download LoRA adapter, continuing without it.", flush=True)

    return TrainedVLM(
        encoder_type=cfg["encoder_type"],
        encoder_model_id=cfg["encoder_model_id"],
        llm_model_id=cfg["llm_model_id"],
        projector_path=projector_path,
        lora_path=lora_path,
        verbose=verbose,
    )


# --------------------------------------------------------------------------- #
#  Interactive mode
# --------------------------------------------------------------------------- #


def interactive_loop(model: TrainedVLM, image: Image.Image):
    """Keep asking questions about the same image until user types 'quit'."""
    print("Interactive mode. Type 'quit' to exit.\n", flush=True)
    while True:
        question = input("Question> ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer = model.generate(image, question=question)
        print(f"Answer: {answer}\n", flush=True)


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

    parser = argparse.ArgumentParser(
        description="Run inference with a trained VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --encoder clip --image photo.jpg
  python inference.py --encoder clip --image photo.jpg --question "What color is the car?"
  python inference.py --encoder clip --image photo.jpg --interactive
  python inference.py --encoder clip --image photo.jpg --from-hub user/repo
        """,
    )
    parser.add_argument(
        "--encoder", type=str, required=True,
        choices=["vit", "clip", "ijepa"],
        help="Which encoder's trained weights to load.",
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to image file or URL.",
    )
    parser.add_argument(
        "--question", type=str, default="Describe the image.",
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--save-dir", type=str, default="outputs",
        help="Local outputs directory (default: outputs/).",
    )
    parser.add_argument(
        "--from-hub", type=str, default=None, metavar="REPO_ID",
        help="Load from HuggingFace Hub instead of local folder.",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive mode: keep asking questions about the image.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed model loading logs.",
    )
    args = parser.parse_args()

    # Load image
    image = _load_image(args.image)

    # Load model
    if args.from_hub:
        model = load_from_hub(
            repo_id=args.from_hub, encoder_name=args.encoder, verbose=args.verbose,
        )
    else:
        model = load_from_local(
            encoder_name=args.encoder, save_dir=args.save_dir, verbose=args.verbose,
        )

    # Run
    if args.interactive:
        interactive_loop(model, image)
    else:
        answer = model.generate(
            image,
            question=args.question,
            max_new_tokens=args.max_tokens,
            do_sample=not args.greedy,
        )
        print(f"\nQuestion: {args.question}")
        print(f"Answer:   {answer}")
