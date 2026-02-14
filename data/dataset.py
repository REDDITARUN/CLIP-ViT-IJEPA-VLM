"""Dataset loading and collation for the_cauldron VLM benchmark."""

from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# --------------------------------------------------------------------------- #
#  Prompt template
# --------------------------------------------------------------------------- #

INSTRUCTION_TEMPLATE = "<image>\n{question}"


def _format_conversation(texts: list) -> tuple:
    """Extract (instruction, response) from a the_cauldron conversation.

    The ``texts`` field is a list of dicts with ``user`` and ``assistant`` keys.
    We take the **first** turn only (single-turn Q&A / captioning).

    Returns:
        (instruction_str, response_str)
    """
    turn = texts[0]
    question = turn.get("user", "Describe the image.")
    answer = turn.get("assistant", "")
    instruction = INSTRUCTION_TEMPLATE.format(question=question)
    return instruction, answer


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #


class CauldronDataset(Dataset):
    """Wraps a subset of HuggingFaceM4/the_cauldron for single-image VLM
    training.

    Each ``__getitem__`` returns::

        {
            "image": PIL.Image,
            "instruction": str,
            "response": str,
        }
    """

    def __init__(
        self,
        dataset_name: str = "HuggingFaceM4/the_cauldron",
        subset: str = "cocoqa",
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.ds = load_dataset(dataset_name, subset, split=split)
        if max_samples is not None:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]

        # images is a list (possibly of lists); take the first image
        images = row["images"]
        if isinstance(images, list) and len(images) > 0:
            img = images[0]
        else:
            img = images
        # Ensure PIL Image in RGB
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        instruction, response = _format_conversation(row["texts"])

        return {
            "image": img,
            "instruction": instruction,
            "response": response,
        }


# --------------------------------------------------------------------------- #
#  Collator
# --------------------------------------------------------------------------- #


class VLMCollator:
    """Collates a batch of ``CauldronDataset`` samples into tensors ready
    for ``StitchedVLM.forward()``.

    Responsibilities:
        1. Preprocess images through the encoder's image processor.
        2. Tokenise instruction + response.
        3. Build labels with ``-100`` on instruction tokens (only supervise
           the response / caption).
    """

    def __init__(
        self,
        image_processor,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 256,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Make sure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        images: List[Image.Image] = [sample["image"] for sample in batch]
        instructions: List[str] = [sample["instruction"] for sample in batch]
        responses: List[str] = [sample["response"] for sample in batch]

        # ---- Image processing ----
        pixel_values = self.image_processor(
            images=images, return_tensors="pt"
        )["pixel_values"]

        # ---- Text tokenization ----
        # We tokenise instruction and response separately so we can mask
        # the instruction portion in the labels.
        input_ids_list = []
        labels_list = []

        for instr, resp in zip(instructions, responses):
            # Tokenise instruction (no loss)
            instr_enc = self.tokenizer(
                instr,
                add_special_tokens=True,
                return_attention_mask=False,
            )
            instr_ids = instr_enc["input_ids"]

            # Tokenise response (compute loss here)
            resp_enc = self.tokenizer(
                resp,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            resp_ids = resp_enc["input_ids"]

            # Append EOS
            eos = [self.tokenizer.eos_token_id]
            full_ids = instr_ids + resp_ids + eos

            # Labels: -100 for instruction, actual ids for response + EOS
            full_labels = (
                [-100] * len(instr_ids)
                + resp_ids
                + eos
            )

            # Truncate to max_seq_len
            full_ids = full_ids[: self.max_seq_len]
            full_labels = full_labels[: self.max_seq_len]

            input_ids_list.append(full_ids)
            labels_list.append(full_labels)

        # ---- Pad to same length ----
        max_len = max(len(ids) for ids in input_ids_list)
        padded_ids = []
        padded_labels = []
        attention_masks = []

        pad_id = self.tokenizer.pad_token_id

        for ids, lbls in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
