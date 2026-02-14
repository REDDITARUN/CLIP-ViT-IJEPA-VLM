"""Stitched Vision-Language Model: frozen encoder + projector + LoRA LLM."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from .vision_encoders import VisionEncoderWrapper, ENCODER_META
from .projector import Projector


class StitchedVLM(nn.Module):
    """A VLM assembled by stitching a frozen vision encoder into an LLM
    through a small trainable projector, with optional LoRA on the LLM.

    Forward flow:
        image -> encoder -> projector -> visual_tokens
        text  -> llm.embed_tokens      -> text_tokens
        concat [visual_tokens, text_tokens] -> llm -> logits
        loss on caption portion only (instruction + visual tokens masked)
    """

    def __init__(
        self,
        encoder_model_id: str,
        encoder_type: str,
        llm_model_id: str,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.dtype = dtype

        # ---- Vision Encoder (frozen) ----
        self.encoder = VisionEncoderWrapper(
            model_id=encoder_model_id,
            encoder_type=encoder_type,
            dtype=dtype,
        )
        vision_dim = self.encoder.hidden_dim

        # ---- LLM ----
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            dtype=dtype,
        )
        llm_dim = self.llm.config.hidden_size  # 896 for Qwen2.5-0.5B

        # Freeze base LLM
        self.llm.requires_grad_(False)

        # ---- LoRA (optional) ----
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules or ["q_proj", "v_proj"],
            )
            self.llm = get_peft_model(self.llm, lora_config)

        # ---- Projector (trainable, same dtype as encoder/LLM) ----
        self.projector = Projector(vision_dim=vision_dim, llm_dim=llm_dim).to(dtype)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_trainable_parameters(self):
        """Yield all trainable parameters (projector + LoRA)."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Full forward pass: encode image, project, concat with text, LM loss.

        Args:
            pixel_values: ``(B, C, H, W)`` preprocessed images.
            input_ids: ``(B, T)`` tokenised text (instruction + caption).
            attention_mask: ``(B, T)`` attention mask for text.
            labels: ``(B, T)`` token ids for loss; positions to ignore
                    should be set to ``-100``.

        Returns:
            dict with ``loss`` and ``logits``.
        """
        device = pixel_values.device

        # 1. Vision tokens  (B, N_patches, vision_dim)
        vision_tokens = self.encoder(pixel_values)

        # 2. Project to LLM dim  (B, N_patches, llm_dim)
        visual_embeds = self.projector(vision_tokens.to(self.dtype))

        # 3. Text embeddings  (B, T, llm_dim)
        #    Access the base model's embed_tokens through peft wrapper.
        #    PeftModel wraps: PeftModel.model = CausalLM -> CausalLM.model = BaseModel
        if isinstance(self.llm, PeftModel):
            embed_layer = self.llm.model.model.embed_tokens
        else:
            embed_layer = self.llm.model.embed_tokens
        text_embeds = embed_layer(input_ids)

        # 4. Concatenate  [visual_tokens | text_tokens]
        #    Shape: (B, N_patches + T, llm_dim)
        B, N, _ = visual_embeds.shape
        T = text_embeds.shape[1]

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # 5. Build attention mask for combined sequence
        vision_mask = torch.ones(B, N, dtype=attention_mask.dtype, device=device)
        combined_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # 6. Build labels: -100 for visual token positions, then the text labels
        vision_labels = torch.full(
            (B, N), fill_value=-100, dtype=labels.dtype, device=device
        )
        combined_labels = torch.cat([vision_labels, labels], dim=1)

        # 7. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}
