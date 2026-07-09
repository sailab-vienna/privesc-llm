"""TRL + PEFT based SFT implementation.

Uses standard HuggingFace transformers model loading with PEFT LoRA,
with bf16 training and LoRA on all-linear layers including `lm_head`.

Usage:
    source .env && uv run --group training python -m src.sft.trl.train
"""
