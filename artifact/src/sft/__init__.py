"""SFT module with multiple backend implementations.

Submodules:
- trl: TRL+PEFT-based SFT (local GPU)
- unsloth: Unsloth-based SFT (local GPU)

Import submodules directly to avoid slow unsloth initialization:
    from src.sft.unsloth import train as unsloth_train
"""

__all__ = ["trl", "unsloth"]
