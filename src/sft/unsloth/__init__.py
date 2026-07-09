"""Unsloth-based SFT implementation.

Importing Unsloth is intentionally deferred so the package can be imported in
environments that do not have the optional local-training dependencies.

Usage:
    source .env && uv run python -m src.sft.unsloth.train
"""


def main() -> None:
    from .train import main as _main

    _main()


__all__ = ["main"]
