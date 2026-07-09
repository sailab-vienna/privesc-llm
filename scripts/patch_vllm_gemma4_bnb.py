"""Apply the vLLM Gemma4 BitsAndBytes quant-state patch.

This applies ``patches/vllm-gemma4-bnb-k-eq-v.patch``, a local copy of the
vLLM PR #40321 delta needed for Gemma4 BnB before the fix is released in the
installed vLLM wheel.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


PATCH_PATH = Path(__file__).resolve().parents[1] / "patches/vllm-gemma4-bnb-k-eq-v.patch"


def _vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vllm is not importable in this environment")
    return Path(spec.origin).parent


def _is_applied(root: Path) -> bool:
    markers = {
        root / "model_executor/model_loader/bitsandbytes_loader.py": (
            "maybe_postprocess_bitsandbytes_quant_state_dict"
        ),
        root / "model_executor/models/gemma4.py": (
            "def _duplicate_k_eq_v_bnb_quant_states("
        ),
        root / "model_executor/models/gemma4_mm.py": (
            "prefix=\"language_model.\""
        ),
    }
    return all(marker in path.read_text() for path, marker in markers.items())


def _run_patch(root: Path, *, dry_run: bool) -> None:
    command = [
        "patch",
        "--forward",
        "--strip=1",
        "--directory",
        str(root),
        "--input",
        str(PATCH_PATH),
    ]
    if dry_run:
        command.insert(1, "--dry-run")
    subprocess.run(command, check=True)


def main() -> None:
    if not PATCH_PATH.is_file():
        raise RuntimeError(f"patch file does not exist: {PATCH_PATH}")

    root = _vllm_root()
    if _is_applied(root):
        print("[INFO] Gemma4 BnB vLLM patch already applied")
        return

    _run_patch(root, dry_run=True)
    _run_patch(root, dry_run=False)
    print(f"[INFO] Applied Gemma4 BnB vLLM patch to {root}")


if __name__ == "__main__":
    main()
