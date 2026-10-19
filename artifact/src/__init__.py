"""Initialize the privesc-llm package."""

from pathlib import Path

from dotenv import load_dotenv

# Resolve repository root (one level above src/)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Automatically load environment variables from .env if present.
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=False)

__all__ = []
