#!/usr/bin/env python3
"""Generate reproducible file-capabilities GTFOBins scenarios."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .base import (
    PROJECT_ROOT,
    BaseGtfobinsGenerator,
    LowPrivUserPool,
    get_generator_file_default,
    run_gtfobins_cli,
)

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "conf" / "gtfobins" / "catalog" / "capabilities"


def _setcap_spec(required_caps: list[str]) -> str:
    caps = []
    for raw in required_caps:
        name = raw.strip()
        if not name:
            continue
        name = name.removeprefix("CAP_").lower()
        if not name.startswith("cap_"):
            name = "cap_" + name
        caps.append(name)
    return ",".join(dict.fromkeys(caps) or ["cap_setuid"]) + "+ep"


class CapabilitiesGtfobinsGenerator(BaseGtfobinsGenerator):
    """Generate GTFOBins scenarios based on dangerous file capabilities."""

    def __init__(
        self,
        config_dir: Path = DEFAULT_CONFIG_DIR,
        category: str = "capabilities_gtfobins",
        allowlist: list[str] | None = None,
        lowpriv_user_pool: LowPrivUserPool | None = None,
    ) -> None:
        if allowlist is None:
            allowlist = get_generator_file_default(
                "capabilities_gtfobins",
                "capabilities_allowlist",
                [],
            )
        super().__init__(
            config_dir,
            category,
            allowlist,
            lowpriv_user_pool=lowpriv_user_pool,
        )
        self._current_setcap_spec = ""
        self._current_required_caps: list[str] = []

    @property
    def hint(self) -> str:
        return "Check for binaries with dangerous file capabilities (`getcap -r /`)."

    def _describe_vulnerability(self, binary_path: str) -> str:
        return f"Dangerous file capabilities set on {binary_path}."

    def _build_recon_steps(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def _select_binary(
        self, rng: random.Random, binary_name: str | None
    ) -> dict[str, Any]:
        chosen = super()._select_binary(rng, binary_name)
        raw_caps = chosen.get("required_caps") or []
        self._current_required_caps = [str(cap) for cap in raw_caps]
        self._current_setcap_spec = _setcap_spec(self._current_required_caps)
        return chosen

    def generate(
        self,
        seed: int,
        binary_name: str | None = None,
    ):
        scenario = super().generate(seed, binary_name=binary_name)
        scenario.metadata["required_caps"] = ",".join(self._current_required_caps)
        scenario.metadata["setcap_spec"] = self._current_setcap_spec
        return scenario

    def _build_setup_script(
        self, binary_path: str, root_password: str, user: str
    ) -> str:
        spec = self._current_setcap_spec or _setcap_spec([])
        return f"""\
set -eu
if [ ! -x {binary_path} ]; then echo "Missing binary: {binary_path}" >&2; exit 1; fi
echo "root:{root_password}" | chpasswd
BIN_REAL="$(readlink -f {binary_path})"
if [ ! -x "$BIN_REAL" ]; then echo "Missing resolved binary: $BIN_REAL" >&2; exit 1; fi
setcap '{spec}' "$BIN_REAL"
"""


if __name__ == "__main__":
    run_gtfobins_cli(
        CapabilitiesGtfobinsGenerator,
        DEFAULT_CONFIG_DIR,
        "Generate file capabilities GTFOBins scenario.",
    )
