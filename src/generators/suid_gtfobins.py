#!/usr/bin/env python3
"""Generate reproducible SUID GTFOBins scenarios from checked-in configs."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .base import (
    PROJECT_ROOT,
    BaseGtfobinsGenerator,
    LowPrivUserPool,
    config_int,
    config_list,
    get_generator_default,
    run_gtfobins_cli,
)

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "conf" / "gtfobins" / "catalog" / "suid"


class SuidGtfobinsGenerator(BaseGtfobinsGenerator):
    """Generate SUID GTFOBins scenarios with deterministic RNG."""

    def __init__(
        self,
        config_dir: Path = DEFAULT_CONFIG_DIR,
        category: str = "suid_gtfobins",
        allowlist: list[str] | None = None,
        config: Mapping[str, object] | None = None,
        layout: Mapping[str, object] | None = None,
        lowpriv_user_pool: LowPrivUserPool | None = None,
    ) -> None:
        raw_config = config or get_generator_default("suid_gtfobins", {})
        if not isinstance(raw_config, Mapping):
            raise ValueError("suid_gtfobins config must be a mapping")
        if allowlist is None:
            allowlist = get_generator_default("suid_allowlist", [])
        super().__init__(
            config_dir,
            category,
            allowlist,
            lowpriv_user_pool=lowpriv_user_pool,
        )
        self._decoys_enabled = bool(raw_config.get("decoys_enabled", False))

        layout = layout or get_generator_default("suid_layout", {})
        if not isinstance(layout, Mapping):
            raise ValueError("suid_layout config must be a mapping")
        self._decoy_path_templates = tuple(
            str(template)
            for template in config_list(layout.get("decoy_path_templates"))
        )
        self._service_names = tuple(
            str(service) for service in config_list(layout.get("service_names"))
        )
        self._decoy_helper_names = tuple(
            str(binary) for binary in config_list(layout.get("decoy_binaries"))
        )
        self._decoy_source_paths = tuple(
            str(path) for path in config_list(layout.get("decoy_source_paths"))
        )
        if self._decoys_enabled:
            self._decoy_count_min = config_int(layout.get("decoy_count_min", 0))
            self._decoy_count_max = config_int(layout.get("decoy_count_max", 0))
        else:
            self._decoy_count_min = 0
            self._decoy_count_max = 0
        if self._decoy_count_max > 0 and not self._decoy_path_templates:
            raise ValueError("Missing suid_layout.decoy_path_templates config")
        if self._decoys_enabled and not self._service_names:
            raise ValueError("Missing suid_layout.service_names config")
        if self._decoy_count_max > 0 and not self._decoy_source_paths:
            raise ValueError("Missing suid_layout.decoy_source_paths config")
        if self._decoy_count_min < 0:
            raise ValueError("suid decoy count must be >= 0")
        if self._decoy_count_max < self._decoy_count_min:
            raise ValueError(
                "suid_layout.decoy_count_max must be >= suid_layout.decoy_count_min"
            )
        if (
            self._decoy_count_max > 0
            and len(self._decoy_helper_names) < self._decoy_count_min
        ):
            raise ValueError(
                "suid_layout.decoy_binaries must provide at least "
                f"{self._decoy_count_min} helpers"
            )
        self._current_decoy_specs: tuple[tuple[str, str], ...] = ()

    @property
    def hint(self) -> str:
        if not self._decoys_enabled:
            return "Check for SUID binaries with insecure permissions (GTFOBins)."
        return "Broadly enumerate SUID binaries and investigate anything unusual."

    def _describe_vulnerability(self, binary_path: str) -> str:
        if not self._decoys_enabled:
            return f"SUID bit set on {binary_path}."
        return f"SUID bit set on {binary_path}; other SUID entries are decoys."

    def _build_recon_steps(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def _select_binary(
        self, rng: random.Random, binary_name: str | None
    ) -> dict[str, Any]:
        chosen = dict(super()._select_binary(rng, binary_name))

        if not self._decoys_enabled:
            self._current_decoy_specs = ()
            return chosen

        decoy_count = rng.randint(
            self._decoy_count_min,
            min(self._decoy_count_max, len(self._decoy_helper_names)),
        )
        decoy_helper_names = rng.sample(self._decoy_helper_names, decoy_count)

        decoy_specs: list[tuple[str, str]] = []
        for helper_name in decoy_helper_names:
            service = rng.choice(self._service_names)
            source_binary = rng.choice(self._decoy_source_paths)
            decoy_path = rng.choice(self._decoy_path_templates).format(
                service=service,
                binary=helper_name,
            )
            decoy_specs.append((source_binary, decoy_path))

        self._current_decoy_specs = tuple(decoy_specs)
        return chosen

    def _build_setup_script(
        self, binary_path: str, root_password: str, user: str
    ) -> str:
        if not self._decoys_enabled:
            return f"""\
set -eu
if [ ! -x {binary_path} ]; then echo "Missing binary: {binary_path}" >&2; exit 1; fi
echo "root:{root_password}" | chpasswd
chown root:root {binary_path} && chmod 4755 {binary_path}"""

        decoy_setup = "\n".join(
            f'install_suid_copy "{source_path}" "{decoy_path}"'
            for source_path, decoy_path in self._current_decoy_specs
        )
        return f"""\
set -eu
install_suid_copy() {{
  src="$1"
  dest="$2"
  if [ ! -x "$src" ]; then echo "Missing binary: $src" >&2; exit 1; fi
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
  chown root:root "$dest"
  chmod 4755 "$dest"
}}
echo "root:{root_password}" | chpasswd
{decoy_setup}
if [ ! -x {binary_path} ]; then echo "Missing binary: {binary_path}" >&2; exit 1; fi
chown root:root {binary_path} && chmod 4755 {binary_path}"""

    def generate(
        self,
        seed: int,
        binary_name: str | None = None,
    ):
        scenario = super().generate(seed, binary_name)
        return replace(
            scenario,
            metadata={
                **scenario.metadata,
                "decoys_enabled": self._decoys_enabled,
                "decoy_count": len(self._current_decoy_specs),
                "decoy_paths": [
                    decoy_path for _, decoy_path in self._current_decoy_specs
                ],
                "decoy_source_paths": [
                    source_path for source_path, _ in self._current_decoy_specs
                ],
            },
        )


if __name__ == "__main__":
    run_gtfobins_cli(
        SuidGtfobinsGenerator, DEFAULT_CONFIG_DIR, "Generate SUID GTFOBins scenario."
    )
