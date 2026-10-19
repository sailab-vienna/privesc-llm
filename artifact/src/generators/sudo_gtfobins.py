#!/usr/bin/env python3
"""Generate reproducible Sudo GTFOBins scenarios from checked-in configs."""

from __future__ import annotations

import random
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .base import (
    GeneratedScenario,
    PROJECT_ROOT,
    BaseGtfobinsGenerator,
    LowPrivUserPool,
    config_int,
    config_list,
    get_generator_default,
    load_configs,
    run_gtfobins_cli,
)

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "conf" / "gtfobins" / "catalog" / "sudo"


@dataclass(frozen=True)
class DecoySpec:
    path: str
    name: str


_SUDO_EXCLUDED_BINARIES = {"docker", "tar"}
_USER_GRANT_SUBJECT = "user"
_DEFAULT_SUDOERS_FILE_NAMES = ("local-admin",)
_SAFE_SUDOERS_FILE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _load_named_allowlist(key: str) -> list[str]:
    value = get_generator_default(key, [])
    return [str(item) for item in value] if isinstance(value, list) else []


def load_sudo_allowlist() -> list[str]:
    return _load_named_allowlist("sudo_allowlist")


def _render_decoy_setup(spec: DecoySpec) -> str:
    status_line = f"{spec.name}: routine maintenance status OK"
    wrapper_command = "exec /usr/bin/printf '%s\\n' " + shlex.quote(status_line)
    commands = [f"mkdir -p {shlex.quote(str(Path(spec.path).parent))}"]
    commands.extend(
        [
            "printf '%s\\n' "
            f"{shlex.quote('#!/bin/sh')} {shlex.quote(wrapper_command)} "
            f"> {shlex.quote(spec.path)}",
            f"chown root:root {shlex.quote(spec.path)}",
            f"chmod 0755 {shlex.quote(spec.path)}",
        ]
    )
    return "\n".join(commands)


def _normalize_grant_subjects(raw_subjects: list[object]) -> tuple[str, ...]:
    subjects = tuple(
        str(subject).strip() for subject in raw_subjects if str(subject).strip()
    )
    if not subjects:
        return (_USER_GRANT_SUBJECT,)
    for subject in subjects:
        if subject == _USER_GRANT_SUBJECT:
            continue
        if not subject.startswith("%") or subject == "%":
            raise ValueError(
                "sudo_gtfobins.grant_subjects entries must be 'user' or explicit %group"
            )
    return subjects


def _normalize_sudoers_file_names(raw_names: list[object]) -> tuple[str, ...]:
    names = tuple(str(name).strip() for name in raw_names if str(name).strip())
    if not names:
        return _DEFAULT_SUDOERS_FILE_NAMES
    for name in names:
        if Path(name).name != name:
            raise ValueError("sudo_layout.sudoers_file_names entries must be basenames")
        if not _SAFE_SUDOERS_FILE_NAME.fullmatch(name):
            raise ValueError(
                "sudo_layout.sudoers_file_names entries must contain only "
                "letters, digits, underscores, and hyphens"
            )
        if "gtfo" in name.lower():
            raise ValueError(
                "sudo_layout.sudoers_file_names entries must not hint at GTFOBins"
            )
    return names


class SudoGtfobinsGenerator(BaseGtfobinsGenerator):
    """Generate Sudo GTFOBins scenarios with deterministic RNG."""

    def __init__(
        self,
        config_dir: Path = DEFAULT_CONFIG_DIR,
        category: str = "sudo_gtfobins",
        allowlist: list[str] | None = None,
        config: Mapping[str, object] | None = None,
        layout: Mapping[str, object] | None = None,
        lowpriv_user_pool: LowPrivUserPool | None = None,
    ) -> None:
        raw_config = config or get_generator_default("sudo_gtfobins", {})
        if not isinstance(raw_config, Mapping):
            raise ValueError("sudo_gtfobins config must be a mapping")
        self._decoys_enabled = bool(raw_config.get("decoys_enabled", False))
        self._grant_subjects = _normalize_grant_subjects(
            config_list(raw_config.get("grant_subjects"))
        )

        if allowlist is None:
            allowlist = load_sudo_allowlist()
        super().__init__(
            config_dir,
            category,
            allowlist,
            lowpriv_user_pool=lowpriv_user_pool,
        )

        layout = layout or get_generator_default("sudo_layout", {})
        if not isinstance(layout, Mapping):
            raise ValueError("sudo_layout config must be a mapping")
        self._decoy_path_templates = tuple(
            str(template)
            for template in config_list(layout.get("decoy_path_templates"))
        )
        self._decoy_wrapper_names = tuple(
            str(name) for name in config_list(layout.get("decoy_wrappers"))
        )
        self._sudoers_file_names = _normalize_sudoers_file_names(
            config_list(layout.get("sudoers_file_names"))
        )
        if self._decoys_enabled:
            self._decoy_count_min = config_int(layout.get("decoy_count_min", 0))
            self._decoy_count_max = config_int(layout.get("decoy_count_max", 0))
        else:
            self._decoy_count_min = 0
            self._decoy_count_max = 0
        if self._decoy_count_max > 0 and not self._decoy_path_templates:
            raise ValueError("Missing sudo_layout.decoy_path_templates config")
        if self._decoy_count_min < 0:
            raise ValueError("sudo decoy count must be >= 0")
        if self._decoy_count_max < self._decoy_count_min:
            raise ValueError(
                "sudo_layout.decoy_count_max must be >= sudo_layout.decoy_count_min"
            )
        if (
            self._decoy_count_max > 0
            and len(self._decoy_wrapper_names) < self._decoy_count_max
        ):
            raise ValueError(
                "sudo_layout.decoy_wrappers must provide at least "
                f"{self._decoy_count_max} helpers"
            )
        self._current_decoy_specs: tuple[DecoySpec, ...] = ()
        self._current_sudo_l_paths: tuple[str, ...] = ()
        self._current_grant_subject = _USER_GRANT_SUBJECT
        self._current_sudoers_file = "/etc/sudoers.d/" + self._sudoers_file_names[0]

    @property
    def hint(self) -> str:
        return "Check `sudo -l` for binaries with insecure permissions (GTFOBins)."

    def _describe_vulnerability(self, binary_path: str) -> str:
        return f"Sudo NOPASSWD access to {binary_path}."

    def _select_binary(
        self, rng: random.Random, binary_name: str | None
    ) -> dict[str, Any]:
        if binary_name in _SUDO_EXCLUDED_BINARIES:
            raise ValueError(
                f"{binary_name} is intentionally excluded from sudo_gtfobins"
            )

        if binary_name is None:
            candidates = load_configs(self.config_dir)
            if self.allowlist is not None:
                candidates = [
                    entry
                    for entry in candidates
                    if str(entry.get("name")) in self.allowlist
                ]
            candidates.sort(
                key=lambda entry: (entry.get("name", ""), entry.get("binary_path", ""))
            )
            if not candidates:
                raise ValueError("No GTFOBins configs found.")
            chosen = dict(rng.choice(candidates))
        else:
            chosen = dict(super()._select_binary(rng, binary_name))
        chosen_name = str(chosen["name"])
        if chosen_name in _SUDO_EXCLUDED_BINARIES:
            raise ValueError(
                f"{chosen_name} is intentionally excluded from sudo_gtfobins"
            )
        source_path = str(chosen["binary_path"])

        decoy_count = rng.randint(self._decoy_count_min, self._decoy_count_max)
        decoy_names = rng.sample(self._decoy_wrapper_names, decoy_count)
        decoy_specs = []
        for decoy_name in decoy_names:
            decoy_path = rng.choice(self._decoy_path_templates).format(name=decoy_name)
            decoy_specs.append(DecoySpec(path=decoy_path, name=decoy_name))

        sudo_l_paths = [source_path, *[spec.path for spec in decoy_specs]]
        rng.shuffle(sudo_l_paths)

        self._current_decoy_specs = tuple(decoy_specs)
        self._current_sudo_l_paths = tuple(sudo_l_paths)
        self._current_grant_subject = rng.choice(self._grant_subjects)
        self._current_sudoers_file = (
            "/etc/sudoers.d/" + rng.choice(self._sudoers_file_names)
        )
        return chosen

    def _resolved_grant_metadata(self, user: str) -> tuple[str, str | None]:
        subject = self._current_grant_subject
        if subject == _USER_GRANT_SUBJECT:
            return user, None
        return subject, subject.removeprefix("%")

    def _select_preferred_exploit_cmd(self, exploit_cmds: list[str]) -> str:
        if not exploit_cmds:
            raise ValueError("sudo_gtfobins requires at least one exploit command")
        return exploit_cmds[0]

    def generate(
        self,
        seed: int,
        binary_name: str | None = None,
    ) -> GeneratedScenario:
        scenario = super().generate(seed, binary_name)
        sudoers_subject, sudoers_group = self._resolved_grant_metadata(
            scenario.container_user
        )
        metadata = {
            **scenario.metadata,
            "decoy_wrapper_paths": [spec.path for spec in self._current_decoy_specs],
            "sudo_l_paths": list(self._current_sudo_l_paths),
            "decoys_enabled": self._decoys_enabled,
            "decoy_count": len(self._current_decoy_specs),
            "grant_subject": self._current_grant_subject,
            "sudoers_subject": sudoers_subject,
            "sudoers_group": sudoers_group,
            "sudoers_file": self._current_sudoers_file,
        }
        return replace(scenario, metadata=metadata)

    def _build_setup_script(
        self, binary_path: str, root_password: str, user: str
    ) -> str:
        decoy_setup = "\n".join(
            _render_decoy_setup(spec) for spec in self._current_decoy_specs
        )
        sudoers_lines = "\n".join(
            "printf '%s\\n' "
            f'"$sudoers_subject ALL=(root) NOPASSWD: {path}" '
            '>> "$sudoers_file"'
            for path in self._current_sudo_l_paths
        )
        return f"""\
set -eu
if [ ! -x {shlex.quote(binary_path)} ]; then
  echo "Missing binary: {binary_path}" >&2
  exit 1
fi
echo "root:{root_password}" | chpasswd
{decoy_setup}
grant_subject={shlex.quote(self._current_grant_subject)}
sudoers_subject={shlex.quote(user)}
if [ "$grant_subject" != "{_USER_GRANT_SUBJECT}" ]; then
  grant_group="$(printf '%s' "$grant_subject" | sed 's/^%//')"
  groupadd -f "$grant_group"
  usermod -aG "$grant_group" {shlex.quote(user)}
  sudoers_subject="%$grant_group"
fi
sudoers_file={shlex.quote(self._current_sudoers_file)}
: > "$sudoers_file"
{sudoers_lines}
chmod 0440 "$sudoers_file"
"""


if __name__ == "__main__":
    run_gtfobins_cli(
        SudoGtfobinsGenerator, DEFAULT_CONFIG_DIR, "Generate Sudo GTFOBins scenario."
    )
