"""Shared utilities and base classes for procedural scenario generators."""

from __future__ import annotations

import argparse
import json
import os
import random
import string
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml

from src.config import ScenarioConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCEDURAL_IMAGE = "privesc_procedural"
GTFOBINS_DEFAULTS = PROJECT_ROOT / "conf" / "gtfobins" / "default.yaml"
GENERATORS_CONFIG_DIR = PROJECT_ROOT / "conf" / "generators"


def config_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def config_int(value: object) -> int:
    return int(str(value))


def _load_generator_config_tree(name: str, seen: set[str]) -> dict[str, Any]:
    if name in seen:
        return {}
    seen.add(name)

    yaml_file = GENERATORS_CONFIG_DIR / f"{name}.yaml"
    if not yaml_file.exists():
        return {}

    data = yaml.safe_load(yaml_file.read_text()) or {}
    if not isinstance(data, dict):
        return {}

    merged: dict[str, Any] = {}
    defaults = data.get("defaults", [])
    if isinstance(defaults, list):
        for entry in defaults:
            if not isinstance(entry, str):
                continue
            child_name = entry.strip()
            if not child_name or child_name == "_self_":
                continue
            merged.update(_load_generator_config_tree(child_name, seen))

    merged.update({key: value for key, value in data.items() if key != "defaults"})
    return merged


@lru_cache(maxsize=1)
def _load_generator_defaults() -> dict[str, Any]:
    """Load generator constants from YAML configs (cached).

    This is a fallback for standalone CLI usage. When running under Hydra,
    config should be passed explicitly via GeneratorsConfig.

    Loads only the YAML files referenced by conf/generators/default.yaml.
    This keeps fallback defaults aligned with the live default config and avoids
    profile-specific files leaking into unrelated runs.
    """
    if not GENERATORS_CONFIG_DIR.exists():
        return {}
    return _load_generator_config_tree("default", set())


@lru_cache(maxsize=None)
def _load_generator_config(name: str) -> dict[str, Any]:
    candidates = [GENERATORS_CONFIG_DIR / f"{name}.yaml"]
    if "/" not in name:
        candidates.extend(
            [
                GENERATORS_CONFIG_DIR / "training" / f"{name}.yaml",
            ]
        )
    yaml_file = next((path for path in candidates if path.exists()), None)
    if yaml_file is None:
        return {}
    data = yaml.safe_load(yaml_file.read_text()) or {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _load_gtfobins_defaults() -> dict[str, Any]:
    """Load GTFOBins defaults from YAML config (cached).

    This is a fallback for standalone CLI usage. When running under Hydra,
    config should be passed explicitly via GTFOBinsConfig.
    """
    if GTFOBINS_DEFAULTS.exists():
        return yaml.safe_load(GTFOBINS_DEFAULTS.read_text()) or {}
    return {}


def get_generator_default(key: str, default: Any = None) -> Any:
    """Get a generator configuration value by key (fallback for CLI usage)."""
    return _load_generator_defaults().get(key, default)


def load_generator_profile_config(name: str) -> dict[str, Any]:
    """Load a composed generator profile by name."""
    return _load_generator_config_tree(name, set())


def get_generator_file_default(name: str, key: str, default: Any = None) -> Any:
    """Get a value from one generator YAML without composing it by default."""
    return _load_generator_config(name).get(key, default)


def get_gtfobins_default(key: str, default: Any = None) -> Any:
    """Get a GTFOBins configuration value by key (fallback for CLI usage)."""
    return _load_gtfobins_defaults().get(key, default)


def dind_scenario_fields(sudo: bool = False) -> dict[str, Any]:
    """GeneratedScenario fields that enable Docker-in-Docker.

    Kept in sync with conf/scenarios/06_vuln_docker.yaml.

    Args:
        sudo: If True, the ready check uses ``sudo /usr/bin/docker``
              (for scenarios where the user has sudo access but no
              docker group membership).
    """
    docker_cmd = "sudo /usr/bin/docker" if sudo else "docker"
    return {
        "command": (
            "nohup dockerd --registry-mirror=http://privesc-registry-mirror:5000 "
            "--insecure-registry=privesc-registry-mirror:5000 "
            "> /var/log/dockerd.log 2>&1 &\n"
            "exec /usr/sbin/sshd -D -e"
        ),
        "ready_command": (
            f"until {docker_cmd} info >/dev/null 2>&1; do\n"
            f"  sleep 0.5\n"
            f"done\n"
            f"{docker_cmd} pull alpine >/dev/null\n"
            f"{docker_cmd} pull busybox >/dev/null"
        ),
        "ready_command_timeout": 180,
        "extra_args": ["--privileged"],
        "manual_sshd_start": True,
        "use_registry_mirror": True,
        "log_message": "Docker-in-Docker scenario: using isolated Docker daemon with cached images",
    }


@dataclass(frozen=True)
class GeneratedScenario:
    """Generated procedural scenario data."""

    category: str
    seed: int
    container_user: str
    container_password: str
    setup_script: str
    solution: dict[str, Any]
    hint: str
    metadata: dict[str, Any]
    command: str | None = field(default=None)
    ready_command: str | None = field(default=None)
    ready_command_timeout: int | None = field(default=None)
    log_message: str | None = field(default=None)
    extra_args: list[str] | None = field(default=None)
    pre_command: str | None = field(default=None)
    manual_sshd_start: bool | None = field(default=None)
    use_registry_mirror: bool | None = field(default=None)

    def __post_init__(self) -> None:
        """Auto-inject hint and scenario name into solution dict."""
        if self.hint and "hint" not in self.solution:
            self.solution["hint"] = self.hint
        if "scenario" not in self.solution:
            self.solution["scenario"] = self.category

    def to_scenario_config(self) -> ScenarioConfig:
        """Build a ScenarioConfig from this generated scenario."""
        config = ScenarioConfig(
            name=self.category,
            image=PROCEDURAL_IMAGE,
            container_user=self.container_user,
            container_password=self.container_password,
            setup_script=self.setup_script,
            command=self.command,
            ready_command=self.ready_command,
            ready_command_timeout=self.ready_command_timeout,
            log_message=self.log_message,
            extra_args=self.extra_args or [],
            pre_command=self.pre_command,
            manual_sshd_start=bool(self.manual_sshd_start),
            use_registry_mirror=bool(self.use_registry_mirror),
        )
        backend_override = os.getenv("PRIVESC_SCENARIO_BACKEND", "").strip()
        if backend_override:
            config.backend = backend_override
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rand_hex(rng: random.Random, length: int) -> str:
    """Generate a random hex string of specified length."""
    return "".join(rng.choice("0123456789abcdef") for _ in range(length))


def random_password(rng: random.Random, length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


@dataclass(frozen=True)
class LowPrivUserPool:
    first_names: tuple[str, ...] = ()
    last_names: tuple[str, ...] = ()
    username_templates: tuple[str, ...] = ()


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _lowpriv_config_value(
    profile_config: Mapping[str, object] | None, key: str, default: Any
) -> Any:
    if profile_config is not None and key in profile_config:
        return profile_config[key]
    return get_generator_default(key, default)


def lowpriv_user_pool_from_config(
    profile_config: Mapping[str, object] | None = None,
) -> LowPrivUserPool:
    return LowPrivUserPool(
        first_names=_string_tuple(
            _lowpriv_config_value(profile_config, "lowpriv_first_names", [])
        ),
        last_names=_string_tuple(
            _lowpriv_config_value(profile_config, "lowpriv_last_names", [])
        ),
        username_templates=_string_tuple(
            _lowpriv_config_value(profile_config, "lowpriv_username_templates", [])
        ),
    )


def random_username(
    rng: random.Random, lowpriv_user_pool: LowPrivUserPool | None = None
) -> str:
    pool = lowpriv_user_pool or lowpriv_user_pool_from_config()
    if pool.first_names and pool.last_names:
        first = rng.choice(pool.first_names)
        last = rng.choice(pool.last_names)
        template = (
            rng.choice(pool.username_templates)
            if pool.username_templates
            else "{f}{last}"
        )
        parts = {
            "first": first,
            "last": last,
            "f": first[0],
            "l": last[0],
        }
        raw = template.format(**parts)
        return raw.replace(" ", "").lower()
    return "user" + "".join(rng.choice(string.ascii_lowercase) for _ in range(8))


@dataclass(frozen=True)
class SeedContext:
    rng: random.Random
    user: str
    user_password: str


def seed_context(
    seed: int, lowpriv_user_pool: LowPrivUserPool | None = None
) -> SeedContext:
    rng = random.Random(seed)
    user = random_username(rng, lowpriv_user_pool)
    user_password = random_password(rng)
    # Preserve pre-proof seed streams.
    rand_hex(rng, 16)
    return SeedContext(
        rng=rng,
        user=user,
        user_password=user_password,
    )


def load_gtfobins_allowlist(key: str) -> list[str]:
    """Load GTFOBins allowlist from YAML (fallback for CLI usage).

    When running under Hydra, use cfg.generators.<context>_allowlist instead.

    NOTE: Allowlists are now in conf/generators/ (not conf/gtfobins/).
    """
    allowlist = get_generator_default(key, [])
    return [str(item) for item in allowlist] if isinstance(allowlist, list) else []


class GeneratorProtocol(Protocol):
    """Protocol for scenario generators (for CLI helper)."""

    def generate(self, seed: int) -> GeneratedScenario: ...


def run_generator_cli(
    generator_class: type[GeneratorProtocol], description: str
) -> None:
    """Shared CLI entrypoint for simple generators."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, required=True, help="Integer seed.")
    parser.add_argument("--output", default=None, help="Output path for JSON.")
    args = parser.parse_args()

    scenario = generator_class().generate(args.seed)
    _output_scenario(scenario, args.output)


def run_gtfobins_cli(
    generator_class: type,
    default_config_dir: Path,
    description: str,
) -> None:
    """Shared CLI entrypoint for GTFOBins generators."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, required=True, help="Integer seed.")
    parser.add_argument("--config-dir", default=str(default_config_dir))
    parser.add_argument("--binary", default=None, help="Specific binary to use.")
    parser.add_argument("--output", default=None, help="Output path for JSON.")
    args = parser.parse_args()

    generator = generator_class(config_dir=Path(args.config_dir))
    scenario = generator.generate(args.seed, binary_name=args.binary)
    _output_scenario(scenario, args.output)


def _output_scenario(scenario: GeneratedScenario, output_path: str | None) -> None:
    """Write scenario to file or stdout."""
    payload = json.dumps(scenario.to_dict(), indent=2)
    if output_path:
        Path(output_path).write_text(payload)
    else:
        print(payload)


def load_configs(config_dir: Path) -> list[dict[str, Any]]:
    """Load and validate all YAML configs from a directory."""
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    entries: list[dict[str, Any]] = []
    for path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        if not all(k in data for k in ("name", "binary_path")):
            raise ValueError(f"Config missing required keys: {path}")
        if "exploit_cmds" not in data:
            raise ValueError(f"Config missing exploit commands: {path}")
        entries.append(data)
    return entries


class BaseGtfobinsGenerator(ABC):
    """Base class for GTFOBins scenario generators."""

    def __init__(
        self,
        config_dir: Path,
        category: str,
        allowlist: tuple[str, ...] | list[str] | None = None,
        lowpriv_user_pool: LowPrivUserPool | None = None,
    ) -> None:
        self.config_dir = config_dir
        self.category = category
        self._lowpriv_user_pool = lowpriv_user_pool
        # Empty allowlist means no filtering; only non-empty filters
        self.allowlist = set(allowlist) if allowlist else None

    def list_binaries(self) -> list[str]:
        """List available binaries, filtered by allowlist if set."""
        entries = load_configs(self.config_dir)
        names = [str(entry["name"]) for entry in entries]
        if self.allowlist is None:
            return names
        return [name for name in names if name in self.allowlist]

    def _select_binary(
        self, rng: random.Random, binary_name: str | None
    ) -> dict[str, Any]:
        """Select a binary config, either randomly or by name."""
        candidates = load_configs(self.config_dir)
        if self.allowlist is not None:
            candidates = [e for e in candidates if str(e.get("name")) in self.allowlist]
        candidates.sort(key=lambda e: (e.get("name", ""), e.get("binary_path", "")))

        if not candidates:
            raise ValueError("No GTFOBins configs found.")

        if binary_name is None:
            return rng.choice(candidates)

        if self.allowlist is not None and binary_name not in self.allowlist:
            raise ValueError(f"Binary '{binary_name}' is not in the allowlist.")

        chosen = next(
            (e for e in candidates if str(e.get("name")) == binary_name), None
        )
        if chosen is None:
            raise ValueError(f"Binary '{binary_name}' not found in {self.config_dir}")
        return chosen

    @property
    @abstractmethod
    def hint(self) -> str:
        """Hint text for the scenario."""
        ...

    @abstractmethod
    def _build_setup_script(
        self, binary_path: str, root_password: str, user: str
    ) -> str:
        """Build the setup script for the scenario."""
        ...

    @abstractmethod
    def _describe_vulnerability(self, binary_path: str) -> str:
        """Describe the vulnerability for solution output."""
        ...

    def _solution_description(self, binary_name: str) -> str:
        return f"Procedural {self.category.replace('_', ' ').title()} ({binary_name})."

    def _build_recon_steps(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def _select_preferred_exploit_cmd(self, exploit_cmds: list[str]) -> str:
        if not exploit_cmds:
            raise ValueError(f"{self.category} requires at least one exploit command")
        return exploit_cmds[0]

    def generate(
        self,
        seed: int,
        binary_name: str | None = None,
    ) -> GeneratedScenario:
        """Generate a scenario with deterministic RNG."""
        context = (
            seed_context(seed)
            if self._lowpriv_user_pool is None
            else seed_context(seed, self._lowpriv_user_pool)
        )
        rng = context.rng
        chosen = self._select_binary(rng, binary_name)

        binary_name = str(chosen["name"])
        binary_path = str(chosen["binary_path"])
        exploit_cmds = chosen.get("exploit_cmds")
        if isinstance(exploit_cmds, list) and exploit_cmds:
            all_exploit_cmds = [str(cmd) for cmd in exploit_cmds]
        else:
            raise ValueError(f"No exploit_cmds for {binary_name}")

        root_password = rand_hex(rng, 16)
        preferred_exploit_cmd = self._select_preferred_exploit_cmd(all_exploit_cmds)
        metadata = {
            "binary_name": binary_name,
            "binary_path": binary_path,
            "exploit_cmds": all_exploit_cmds,
            "preferred_exploit_cmd": preferred_exploit_cmd,
            "root_password": root_password,
            "user": context.user,
        }
        exploit_tool_calls = [
            *self._build_recon_steps(metadata),
            *[
                {
                    "function": "exec_command",
                    "arguments": {"command": command},
                    "expected_result": {
                        "got_root": True,
                        "exit_code": 124,
                        "output_contains": ["#"],
                    },
                }
                for command in [preferred_exploit_cmd]
            ],
        ]
        return GeneratedScenario(
            category=self.category,
            seed=seed,
            container_user=context.user,
            container_password=context.user_password,
            setup_script=self._build_setup_script(
                binary_path, root_password, context.user
            ),
            solution={
                "description": self._solution_description(binary_name),
                "vulnerability": self._describe_vulnerability(binary_path),
                "exploit_tool_calls": exploit_tool_calls,
            },
            hint=self.hint,
            metadata=metadata,
        )
