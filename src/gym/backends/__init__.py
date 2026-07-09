from __future__ import annotations

from typing import Any, Callable

from src.config import ScenarioConfig, SSHConfig
from src.gym.backends.base import (
    AuthOutcome,
    CommandOutcome,
    CONNECTION_ERROR_EXIT_CODE,
    TIMEOUT_EXIT_CODE,
    ScenarioBackend,
    calculate_command_timeout as _calculate_command_timeout,
)
from src.gym.backends.remote_ssh import RemoteSshDockerBackend

_LocalDockerBackend: Any = None

try:
    from src.gym.backends import local_docker as _local_docker

    _LocalDockerBackend = _local_docker.LocalDockerBackend
except ModuleNotFoundError as err:  # pragma: no cover - optional runtime dependency
    if err.name == "docker":
        _LocalDockerBackend = None
    else:
        raise

LocalDockerBackend = _LocalDockerBackend


def build_backend(
    ssh_cfg: SSHConfig,
    scen_cfg: ScenarioConfig,
    log: Callable[[str], None],
) -> ScenarioBackend:
    backend = (scen_cfg.backend or "remote_ssh").strip().lower()
    if backend == "remote_ssh":
        return RemoteSshDockerBackend(ssh_cfg, scen_cfg, log)
    if backend == "local_docker":
        if _LocalDockerBackend is None:
            raise RuntimeError(
                "local_docker backend requires the 'docker' Python package"
            )
        return _LocalDockerBackend(ssh_cfg, scen_cfg, log)
    raise ValueError(f"Unknown scenario backend: {scen_cfg.backend}")


def calculate_command_timeout(command: str, scen_cfg: ScenarioConfig) -> int:
    return _calculate_command_timeout(
        command=command,
        base_command_timeout=scen_cfg.base_command_timeout,
        slow_command_timeout=scen_cfg.slow_command_timeout,
        max_command_timeout=scen_cfg.max_command_timeout,
    )


__all__ = [
    "AuthOutcome",
    "CommandOutcome",
    "CONNECTION_ERROR_EXIT_CODE",
    "LocalDockerBackend",
    "RemoteSshDockerBackend",
    "ScenarioBackend",
    "TIMEOUT_EXIT_CODE",
    "build_backend",
    "calculate_command_timeout",
]
