"""Tests for the procedural base container image.

Verifies that the privesc_procedural Docker image ships all expected
enumeration tools and exposes a sane PATH to low-privilege SSH sessions.

These are the tools that agents commonly use during privilege escalation
and that were previously missing (causing 'command not found' in traces).
"""

import logging
import os

import pytest

from src.config import ScenarioConfig
from src.generators.base import PROCEDURAL_IMAGE
from src.gym.scenario import PrivEscScenario

from .test_generators_base import ssh_config_from_env

log = logging.getLogger(__name__)
logging.getLogger("asyncssh").setLevel(logging.WARNING)

# Tools that agents rely on during enumeration and exploitation.
# Each entry is (binary_name, reason).
# Sorted alphabetically for easy maintenance.
EXPECTED_BINARIES = [
    # Core utilities (coreutils / findutils / util-linux)
    "cat",
    "chmod",
    "chown",
    "cp",
    "cut",
    "df",
    "du",
    "env",
    "find",
    "grep",
    "head",
    "id",
    "ls",
    "mkdir",
    "mv",
    "ps",
    "pwd",
    "rm",
    "sort",
    "stat",
    "tail",
    "tee",
    "touch",
    "tr",
    "uname",
    "uniq",
    "wc",
    "which",
    "whoami",
    "xargs",
    # File inspection (packages: file, binutils)
    "file",
    "strings",
    # Networking (packages: iproute2, net-tools)
    "ip",
    "ss",
    "ifconfig",
    "netstat",
    # Capabilities (package: libcap2-bin) — must be on PATH via login.defs
    "getcap",
    "getpcaps",
    # Misc enumeration tools
    "awk",
    "base64",
    "curl",
    "hexdump",
    "lsb_release",
    "sed",
    "sudo",
    "ssh",
]


def _bare_scenario_config() -> ScenarioConfig:
    """Minimal config that boots the procedural image with no setup script."""
    config = ScenarioConfig(
        name="container_tooling_test",
        image=PROCEDURAL_IMAGE,
        setup_script=(
            "id -u lowpriv >/dev/null 2>&1 || useradd -m -s /bin/bash lowpriv\n"
            "echo 'lowpriv:trustno1' | chpasswd"
        ),
    )
    backend_override = os.getenv("PRIVESC_SCENARIO_BACKEND", "").strip()
    if backend_override:
        config.backend = backend_override
    return config


@pytest.mark.asyncio
@pytest.mark.slow
async def test_expected_binaries_on_path():
    """Every binary in EXPECTED_BINARIES must be discoverable via 'command -v' as lowpriv."""
    ssh_config = ssh_config_from_env()
    scenario_config = _bare_scenario_config()

    async with PrivEscScenario(ssh_config, scenario_config) as sc:
        # Build a single command that checks all binaries at once.
        # Output format: one line per binary, "BINARY:PATH" or "BINARY:MISSING".
        checks = " && ".join(
            f'printf "{b}:"; command -v {b} || printf "MISSING\\n"'
            for b in EXPECTED_BINARIES
        )
        result = await sc.exec_command(f"{{ {checks}; }} 2>/dev/null")
        assert result.exit_code == 0, f"Check command failed: {result.output}"

        missing = []
        for line in result.output.strip().splitlines():
            if "MISSING" in line:
                binary = line.split(":")[0]
                missing.append(binary)

        assert not missing, (
            f"{len(missing)} binaries not found on lowpriv PATH: {', '.join(missing)}"
        )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_path_includes_sbin():
    """Low-priv SSH session PATH must include /usr/sbin and /sbin."""
    ssh_config = ssh_config_from_env()
    scenario_config = _bare_scenario_config()

    async with PrivEscScenario(ssh_config, scenario_config) as sc:
        result = await sc.exec_command("echo $PATH")
        path = result.output.strip()
        log.info(f"lowpriv PATH: {path}")
        assert "/usr/sbin" in path, f"PATH missing /usr/sbin: {path}"
        assert "/sbin" in path, f"PATH missing /sbin: {path}"
