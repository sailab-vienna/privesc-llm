"""Procedural scenario generation for privilege escalation training."""

from typing import Any, Type

from .base import BaseGtfobinsGenerator, GeneratedScenario, GeneratorProtocol
from .capabilities_gtfobins import CapabilitiesGtfobinsGenerator
from .credential_artifact import (
    CredentialArtifactBase,
    PasswordFileGenerator,
    PasswordHistoryGenerator,
)
from .cron_wildcard import CronWildcardGenerator
from .cron_writable_script import CronWritableScriptGenerator
from .password_reuse import PasswordReuseGenerator
from .ssh_key_reuse import SshKeyReuseGenerator
from .suid_gtfobins import SuidGtfobinsGenerator
from .sudo_gtfobins import SudoGtfobinsGenerator
from .weak_password import WeakPasswordGenerator

# Registry mapping generator names to their classes
# Names use snake_case to match config YAML format
GENERATOR_REGISTRY: dict[str, Type[Any]] = {
    "capabilities_gtfobins": CapabilitiesGtfobinsGenerator,
    "suid_gtfobins": SuidGtfobinsGenerator,
    "sudo_gtfobins": SudoGtfobinsGenerator,
    "cron_wildcard": CronWildcardGenerator,
    "cron_writable_script": CronWritableScriptGenerator,
    "password_file": PasswordFileGenerator,
    "password_history": PasswordHistoryGenerator,
    "password_reuse": PasswordReuseGenerator,
    "ssh_key_reuse": SshKeyReuseGenerator,
    "weak_password": WeakPasswordGenerator,
}

__all__ = [
    "BaseGtfobinsGenerator",
    "CapabilitiesGtfobinsGenerator",
    "CredentialArtifactBase",
    "CronWildcardGenerator",
    "CronWritableScriptGenerator",
    "GeneratedScenario",
    "GeneratorProtocol",
    "GENERATOR_REGISTRY",
    "PasswordFileGenerator",
    "PasswordHistoryGenerator",
    "PasswordReuseGenerator",
    "SshKeyReuseGenerator",
    "SuidGtfobinsGenerator",
    "SudoGtfobinsGenerator",
    "WeakPasswordGenerator",
]
