#!/usr/bin/env python3
"""
Extract exploitable binaries from GTFOBins for privilege escalation scenarios.

Parses GTFOBins YAML files and extracts binaries with shell functions that
have suid/sudo/capabilities contexts. Also generates packages.txt for Docker.

Usage:
    python -m src.generators.extract_gtfobins
    python -m src.generators.extract_gtfobins gtfobins.output_dir=conf/gtfobins/catalog
"""

import logging
import re
from pathlib import Path
from typing import Any, cast

import hydra
import yaml
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.config import AppConfig, register_with_hydra

log = logging.getLogger("privesc_gtfobins")


def parse_gtfobins_file(filepath: Path) -> dict[str, Any] | None:
    """Parse GTFOBins YAML frontmatter from a binary definition file."""
    content = filepath.read_text()
    if not content.startswith("---"):
        return None

    lines = content.split("\n")
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() in ("---", "..."):
            try:
                return yaml.safe_load("\n".join(lines[1:i]))
            except yaml.YAMLError as e:
                log.warning("Failed to parse %s: %s", filepath.name, e)
                return None
    return None


# Binaries that aren't in /usr/bin on Debian
BINARY_PATH_OVERRIDES: dict[str, str] = {
    "capsh": "/usr/sbin/capsh",
    "chroot": "/usr/sbin/chroot",
    "logsave": "/usr/sbin/logsave",
    "start-stop-daemon": "/sbin/start-stop-daemon",
}


SKIP_SNIPPETS = (
    "/path/to",
    "attacker.com",
    "ATTACKER",
    "LHOST",
    "RHOST",
)

# These binaries are shells themselves -- the exploit text won't reference /bin/sh
SHELL_BINARIES = frozenset({"bash", "sh", "dash", "zsh", "ksh", "csh", "fish", "ash"})

# These binaries implicitly drop into a shell without referencing /bin/sh in the command
IMPLICIT_SHELL_BINARIES = frozenset(
    {
        "capsh",
        "chroot",
        "run-parts",
        "script",
        "sg",
    }
)

GENERATOR_GTFOBINS_ALLOWLIST_KEYS = (
    "capabilities_allowlist",
    "suid_allowlist",
    "sudo_allowlist",
)


def _references_shell(name: str, code: str) -> bool:
    """Check whether an exploit command will produce a shell."""
    if "/bin/sh" in code or "/bin/bash" in code:
        return True
    if name in SHELL_BINARIES or name in IMPLICIT_SHELL_BINARIES:
        return True
    return False


_SHELL_TOKEN_BOUNDARY = r"[\s|&;(){}<>]"


def _uses_sudo_binary(name: str, binary_path: str, text: str) -> bool:
    markers = (f"sudo {name}", f"sudo {binary_path}")
    return any(marker in text for marker in markers)


def _references_target_binary(name: str, binary_path: str, text: str) -> bool:
    pattern = re.compile(
        rf"(^|{_SHELL_TOKEN_BOUNDARY})"
        rf"({re.escape(name)}|{re.escape(binary_path)})"
        rf"(?=$|{_SHELL_TOKEN_BOUNDARY})"
    )
    return pattern.search(text) is not None


def _rewrite_binary_invocations(command: str, name: str, binary_path: str) -> str:
    pattern = re.compile(
        rf"(^|{_SHELL_TOKEN_BOUNDARY}){re.escape(name)}"
        rf"(?=$|{_SHELL_TOKEN_BOUNDARY})"
    )
    return pattern.sub(rf"\1{binary_path}", command)


def _ensure_sudo_wrap(command: str, name: str, binary_path: str) -> str:
    if _uses_sudo_binary(name, binary_path, command):
        return command

    prefixes = (name, binary_path)
    stripped = command.lstrip()
    if any(
        stripped == prefix or stripped.startswith(f"{prefix} ") for prefix in prefixes
    ):
        return f"sudo {command}"

    segment_pattern = re.compile(
        rf"(^|(?:&&|\|\||;)\s*)"
        rf"({re.escape(binary_path)}|{re.escape(name)})"
        rf"(?=$|{_SHELL_TOKEN_BOUNDARY})"
    )
    return segment_pattern.sub(r"\1sudo \2", command, count=1)


def _normalize_shell_exploit(name: str, code: str, context: str) -> str:
    text = code.strip()
    if not text:
        return text

    if context == "sudo":
        text = " && ".join(line.strip() for line in text.splitlines() if line.strip())
        if name == "zip":
            text = text.replace("/path/to/temp-file", "/tmp/hosts.zip")
        if "localhost:" in text and ">localhost" in text and not text.startswith("cd "):
            text = f"cd /tmp && {text}"

    return text


def is_usable_exploit(
    name: str,
    binary_path: str,
    code: str,
    context: str,
    ctx_data: dict[str, Any] | None = None,
) -> bool:
    text = code.strip()
    if not text:
        return False
    if (
        context == "suid"
        and isinstance(ctx_data, dict)
        and ctx_data.get("shell") is True
    ):
        return False
    if any(snippet in text for snippet in SKIP_SNIPPETS):
        return False
    if context == "sudo" and ("PERL5OPT" in text or "PERL5DB" in text):
        return False
    if context in ("sudo", "suid") and not _references_shell(name, text):
        return False
    if context == "sudo" and not (
        _references_target_binary(name, binary_path, text)
        or _uses_sudo_binary(name, binary_path, text)
    ):
        return False
    return True


def extract_shell_exploit(
    name: str, data: dict[str, Any], context: str
) -> list[dict[str, Any]]:
    """Extract all shell exploit commands for a given context."""
    shell_funcs = data.get("functions", {}).get("shell", [])
    if not shell_funcs:
        return []

    exploits: list[dict[str, Any]] = []

    for entry in shell_funcs:
        contexts = entry.get("contexts", {})
        if context not in contexts:
            continue

        ctx_data = contexts[context]
        if isinstance(ctx_data, dict) and "code" in ctx_data:
            code = ctx_data["code"]
        else:
            code = entry.get("code", "")

        if not code:
            continue

        binary_path = BINARY_PATH_OVERRIDES.get(name, f"/usr/bin/{name}")
        exploit_cmd = _normalize_shell_exploit(name, code, context)

        if not is_usable_exploit(
            name,
            binary_path,
            exploit_cmd,
            context,
            ctx_data if isinstance(ctx_data, dict) else None,
        ):
            continue

        exploit_cmd = _rewrite_binary_invocations(exploit_cmd, name, binary_path)
        if context == "sudo":
            exploit_cmd = _ensure_sudo_wrap(exploit_cmd, name, binary_path)
        payload = {
            "name": name,
            "binary_path": binary_path,
            "exploit_cmd": exploit_cmd,
        }
        if context == "capabilities" and isinstance(ctx_data, dict):
            required = ctx_data.get("list", [])
            if required:
                payload["required_caps"] = required

        exploits.append(payload)

    return exploits


def extract_all_binaries(
    gtfobins_dir: Path,
    contexts: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Extract exploitable binaries from GTFOBins directory.

    Args:
        gtfobins_dir: Path to GTFOBins _gtfobins directory

    Returns:
        Tuple of (context->binaries, skipped_no_shell_count).
    """
    binaries: dict[str, list[dict[str, Any]]] = {ctx: [] for ctx in contexts}
    skipped_no_shell: list[str] = []

    for filepath in sorted(gtfobins_dir.iterdir()):
        if filepath.is_dir() or filepath.name.startswith("."):
            continue

        name = filepath.name

        data = parse_gtfobins_file(filepath)
        if not data or "shell" not in data.get("functions", {}):
            skipped_no_shell.append(name)
            continue

        for context in contexts:
            exploits = extract_shell_exploit(name, data, context)
            if exploits:
                binaries[context].extend(exploits)
                log.debug("Found %s exploits: %s", context, name)

    return binaries, len(skipped_no_shell)


def write_binary_configs(
    output_dir: Path,
    binaries: dict[str, list[dict[str, Any]]],
    contexts: set[str],
) -> None:
    """Write per-binary YAML configs grouped by context."""
    for context in ("suid", "sudo", "capabilities"):
        if context not in contexts:
            continue
        entries = binaries.get(context, [])

        context_dir = output_dir / context
        context_dir.mkdir(parents=True, exist_ok=True)
        for existing in context_dir.glob("*.yaml"):
            existing.unlink()

        grouped: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
        for entry in entries:
            name = str(entry.get("name", ""))
            binary_path = str(entry.get("binary_path", ""))
            required = tuple(entry.get("required_caps", []) or [])
            exploit_cmd = str(entry.get("exploit_cmd", "")).strip()
            key = (name, binary_path, required)
            grouped.setdefault(key, []).append(exploit_cmd)

        for (name, binary_path, required), exploit_cmds in sorted(grouped.items()):
            exploit_cmds = [cmd for cmd in exploit_cmds if cmd]
            if not exploit_cmds:
                continue
            payload = {
                "name": name,
                "binary_path": binary_path,
                "exploit_cmds": exploit_cmds,
            }
            if context == "capabilities" and required:
                payload["required_caps"] = list(required)

            config_path = context_dir / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def generate_packages_file(
    config_dir: Path,
    binaries: dict[str, list[dict[str, Any]]],
    docker_dir: Path,
    allowlist: set[str] | None = None,
) -> None:
    """Generate packages.txt from extracted binaries using package mappings.

    Args:
        config_dir: Path to gtfobins config directory (conf/gtfobins)
        binaries: Extracted binaries by context
        docker_dir: Path to docker/procedural directory
        allowlist: Optional allowlist of binary names to include
    """
    packages_file = config_dir / "packages.yaml"
    if not packages_file.exists():
        log.warning("packages.yaml not found, skipping package generation")
        return

    package_map = yaml.safe_load(packages_file.read_text()) or {}

    # Collect all unique binary names across all contexts
    all_binaries = set()
    for entries in binaries.values():
        for entry in entries:
            all_binaries.add(entry.get("name"))

    # Limit to allowlist if provided
    if allowlist is not None:
        all_binaries = {name for name in all_binaries if name in allowlist}

    # Map to packages
    packages = set()
    missing = []
    for binary in sorted(all_binaries):
        pkg = package_map.get(binary)
        if pkg and pkg != "null":
            packages.add(pkg)
        else:
            missing.append(binary)

    if missing:
        log.warning("No package mapping for: %s", ", ".join(missing[:10]))
        if len(missing) > 10:
            log.warning("  ... and %d more", len(missing) - 10)

    # Write packages.txt
    packages_txt = docker_dir / "packages.txt"
    packages_txt.parent.mkdir(parents=True, exist_ok=True)
    packages_txt.write_text("\n".join(sorted(packages)) + "\n")
    log.info("Generated %s with %d packages", packages_txt, len(packages))


def collect_generator_gtfobins_allowlist(config_dir: Path) -> set[str]:
    """Collect the GTFOBins binary union required by all generator configs."""
    binaries: set[str] = set()
    if not config_dir.exists():
        return binaries

    for yaml_file in sorted(config_dir.rglob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text()) or {}
        if not isinstance(data, dict):
            continue
        for key in GENERATOR_GTFOBINS_ALLOWLIST_KEYS:
            values = data.get(key, [])
            if not isinstance(values, list):
                continue
            binaries.update(str(value) for value in values if str(value).strip())
    return binaries


def run_extraction(cfg: AppConfig) -> dict[str, Any]:
    """Run the GTFOBins extraction pipeline.

    Args:
        cfg: Application configuration

    Returns:
        Extracted binaries data dict.
    """
    orig_cwd = Path(get_original_cwd())
    gtfobins_dir = orig_cwd / cfg.gtfobins.gtfobins_dir
    output_dir = orig_cwd / cfg.gtfobins.output_dir

    if not gtfobins_dir.exists():
        log.error("GTFOBins directory not found: %s", gtfobins_dir)
        log.error("Run: git submodule update --init")
        raise FileNotFoundError(f"GTFOBins directory not found: {gtfobins_dir}")

    log.info("Extracting from: %s", gtfobins_dir)

    allowed_contexts = {"suid", "sudo", "capabilities"}
    raw_contexts = cfg.gtfobins.contexts or ["suid"]
    contexts = {ctx.lower() for ctx in raw_contexts}
    invalid = contexts - allowed_contexts
    if invalid:
        raise ValueError(f"Invalid GTFOBins contexts: {sorted(invalid)}")

    binaries, skipped_no_shell = extract_all_binaries(gtfobins_dir, contexts)

    log.info(
        "Extracted: %d SUID, %d sudo, %d capabilities exploits",
        len(binaries.get("suid", [])),
        len(binaries.get("sudo", [])),
        len(binaries.get("capabilities", [])),
    )
    log.info(
        "Skipped: %d no shell function",
        skipped_no_shell,
    )

    # Write per-binary configs
    output_dir.mkdir(parents=True, exist_ok=True)
    write_binary_configs(output_dir, binaries, contexts)
    log.info("Per-binary configs written to: %s", output_dir)

    # Generate packages.txt for Docker
    docker_dir = orig_cwd / "docker" / "procedural"
    config_dir = orig_cwd / "conf" / "gtfobins"
    generator_config_dir = orig_cwd / "conf" / "generators"
    allowlist = collect_generator_gtfobins_allowlist(generator_config_dir)
    cfg_container = cast(DictConfig, cfg)
    allowlist.update(
        str(value)
        for value in (
            OmegaConf.select(cfg_container, "gtfobins.extra_binaries", default=[]) or []
        )
        if str(value).strip()
    )
    generate_packages_file(config_dir, binaries, docker_dir, allowlist)

    return binaries


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for GTFOBins extraction."""
    log.info("Starting GTFOBins extraction")
    run_extraction(cfg)  # type: ignore[arg-type]
    log.info("Extraction complete")


if __name__ == "__main__":
    register_with_hydra()
    main()
