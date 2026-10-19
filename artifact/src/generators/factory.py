"""Factory for procedural generator construction from resolved config."""

from __future__ import annotations

from typing import Any

from src.generators import GENERATOR_REGISTRY
from src.generators.base import get_generator_default, lowpriv_user_pool_from_config


def _mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _list(value: Any) -> list[Any] | None:
    return value if isinstance(value, list) else None


def _config_with_decoy_switch(
    value: Any, profile_config: dict[str, Any], default_key: str | None = None
) -> dict[str, Any] | None:
    mapping = _mapping(value)
    if mapping is None:
        if not profile_config.get("decoys_enabled", False):
            return None
        mapping = (
            dict(get_generator_default(default_key, {}))
            if default_key is not None
            else {}
        )
    if "decoys_enabled" in mapping:
        return mapping
    return {
        **mapping,
        "decoys_enabled": bool(profile_config.get("decoys_enabled", False)),
    }


def build_generator(generator_name: str, profile_config: dict[str, Any]):
    generator_class = GENERATOR_REGISTRY[generator_name]
    lowpriv_user_pool = lowpriv_user_pool_from_config(profile_config)

    if generator_name == "capabilities_gtfobins":
        allowlist = _list(profile_config.get("capabilities_allowlist"))
        return generator_class(
            allowlist=list(allowlist) if allowlist else None,
            lowpriv_user_pool=lowpriv_user_pool,
        )

    if generator_name == "suid_gtfobins":
        allowlist = _list(profile_config.get("suid_allowlist"))
        return generator_class(
            allowlist=list(allowlist) if allowlist else None,
            config=_config_with_decoy_switch(
                profile_config.get("suid_gtfobins"), profile_config
            ),
            layout=_mapping(profile_config.get("suid_layout")),
            lowpriv_user_pool=lowpriv_user_pool,
        )

    if generator_name == "sudo_gtfobins":
        allowlist = _list(profile_config.get("sudo_allowlist"))
        return generator_class(
            allowlist=list(allowlist) if allowlist else None,
            config=_config_with_decoy_switch(
                profile_config.get("sudo_gtfobins"), profile_config, "sudo_gtfobins"
            ),
            layout=_mapping(profile_config.get("sudo_layout")),
            lowpriv_user_pool=lowpriv_user_pool,
        )

    if generator_name == "cron_wildcard":
        return generator_class(
            backup_dirs=_list(profile_config.get("cron_backup_dirs")),
            config=_mapping(profile_config.get("cron_wildcard")),
            lowpriv_user_pool=lowpriv_user_pool,
        )

    if generator_name == "cron_writable_script":
        return generator_class(
            script_names=_list(profile_config.get("cron_script_names")),
            config=_mapping(profile_config.get("cron_writable_script")),
            lowpriv_user_pool=lowpriv_user_pool,
        )

    if generator_name in {"password_file", "password_history"}:
        return generator_class(
            config=_mapping(profile_config.get(generator_name)),
            lowpriv_user_pool=lowpriv_user_pool,
        )

    if generator_name == "password_reuse":
        reuse_config = profile_config.get("password_reuse")
        return generator_class(
            config=_config_with_decoy_switch(
                reuse_config, profile_config, "password_reuse"
            ),
            lowpriv_user_pool=lowpriv_user_pool,
        )

    if generator_name == "ssh_key_reuse":
        ssh_key_reuse = profile_config.get("ssh_key_reuse")
        ssh_target = None
        if isinstance(ssh_key_reuse, dict):
            ssh_target = ssh_key_reuse.get("ssh_target")
        return generator_class(
            ssh_key_configs=_list(profile_config.get("ssh_key_configs")),
            user_ssh_dir_suffixes=_list(profile_config.get("user_ssh_dir_suffixes")),
            shared_ssh_dirs=_list(profile_config.get("shared_ssh_dirs")),
            ssh_target=str(ssh_target) if isinstance(ssh_target, str) else None,
            config=_mapping(profile_config.get("ssh_key_reuse")),
            lowpriv_user_pool=lowpriv_user_pool,
        )

    if generator_name == "weak_password":
        weak_password = profile_config.get("weak_password")
        return generator_class(
            config=_config_with_decoy_switch(
                weak_password, profile_config, "weak_password"
            ),
            lowpriv_user_pool=lowpriv_user_pool,
        )

    return generator_class()
