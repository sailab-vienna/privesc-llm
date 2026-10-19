from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.dataset.privesc.trace_audit import hash_json


PROFILE_SPLIT_SURFACES = (
    ("lowpriv first names", ("lowpriv_first_names",), None),
    ("lowpriv last names", ("lowpriv_last_names",), None),
    ("capabilities GTFOBins", ("capabilities_allowlist",), None),
    ("SUID GTFOBins", ("suid_allowlist",), None),
    ("SUID service names", ("suid_layout", "service_names"), None),
    ("SUID decoy binaries", ("suid_layout", "decoy_binaries"), None),
    ("sudo GTFOBins", ("sudo_allowlist",), None),
    ("sudoers filenames", ("sudo_layout", "sudoers_file_names"), None),
    ("sudo decoy wrappers", ("sudo_layout", "decoy_wrappers"), None),
    ("sudo decoy path templates", ("sudo_layout", "decoy_path_templates"), None),
    ("weak passwords", ("weak_password", "common_passwords"), None),
    ("weak uid0 users", ("weak_password", "uid0_users"), None),
    ("weak decoy users", ("weak_password", "decoy_users"), None),
    ("password-file filenames", ("password_file", "file_templates"), "filename"),
    ("password-file contents", ("password_file", "file_templates"), "content"),
    ("history filenames", ("password_history", "history_filenames"), None),
    ("history templates", ("password_history", "history_templates"), None),
    ("history noise commands", ("password_history", "history_noise_commands"), None),
    ("password reuse service users", ("password_reuse", "service_users"), None),
    ("cron directory names", ("cron_backup_dirs",), None),
    ("cron wildcard job names", ("cron_wildcard", "job_names"), None),
    ("cron wildcard archive names", ("cron_wildcard", "archive_path"), "basename"),
    (
        "cron wildcard exploit script names",
        ("cron_wildcard", "exploit_script_name"),
        None,
    ),
    ("cron writable configured script names", ("cron_script_names",), None),
    ("cron writable job names", ("cron_writable_script", "job_names"), None),
    ("SSH key names", ("ssh_key_configs",), "key_name"),
    ("SSH keygen args", ("ssh_key_configs",), "keygen_args"),
    ("SSH user key dirs", ("user_ssh_dir_suffixes",), None),
    ("SSH shared key dirs", ("shared_ssh_dirs",), None),
)


def generator_profile_surface_sets(profile: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        label: _surface_values(profile, path, field)
        for label, path, field in PROFILE_SPLIT_SURFACES
    }


def generator_profile_diff_rows(
    training: Mapping[str, Any], validation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    train_sets = generator_profile_surface_sets(training)
    val_sets = generator_profile_surface_sets(validation)
    rows: list[dict[str, Any]] = []
    for category in sorted(train_sets | val_sets):
        training_values = sorted(train_sets.get(category, set()))
        validation_values = sorted(val_sets.get(category, set()))
        overlap = sorted(set(training_values) & set(validation_values))
        rows.append(
            {
                "category": category,
                "training_count": len(training_values),
                "validation_count": len(validation_values),
                "overlap_count": len(overlap),
                "status": "disjoint" if not overlap else "overlap",
                "training_values_sha256": hash_json(training_values),
                "validation_values_sha256": hash_json(validation_values),
                "overlap_values": json.dumps(overlap, sort_keys=True),
            }
        )
    return rows


def _surface_values(
    profile: Mapping[str, Any], path: tuple[str, ...], field: str | None
) -> set[str]:
    value = _select(profile, path)
    if field == "basename":
        return {Path(str(value)).name} if value not in (None, "") else set()
    if field:
        return _set(item.get(field) for item in _records(value))
    return _set(_flatten(value))


def _select(profile: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = profile
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _flatten(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _set(values: Iterable[Any]) -> set[str]:
    return {str(value) for value in values if value not in (None, "")}
