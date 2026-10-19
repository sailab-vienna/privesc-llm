"""Tests for procedural low-priv username generation."""

from pathlib import Path

import yaml

from src.scenarios.procedural import ProceduralScenarioSource


def _load_username_config() -> tuple[list[str], list[str], list[str]]:
    data = yaml.safe_load(
        Path("conf/generators/training/lowpriv_usernames.yaml").read_text()
    )
    first = [str(v).strip().lower() for v in data.get("lowpriv_first_names", [])]
    last = [str(v).strip().lower() for v in data.get("lowpriv_last_names", [])]
    templates = [
        str(v).strip() for v in data.get("lowpriv_username_templates", [])
    ]
    return first, last, templates


def _build_valid_usernames(
    first: list[str], last: list[str], templates: list[str]
) -> set[str]:
    if not first or not last:
        return set()
    valid: set[str] = set()
    for f in first:
        for last_name in last:
            parts = {"first": f, "last": last_name, "f": f[0], "l": last_name[0]}
            for template in templates:
                valid.add(template.format(**parts).replace(" ", "").lower())
    return valid


def test_procedural_usernames_match_templates():
    first, last, templates = _load_username_config()
    valid = _build_valid_usernames(first, last, templates)
    assert valid

    source = ProceduralScenarioSource(
        generators=["password_file"], base_seed=42,
    )

    for run_index in range(8):
        instance = source.build(run_index)
        assert instance.config.container_user in valid
