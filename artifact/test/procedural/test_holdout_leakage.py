"""Tests to verify no leakage from benchmark holdout set into live training generators."""

from dataclasses import asdict
import json

import pytest
import yaml

from src.generators import GENERATOR_REGISTRY
from src.generators.base import (
    PROJECT_ROOT,
    get_generator_default,
    get_generator_file_default,
)
from src.generators.capabilities_gtfobins import CapabilitiesGtfobinsGenerator
from src.generators.holdout_manifest import (
    GENERATOR_HOLDOUT_SCAN_RULES,
    HOLDOUT_SCAN_CATEGORY_EXEMPTIONS,
    LIVE_GENERATORS,
    REQUIRED_HOLDOUT_CATEGORIES,
    STATIC_BENCHMARK_CASES,
    matching_holdout_rules,
    sample_holdout_seeds,
)
from src.generators.credential_artifact import (
    PasswordFileGenerator,
    PasswordHistoryGenerator,
)
from src.generators.sudo_gtfobins import load_sudo_allowlist

SUID_LAYOUT = get_generator_default("suid_layout", {})
SUID_DECOY_BINARIES = set(SUID_LAYOUT.get("decoy_binaries", []))
FORBIDDEN_SUID_DECOY_NAMES = {
    "basename",
    "dirname",
    "find",
    "groups",
    "hostid",
    "id",
    "printenv",
    "stat",
    "uname",
    "uptime",
    "users",
    "whoami",
}


def _scenario_field_blobs(scenario) -> dict[str, str]:
    return {
        "setup_script": scenario.setup_script,
        "metadata": json.dumps(scenario.metadata, sort_keys=True),
        "solution": json.dumps(scenario.solution, sort_keys=True),
        "config": json.dumps(asdict(scenario.to_scenario_config()), sort_keys=True),
    }


def _matching_rule_patterns(generator_name: str, payload: str) -> list[str]:
    return [rule.pattern for rule in matching_holdout_rules(generator_name, payload)]


def _holdout_blob_hits(generator_name: str, *, field_name: str, blob: str) -> list[str]:
    hits: list[str] = []
    for rule in matching_holdout_rules(generator_name, blob):
        benchmark_cases = ", ".join(rule.benchmark_cases)
        hits.append(
            f"{field_name}: {rule.category} ({benchmark_cases}) -> {rule.pattern}"
        )
    return hits


def _holdout_hits(generator_name: str, scenario) -> list[str]:
    hits: list[str] = []
    for field_name, blob in _scenario_field_blobs(scenario).items():
        hits.extend(
            _holdout_blob_hits(generator_name, field_name=field_name, blob=blob)
        )
    return hits


def _generator_default_config_blob(generator_name: str) -> str:
    config_path = (
        PROJECT_ROOT / "conf" / "generators" / "training" / f"{generator_name}.yaml"
    )
    config = yaml.safe_load(config_path.read_text()) or {}
    return json.dumps(config, sort_keys=True)


def _assert_no_holdout_hits(generator_name: str, scenario, *, label: str) -> None:
    hits = _holdout_hits(generator_name, scenario)
    assert not hits, (
        f"{generator_name} leaked benchmark artifacts for {label}:\n"
        + "\n".join(sorted(set(hits)))
    )


class TestHoldoutLeakage:
    """Verify benchmark values are excluded from live training generators."""

    def test_suid_allowlist_excludes_benchmark_binaries(self):
        allowlist = set(get_generator_default("suid_allowlist", []))

        assert "find" not in allowlist
        assert "python" not in allowlist
        assert "python3" not in allowlist
        assert "python3.11" not in allowlist
        assert "tar" not in allowlist

    def test_sudo_allowlist_excludes_benchmark_binaries(self):
        allowlist = set(load_sudo_allowlist())

        assert "tar" not in allowlist
        assert "docker" not in allowlist
        assert "find" not in allowlist
        assert "python" not in allowlist
        assert "python3" not in allowlist
        assert "python3.11" not in allowlist
        assert "less" not in allowlist
        assert "man" not in allowlist
        assert "bash" not in allowlist
        assert "env" not in allowlist

    def test_sudo_yaml_allowlist_excludes_benchmark_binaries(self):
        allowlist = set(get_generator_default("sudo_allowlist", []))

        for binary in ("tar", "docker", "less", "man"):
            assert binary not in allowlist

    def test_suid_decoys_avoid_gtfobins_and_recon_overlaps(self):
        catalog_names = {
            path.stem
            for path in (PROJECT_ROOT / "conf" / "gtfobins" / "catalog").glob(
                "**/*.yaml"
            )
        }
        upstream_dir = PROJECT_ROOT / "external" / "GTFOBins" / "_gtfobins"
        upstream_names = {
            path.name for path in upstream_dir.iterdir() if path.is_file()
        }

        assert not (SUID_DECOY_BINARIES & catalog_names), (
            "SUID decoys overlap GTFOBins catalogs: "
            f"{sorted(SUID_DECOY_BINARIES & catalog_names)}"
        )
        assert not (SUID_DECOY_BINARIES & upstream_names), (
            "SUID decoys overlap upstream GTFOBins entries: "
            f"{sorted(SUID_DECOY_BINARIES & upstream_names)}"
        )
        assert not (SUID_DECOY_BINARIES & FORBIDDEN_SUID_DECOY_NAMES), (
            "SUID decoys shadow common recon commands: "
            f"{sorted(SUID_DECOY_BINARIES & FORBIDDEN_SUID_DECOY_NAMES)}"
        )


class TestHoldoutCoverageManifest:
    """Verify the benchmark holdout manifest stays aligned with live code."""

    def test_holdout_manifest_covers_live_generator_mix(self):
        assert tuple(GENERATOR_HOLDOUT_SCAN_RULES) == LIVE_GENERATORS

    def test_holdout_manifest_covers_all_static_benchmark_cases(self):
        benchmark_cases = {
            path.stem
            for path in (
                PROJECT_ROOT
                / "external"
                / "benchmark-privesc-linux"
                / "docker"
                / "scenarios"
            ).glob("*.sh")
        }

        assert set(STATIC_BENCHMARK_CASES) == benchmark_cases

    def test_holdout_scan_rules_cover_all_manifest_cases(self):
        covered_cases = {
            benchmark_case
            for rules in GENERATOR_HOLDOUT_SCAN_RULES.values()
            for rule in rules
            for benchmark_case in rule.benchmark_cases
        }

        assert covered_cases == set(STATIC_BENCHMARK_CASES)

    def test_holdout_manifest_covers_all_required_categories(self):
        manifest_categories = {
            category
            for case in STATIC_BENCHMARK_CASES.values()
            for category in case.categories
        }
        rule_categories = {
            rule.category
            for rules in GENERATOR_HOLDOUT_SCAN_RULES.values()
            for rule in rules
        }
        exempt_categories = {
            category
            for category, comment in HOLDOUT_SCAN_CATEGORY_EXEMPTIONS.items()
            if comment.strip()
        }

        assert set(REQUIRED_HOLDOUT_CATEGORIES) <= manifest_categories
        assert manifest_categories <= rule_categories | exempt_categories

    def test_regression_generators_are_explicitly_included(self):
        assert {
            "capabilities_gtfobins",
            "suid_gtfobins",
            "sudo_gtfobins",
            "password_file",
            "password_history",
        } <= set(GENERATOR_HOLDOUT_SCAN_RULES)

        assert {"01_vuln_suid_gtfo", "05_vuln_sudo_gtfo"} <= {
            benchmark_case
            for rule in GENERATOR_HOLDOUT_SCAN_RULES["suid_gtfobins"]
            for benchmark_case in rule.benchmark_cases
        }
        assert {
            "01_vuln_suid_gtfo",
            "03_vuln_sudo_no_password",
            "04_vuln_sudo_gtfo_interactive",
            "05_vuln_sudo_gtfo",
            "06_vuln_docker",
        } <= {
            benchmark_case
            for rule in GENERATOR_HOLDOUT_SCAN_RULES["sudo_gtfobins"]
            for benchmark_case in rule.benchmark_cases
        }


class TestRuleMatching:
    """Sanity-check critical holdout regexes and token rules."""

    def test_test_credentials_root_root_is_rejected(self):
        assert _matching_rule_patterns("weak_password", "test_credentials root:root")

    def test_username_password_json_is_rejected(self):
        payload = '{"username": "root", "password": "trustno1"}'

        assert _matching_rule_patterns("weak_password", payload)

    def test_docker_tokens_are_rejected(self):
        assert _matching_rule_patterns("sudo_gtfobins", '"docker"')
        assert _matching_rule_patterns(
            "sudo_gtfobins",
            "docker run --rm -it -v /:/mnt alpine chroot /mnt /bin/sh",
        )

    def test_docker_registry_image_is_allowed(self):
        assert not _matching_rule_patterns(
            "sudo_gtfobins", "docker.io/library/alpine:latest"
        )


class TestGeneratedHoldoutScanning:
    """Scan live generator outputs for benchmark artifacts over multiple seeds."""

    def test_randomized_seed_sampling_uses_multiple_seeds(self):
        for generator_name in LIVE_GENERATORS:
            seeds = sample_holdout_seeds(generator_name)

            assert len(seeds) > 2, generator_name
            assert len(seeds) == len(set(seeds)), generator_name
            assert 0 in seeds, generator_name
            assert 42 in seeds, generator_name

    @pytest.mark.parametrize("generator_name", LIVE_GENERATORS)
    def test_default_generator_config_excludes_benchmark_artifacts(
        self, generator_name
    ):
        hits = _holdout_blob_hits(
            generator_name,
            field_name="default_config",
            blob=_generator_default_config_blob(generator_name),
        )
        assert not hits, (
            f"{generator_name} default config leaked benchmark artifacts:\n"
            + "\n".join(sorted(set(hits)))
        )

    @pytest.mark.parametrize("generator_name", LIVE_GENERATORS)
    def test_generated_fields_exclude_benchmark_artifacts(self, generator_name):
        generator = GENERATOR_REGISTRY[generator_name]()

        for seed in sample_holdout_seeds(generator_name):
            scenario = generator.generate(seed)
            _assert_no_holdout_hits(
                generator_name,
                scenario,
                label=f"seed {seed}",
            )


class TestEnumeratedGeneratorBranches:
    """Exhaustively scan configured generator branches that seed sampling can miss."""

    @pytest.mark.parametrize("profile", ("training", "holdout"))
    def test_capabilities_profile_binaries_exclude_benchmark_artifacts(
        self, profile
    ):
        config_path = (
            PROJECT_ROOT
            / "conf"
            / "generators"
            / profile
            / "capabilities_gtfobins.yaml"
        )
        config = yaml.safe_load(config_path.read_text()) or {}
        generator = CapabilitiesGtfobinsGenerator(
            allowlist=config["capabilities_allowlist"]
        )

        for binary_name in generator.list_binaries():
            scenario = generator.generate(0, binary_name=binary_name)
            _assert_no_holdout_hits(
                "capabilities_gtfobins",
                scenario,
                label=f"{profile} binary {binary_name}",
            )

    @pytest.mark.parametrize(
        "generator_name",
        ("suid_gtfobins", "sudo_gtfobins"),
    )
    def test_gtfobins_allowlisted_binaries_exclude_benchmark_artifacts(
        self, generator_name
    ):
        generator = GENERATOR_REGISTRY[generator_name]()

        for binary_name in generator.list_binaries():
            scenario = generator.generate(0, binary_name=binary_name)
            _assert_no_holdout_hits(
                generator_name,
                scenario,
                label=f"binary {binary_name}",
            )

    def test_password_file_templates_exclude_benchmark_artifacts(self):
        config = get_generator_file_default("password_file", "password_file", {})
        for template in config["file_templates"]:
            generator = PasswordFileGenerator(
                config={"file_templates": [template]},
            )
            scenario = generator.generate(0)
            _assert_no_holdout_hits(
                "password_file",
                scenario,
                label=f"file template {template['filename']}",
            )

    def test_password_history_templates_exclude_benchmark_artifacts(self):
        config = get_generator_file_default("password_history", "password_history", {})
        for template in config["history_templates"]:
            generator = PasswordHistoryGenerator(
                config={"history_templates": [template]},
            )
            scenario = generator.generate(0)
            _assert_no_holdout_hits(
                "password_history",
                scenario,
                label=f"history template {template}",
            )

    def test_password_reuse_patterns_exclude_benchmark_artifacts(self):
        config = get_generator_default("password_reuse", {})

        for pattern in config["patterns"]:
            generator = GENERATOR_REGISTRY["password_reuse"](
                config={**config, "patterns": [pattern]}
            )
            scenario = generator.generate(0)
            _assert_no_holdout_hits(
                "password_reuse",
                scenario,
                label=f"pattern {pattern['name']}",
            )

    def test_weak_password_patterns_exclude_benchmark_artifacts(self):
        config = get_generator_default("weak_password", {})

        for pattern in config["patterns"]:
            generator = GENERATOR_REGISTRY["weak_password"](
                config={**config, "patterns": [pattern]}
            )
            scenario = generator.generate(0)
            _assert_no_holdout_hits(
                "weak_password",
                scenario,
                label=f"pattern {pattern['name']}",
            )
