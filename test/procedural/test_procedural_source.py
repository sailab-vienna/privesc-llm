"""Tests for ProceduralScenarioSource seeding behavior."""

import random

import pytest

from src.scenarios.procedural import ProceduralScenarioSource


GENERATORS = ["password_file", "suid_gtfobins", "password_reuse"]


def _minimal_password_artifact_config() -> dict:
    return {
        "password_file": {
            "placement_locations": {
                "mail": ["{user_home}"],
                "config": ["{user_home}"],
                "note": ["{user_home}"],
            },
            "file_templates": [
                {"filename": "admin_notes.txt", "content": "Root password: {password}"}
            ],
        },
        "password_history": {
            "history_templates": ["mysql -u root -p'{password}' -e 'SHOW DATABASES'"],
            "history_noise_commands": ["id", "pwd", "whoami"],
            "history_filenames": [".bash_history"],
            "history_noise": {
                "prefix_count_min": 1,
                "prefix_count_max": 1,
                "suffix_start": 1,
                "suffix_count_min": 0,
                "suffix_count_max": 0,
            },
        },
    }


@pytest.fixture
def source():
    return ProceduralScenarioSource(generators=GENERATORS, base_seed=42)


@pytest.fixture
def source_fixed_env():
    return ProceduralScenarioSource(
        generators=GENERATORS, base_seed=42, randomize_env=False
    )


@pytest.fixture
def source_random_seed():
    return ProceduralScenarioSource(
        generators=GENERATORS, base_seed=42, random_seed=True
    )


@pytest.fixture
def source_random_seed_fixed_env():
    return ProceduralScenarioSource(
        generators=GENERATORS,
        base_seed=42,
        random_seed=True,
        randomize_env=False,
    )


class TestRandomizeEnvDefault:
    def test_trace_seed_increments_per_run(self, source):
        """Each run produces a unique trace seed (base_seed + run_index)."""
        seeds = [source.build(i).metadata["seed"] for i in range(6)]
        assert seeds == list(range(42, 48))

    def test_different_runs_same_generator_produce_different_envs(self, source):
        """With randomize_env=True, repeated runs on the same generator differ."""
        # password_file is at generator_idx=0, so run_index 0, 3, 6 all map to it
        instances = [source.build(i) for i in [0, 3, 6]]
        passwords = [inst.metadata["root_password"] for inst in instances]
        assert len(set(passwords)) == 3, "Each run should produce a unique environment"

    def test_no_layout_seed_in_metadata(self, source):
        """layout_seed should not appear in metadata in normal mode."""
        inst = source.build(0)
        assert "layout_seed" not in inst.metadata


class TestRandomizeEnvDisabled:
    def test_trace_seed_still_increments(self, source_fixed_env):
        """trace_seed (dedup key) must still increment even when env is fixed."""
        seeds = [source_fixed_env.build(i).metadata["seed"] for i in range(6)]
        assert seeds == list(range(42, 48))

    def test_same_generator_produces_same_env(self, source_fixed_env):
        """All runs for a given generator must produce identical environments."""
        # password_file is at generator_idx=0, run_indices 0, 3, 6
        instances = [source_fixed_env.build(i) for i in [0, 3, 6]]
        passwords = [inst.metadata["root_password"] for inst in instances]
        assert len(set(passwords)) == 1, "Fixed env: all runs must share same password"

    def test_different_generators_produce_different_envs(self, source_fixed_env):
        """Different generators must still get different environments from each other."""
        # run_index 0 -> password_file (generator_idx=0), layout_seed = 42 + 0
        # run_index 1 -> suid_gtfobins (generator_idx=1), layout_seed = 42 + 1
        inst0 = source_fixed_env.build(0)
        inst1 = source_fixed_env.build(1)
        assert inst0.metadata["generator_name"] != inst1.metadata["generator_name"]
        assert inst0.metadata["seed"] != inst1.metadata["seed"]

    def test_layout_seed_in_metadata(self, source_fixed_env):
        """layout_seed must be recorded in metadata for observability."""
        inst = source_fixed_env.build(3)  # password_file, run 1
        assert "layout_seed" in inst.metadata
        # layout_seed is base_seed + generator_idx (0), not trace_seed (42 + 3 = 45)
        assert inst.metadata["layout_seed"] == 42
        assert inst.metadata["seed"] == 45

    def test_first_run_layout_seed_equals_trace_seed(self, source_fixed_env):
        """For run_index 0 (first occurrence of each generator), seeds must coincide."""
        inst = source_fixed_env.build(0)
        assert inst.metadata["layout_seed"] == inst.metadata["seed"]


class TestRandomSeedEnabled:
    def test_trace_seed_is_deterministic_but_not_sequential(self, source_random_seed):
        seeds = [source_random_seed.build(i).metadata["seed"] for i in range(6)]
        expected = [random.Random(42 + i).randrange(2**31) for i in range(6)]
        assert seeds == expected
        assert seeds != list(range(42, 48))

    def test_same_generator_still_varies_env_when_randomized(self, source_random_seed):
        instances = [source_random_seed.build(i) for i in [0, 3, 6]]
        passwords = [inst.metadata["root_password"] for inst in instances]
        assert len(set(passwords)) == 3

    def test_fixed_env_uses_layout_seed_even_with_random_trace_seed(
        self, source_random_seed_fixed_env
    ):
        instances = [source_random_seed_fixed_env.build(i) for i in [0, 3, 6]]
        seeds = [inst.metadata["seed"] for inst in instances]
        layout_seeds = [inst.metadata["layout_seed"] for inst in instances]
        passwords = [inst.metadata["root_password"] for inst in instances]

        assert len(set(seeds)) == 3
        assert layout_seeds == [42, 42, 42]
        assert len(set(passwords)) == 1


class TestGeneratorConfigOverrides:
    def test_password_history_profile_override_is_used(self):
        source = ProceduralScenarioSource(
            generators=["password_history"],
            base_seed=42,
            generator_configs={
                "password_history": {
                    "history_templates": [
                        "mysql -u root -p'{password}' -e 'SHOW DATABASES'"
                    ],
                    "history_noise_commands": [
                        "id",
                        "pwd",
                        "whoami",
                        "df -h",
                        "ps aux",
                    ],
                    "history_filenames": [".bash_history"],
                    "hint": "Search for readable credential artifacts.",
                    "history_noise": {
                        "prefix_count_min": 2,
                        "prefix_count_max": 2,
                        "suffix_start": 2,
                        "suffix_count_min": 1,
                        "suffix_count_max": 1,
                    },
                },
            },
        )

        instance = source.build(0)

        assert instance.metadata["artifact_mode"] == "history"
        assert (
            instance.metadata["history_template"]
            == "mysql -u root -p'{password}' -e 'SHOW DATABASES'"
        )

    def test_weak_password_profile_override_is_used(self):
        source = ProceduralScenarioSource(
            generators=["weak_password"],
            base_seed=42,
            generator_configs={
                "weak_password": {
                    "patterns": [
                        {
                            "name": "root_common_password",
                            "target_role": "root",
                            "password_source": "common_password",
                        }
                    ],
                    "common_passwords": [f"pw{i}" for i in range(100)],
                    "uid0_users": ["monitoring"],
                    "decoy_users": ["decoy"],
                    "decoy_count_min": 1,
                    "decoy_count_max": 1,
                }
            },
        )

        instance = source.build(0)

        assert instance.metadata["weak_password_pattern"] == "root_common_password"
        assert instance.metadata["intended_target_role"] == "root"

    def test_password_file_and_history_aliases_preserve_standard_identities(self):
        source = ProceduralScenarioSource(
            generators=["password_file", "password_history"],
            base_seed=42,
            generator_configs=_minimal_password_artifact_config(),
        )

        password_file = source.build(0)
        password_history = source.build(1)

        assert password_file.id == "password_file"
        assert password_file.metadata["generator_name"] == "password_file"
        assert password_file.metadata["category"] == "password_file"
        assert password_file.metadata["artifact_mode"] == "file"

        assert password_history.id == "password_history"
        assert password_history.metadata["generator_name"] == "password_history"
        assert password_history.metadata["category"] == "password_history"
        assert password_history.metadata["artifact_mode"] == "history"

    def test_training_decoy_switch_is_used_by_generators(self):
        source = ProceduralScenarioSource(
            generators=[
                "suid_gtfobins",
                "sudo_gtfobins",
                "password_reuse",
                "weak_password",
            ],
            base_seed=42,
            generator_configs={
                "decoys_enabled": False,
                "suid_gtfobins": {},
                "suid_allowlist": ["php"],
                "suid_layout": {"decoy_count_min": 1, "decoy_count_max": 1},
                "sudo_gtfobins": {},
                "sudo_allowlist": ["php"],
                "sudo_layout": {"decoy_count_min": 1, "decoy_count_max": 1},
                "password_reuse": {
                    "patterns": [{"name": "root_account_reuse"}],
                    "service_users": ["backup"],
                },
                "weak_password": {
                    "patterns": [
                        {
                            "name": "root_common_password",
                            "target_role": "root",
                            "password_source": "common_password",
                        }
                    ],
                    "common_passwords": [
                        "password",
                        "password123",
                        "123456",
                        "admin",
                        "administrator",
                        "toor",
                        "letmein",
                        "qwerty",
                        "abc123",
                        "changeme",
                        "welcome",
                        "pass123",
                        "secret",
                        "linux",
                        "server",
                    ],
                    "uid0_users": ["monitoring"],
                    "decoy_users": [],
                    "decoy_count_min": 1,
                    "decoy_count_max": 1,
                },
            },
        )

        built_instances = [source.build(i) for i in range(4)]
        instances = {instance.id: instance for instance in built_instances}

        assert instances["suid_gtfobins"].metadata["decoys_enabled"] is False
        assert instances["suid_gtfobins"].metadata["decoy_count"] == 0
        assert instances["sudo_gtfobins"].metadata["decoys_enabled"] is False
        assert instances["sudo_gtfobins"].metadata["decoy_count"] == 0
        assert instances["password_reuse"].metadata["decoys_enabled"] is False
        assert instances["password_reuse"].metadata["decoy_count"] == 0
        assert instances["weak_password"].metadata["common_password_count"] == 15
        assert instances["weak_password"].metadata["decoys_enabled"] is False
        assert instances["weak_password"].metadata["decoy_count"] == 0
