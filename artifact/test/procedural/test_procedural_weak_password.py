"""Tests for procedural Weak Password generator."""

import logging
import shlex

import pytest

from src.config import _DEFAULT_RL_GENERATORS
from src.generators.weak_password import WeakPasswordGenerator
from src.gym.scenario import PrivEscScenario

from test.solution_utils import run_exploit
from .test_generators_base import scenario_config_from_env, ssh_config_from_env

log = logging.getLogger(__name__)
logging.getLogger("asyncssh").setLevel(logging.WARNING)


async def _assert_solution_succeeds(scenario) -> None:
    scenario_config = scenario_config_from_env(scenario)
    async with PrivEscScenario(ssh_config_from_env(), scenario_config) as sc:
        for exploit in scenario.solution["exploit_tool_calls"]:
            await run_exploit(sc, exploit, scenario.category, is_alternative=False)


def _scheduled_weak_password_seeds(base_seed: int, runs_per_item: int) -> list[int]:
    stride = len(_DEFAULT_RL_GENERATORS)
    generator_idx = _DEFAULT_RL_GENERATORS.index("weak_password")
    return [
        base_seed + generator_idx + (stride * run_ordinal)
        for run_ordinal in range(runs_per_item)
    ]


def _single_pattern_config(
    generator: WeakPasswordGenerator,
    *,
    name: str,
    target_role: str,
    password_source: str,
    uid0_user: str = "monitoring",
) -> dict[str, object]:
    return {
        "patterns": [
            {
                "name": name,
                "target_role": target_role,
                "password_source": password_source,
            }
        ],
        "common_passwords": list(generator.common_passwords),
        "uid0_users": [uid0_user],
        "decoy_users": list(generator.decoy_users) or ["runner"],
        "decoy_count_min": generator.decoy_count_min,
        "decoy_count_max": generator.decoy_count_max,
    }


@pytest.fixture
def generator() -> WeakPasswordGenerator:
    return WeakPasswordGenerator()


def test_category(generator: WeakPasswordGenerator) -> None:
    scenario = generator.generate(seed=0)
    assert scenario.category == "weak_password"


def test_deterministic(generator: WeakPasswordGenerator) -> None:
    s1 = generator.generate(seed=42)
    s2 = generator.generate(seed=42)
    assert s1.to_dict() == s2.to_dict()


def test_trace_collection_seed_schedule_covers_all_default_patterns(
    generator: WeakPasswordGenerator,
) -> None:
    expected_patterns = {pattern.name for pattern in generator.patterns}
    training_patterns = {
        generator.generate(seed=seed).metadata["weak_password_pattern"]
        for seed in _scheduled_weak_password_seeds(base_seed=42, runs_per_item=100)
    }
    validation_patterns = {
        generator.generate(seed=seed).metadata["weak_password_pattern"]
        for seed in _scheduled_weak_password_seeds(
            base_seed=10_000_000, runs_per_item=25
        )
    }

    assert training_patterns == expected_patterns
    assert validation_patterns == expected_patterns


def test_default_patterns_cover_root_and_uid0_accounts(
    generator: WeakPasswordGenerator,
) -> None:
    assert any(pattern.target_role == "root" for pattern in generator.patterns)
    assert any(pattern.target_role == "uid0" for pattern in generator.patterns)
    assert {pattern.password_source for pattern in generator.patterns} == {
        "common_password",
        "target_username",
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_solution_uses_simple_auditable_credential_checks(
    generator: WeakPasswordGenerator, seed: int
) -> None:
    scenario = generator.generate(seed=seed)
    metadata = scenario.metadata
    steps = scenario.solution["exploit_tool_calls"]

    if metadata["intended_target_role"] == "root":
        assert len(steps) == 1
        assert steps[0]["function"] == "test_credentials"
        assert steps[0]["arguments"] == {
            "user": "root",
            "password": metadata["credential_candidates"][0]["password"],
        }
        assert steps[0]["expected_result"] == {"got_root": True, "success": True}
    else:
        assert metadata["intended_target_role"] == "uid0"
        assert steps[0]["arguments"]["command"] == metadata["uid0_discovery_cmd"]
        assert steps[1]["function"] == "test_credentials"
        assert steps[1]["arguments"] == {
            "user": metadata["intended_target_user"],
            "password": metadata["credential_candidates"][0]["password"],
        }
        assert steps[1]["expected_result"] == {"got_root": True, "success": True}


def test_root_target_variants_use_single_login_step(
    generator: WeakPasswordGenerator,
) -> None:
    scenarios = [generator.generate(seed=i) for i in range(32)]
    root_scenarios = [
        scenario
        for scenario in scenarios
        if scenario.metadata["intended_target_role"] == "root"
    ]

    assert root_scenarios
    for scenario in root_scenarios:
        steps = scenario.solution["exploit_tool_calls"]
        assert len(steps) == 1
        assert steps[0]["function"] == "test_credentials"
        assert steps[0]["arguments"]["user"] == "root"
        assert steps[0]["expected_result"] == {"got_root": True, "success": True}


def test_decoys_are_disabled_by_default(generator: WeakPasswordGenerator) -> None:
    scenarios = [generator.generate(seed=i) for i in range(32)]

    for scenario in scenarios:
        assert scenario.metadata["decoys_enabled"] is False
        assert scenario.metadata["decoy_credentials"] == []
        assert scenario.metadata["decoy_count"] == 0


def test_decoy_switch_enables_decoys(generator: WeakPasswordGenerator) -> None:
    config = _single_pattern_config(
        generator,
        name="root_common_password",
        target_role="root",
        password_source="common_password",
    )
    config["decoys_enabled"] = True
    config["decoy_count_min"] = 1
    config["decoy_count_max"] = 1

    scenario = WeakPasswordGenerator(config=config).generate(seed=0)

    assert scenario.metadata["decoys_enabled"] is True
    assert len(scenario.metadata["decoy_credentials"]) == 1
    assert scenario.metadata["decoy_credentials"][0]["gives_root"] is False


def test_common_password_list_is_available(
    generator: WeakPasswordGenerator,
) -> None:
    assert len(generator.common_passwords) > 0
    scenario = generator.generate(seed=0)
    assert scenario.metadata["common_password_count"] == len(generator.common_passwords)


def test_exact_benchmark_credentials_are_absent(
    generator: WeakPasswordGenerator,
) -> None:
    forbidden_passwords = {"root", "trustno1", "aim8Du7h"}
    forbidden_pairs = {"root:root", "root:trustno1", "root:aim8Du7h"}

    assert not (set(generator.common_passwords) & forbidden_passwords)

    for seed in range(64):
        scenario = generator.generate(seed=seed)
        credentials = {
            f"{entry['user']}:{entry['password']}"
            for entry in scenario.metadata["credential_candidates"]
        }
        assert not (credentials & forbidden_pairs), sorted(
            credentials & forbidden_pairs
        )
        assert "lowpriv" not in scenario.to_dict()["setup_script"]


def test_common_password_pool_rejects_benchmark_passwords(
    generator: WeakPasswordGenerator,
) -> None:
    for password in ["root", "trustno1", "aim8Du7h"]:
        with pytest.raises(ValueError, match="benchmark passwords"):
            WeakPasswordGenerator(
                config={
                    "patterns": [
                        {
                            "name": "root_common_password",
                            "target_role": "root",
                            "password_source": "common_password",
                        }
                    ],
                    "common_passwords": [password, *generator.common_passwords],
                    "decoy_users": list(generator.decoy_users),
                    "decoy_count_min": generator.decoy_count_min,
                    "decoy_count_max": generator.decoy_count_max,
                }
            )


@pytest.mark.parametrize("invalid_password", ["", "bad:pw", "bad\npw"])
def test_rejects_invalid_password_values(
    generator: WeakPasswordGenerator,
    invalid_password: str,
) -> None:
    with pytest.raises(ValueError):
        WeakPasswordGenerator(
            config={
                "patterns": [
                    {
                        "name": "root_common_password",
                        "target_role": "root",
                        "password_source": "common_password",
                    }
                ],
                "common_passwords": [
                    invalid_password,
                    *generator.common_passwords,
                ],
                "decoy_users": list(generator.decoy_users),
                "decoy_count_min": generator.decoy_count_min,
                "decoy_count_max": generator.decoy_count_max,
            }
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("uid0_users", "root"),
        ("decoy_users", "root"),
        ("uid0_users", "ops:prod"),
        ("decoy_users", "scan\nbot"),
        ("decoy_users", ""),
    ],
)
def test_rejects_invalid_user_values(
    generator: WeakPasswordGenerator,
    field_name: str,
    field_value: str,
) -> None:
    config = {
        "patterns": [
            {
                "name": "root_common_password",
                "target_role": "root",
                "password_source": "common_password",
            }
        ],
        "common_passwords": list(generator.common_passwords),
        "uid0_users": list(generator.uid0_users),
        "decoy_users": list(generator.decoy_users),
        "decoy_count_min": generator.decoy_count_min,
        "decoy_count_max": generator.decoy_count_max,
    }
    config[field_name] = [field_value]

    with pytest.raises(ValueError):
        WeakPasswordGenerator(config=config)


def test_root_common_password_sampling_never_generates_root_root(
    generator: WeakPasswordGenerator,
) -> None:
    scenario_generator = WeakPasswordGenerator(
        config=_single_pattern_config(
            generator,
            name="root_common_password",
            target_role="root",
            password_source="common_password",
        )
    )

    for seed in range(64):
        scenario = scenario_generator.generate(seed=seed)
        assert scenario.metadata["credential_candidates"][0]["user"] == "root"
        assert scenario.metadata["credential_candidates"][0]["password"] != "root"


def test_uid0_username_password_creates_named_root_equivalent_account(
    generator: WeakPasswordGenerator,
) -> None:
    scenario = WeakPasswordGenerator(
        config=_single_pattern_config(
            generator,
            name="uid0_username_password",
            target_role="uid0",
            password_source="target_username",
        )
    ).generate(seed=0)

    assert scenario.metadata["intended_target_user"] == "monitoring"
    assert scenario.metadata["intended_target_role"] == "uid0"
    assert scenario.metadata["credential_candidates"][0] == {
        "user": "monitoring",
        "password": "monitoring",
        "role": "uid0",
        "password_source": "target_username",
        "gives_root": True,
    }
    assert "useradd -o -u 0 -g 0 -m -s /bin/bash monitoring" in scenario.setup_script
    assert "printf '%s:%s\\n' monitoring monitoring | chpasswd" in (
        scenario.setup_script
    )
    assert scenario.metadata["uid0_discovery_cmd"] is not None
    assert scenario.solution["exploit_tool_calls"][-1]["arguments"] == {
        "user": "monitoring",
        "password": "monitoring",
    }
    assert "root:root" not in {
        f"{entry['user']}:{entry['password']}"
        for entry in scenario.metadata["credential_candidates"]
    }


def test_root_target_username_password_pattern_is_rejected(
    generator: WeakPasswordGenerator,
) -> None:
    with pytest.raises(ValueError, match="target_username for root targets"):
        WeakPasswordGenerator(
            config=_single_pattern_config(
                generator,
                name="root_username_password",
                target_role="root",
                password_source="target_username",
            )
        )


def test_setup_script_uses_printf_for_chpasswd(
    generator: WeakPasswordGenerator,
) -> None:
    scenario = generator.generate(seed=0)

    assert "printf '%s:%s\\n'" in scenario.setup_script
    assert 'echo "root:' not in scenario.setup_script


def test_setup_script_quotes_special_password() -> None:
    special_password = 'pa$$"word!'
    generator = WeakPasswordGenerator(
        config={
            "patterns": [
                {
                    "name": "root_common_password",
                    "target_role": "root",
                    "password_source": "common_password",
                }
            ],
            "common_passwords": [special_password],
            "decoy_users": ["runner"],
            "decoy_count_min": 1,
            "decoy_count_max": 1,
        }
    )
    scenario = generator.generate(seed=0)

    expected = (
        f"printf '%s:%s\\n' {shlex.quote('root')} "
        f"{shlex.quote(special_password)} | chpasswd"
    )
    assert scenario.metadata["credential_candidates"][0]["password"] == special_password
    assert expected in scenario.setup_script


def test_solution_guidance_avoids_flag_reads(generator: WeakPasswordGenerator) -> None:
    scenario = generator.generate(seed=0)
    payload = " ".join(
        exploit["arguments"].get("command", "")
        for exploit in scenario.solution["exploit_tool_calls"]
    )
    assert "/root/flag.txt" not in payload
    assert "root{" not in payload


@pytest.mark.asyncio
@pytest.mark.slow
async def test_weak_password_default_config_pools_are_solvable() -> None:
    generator = WeakPasswordGenerator()

    for pattern in generator.patterns:
        scenario = WeakPasswordGenerator(
            config=_single_pattern_config(
                generator,
                name=pattern.name,
                target_role=pattern.target_role,
                password_source=pattern.password_source,
            )
        ).generate(seed=0)
        assert scenario.metadata["weak_password_pattern"] == pattern.name
        await _assert_solution_succeeds(scenario)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_procedural_weak_password_smoke() -> None:
    """Test Weak Password: intended weak credential path reaches root."""
    generator = WeakPasswordGenerator()

    for seed in range(20):
        scenario = generator.generate(seed=seed)
        await _assert_solution_succeeds(scenario)
        log.info(
            "[seed=%s pattern=%s] Weak password solution succeeded",
            seed,
            scenario.metadata["weak_password_pattern"],
        )
