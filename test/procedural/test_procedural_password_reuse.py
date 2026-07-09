"""Tests for procedural Password Reuse generator."""

import logging
import random
import shlex

import pytest

from src.config import _DEFAULT_RL_GENERATORS
from src.generators import password_reuse as password_reuse_module
from src.generators.base import SeedContext
from src.generators.password_reuse import PasswordReuseGenerator
from src.gym.scenario import PrivEscScenario

from test.solution_utils import run_exploit
from .test_generators_base import scenario_config_from_env, ssh_config_from_env

log = logging.getLogger(__name__)
logging.getLogger("asyncssh").setLevel(logging.WARNING)


def _single_pattern_config(
    *,
    name: str = "root_account_reuse",
    service_user: str = "backup",
    add_service_distractor: bool = False,
) -> dict[str, object]:
    return {
        "patterns": [
            {
                "name": name,
                "add_service_distractor": add_service_distractor,
            }
        ],
        "service_users": [service_user],
        "decoys_enabled": add_service_distractor,
    }


async def _assert_solution_succeeds(scenario) -> None:
    scenario_config = scenario_config_from_env(scenario)
    async with PrivEscScenario(ssh_config_from_env(), scenario_config) as sc:
        for exploit in scenario.solution["exploit_tool_calls"]:
            await run_exploit(sc, exploit, scenario.category, is_alternative=False)


def _scheduled_password_reuse_seeds(base_seed: int, runs_per_item: int) -> list[int]:
    stride = len(_DEFAULT_RL_GENERATORS)
    generator_idx = _DEFAULT_RL_GENERATORS.index("password_reuse")
    return [
        base_seed + generator_idx + (stride * run_ordinal)
        for run_ordinal in range(runs_per_item)
    ]


@pytest.fixture
def generator() -> PasswordReuseGenerator:
    return PasswordReuseGenerator()


def test_category(generator: PasswordReuseGenerator) -> None:
    scenario = generator.generate(seed=0)
    assert scenario.category == "password_reuse"


def test_deterministic(generator: PasswordReuseGenerator) -> None:
    s1 = generator.generate(seed=42)
    s2 = generator.generate(seed=42)
    assert s1.to_dict() == s2.to_dict()


def test_trace_collection_seed_schedule_covers_all_default_patterns(
    generator: PasswordReuseGenerator,
) -> None:
    expected_patterns = {pattern.name for pattern in generator.patterns}
    training_patterns = {
        generator.generate(seed=seed).metadata["reuse_pattern"]
        for seed in _scheduled_password_reuse_seeds(base_seed=42, runs_per_item=100)
    }
    validation_patterns = {
        generator.generate(seed=seed).metadata["reuse_pattern"]
        for seed in _scheduled_password_reuse_seeds(
            base_seed=10_000_000, runs_per_item=25
        )
    }

    assert training_patterns == expected_patterns
    assert validation_patterns == expected_patterns


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_solution_is_single_root_login_step(
    generator: PasswordReuseGenerator, seed: int
) -> None:
    scenario = generator.generate(seed=seed)
    metadata = scenario.metadata
    steps = scenario.solution["exploit_tool_calls"]

    assert metadata["intended_reuse_user"] == "root"
    assert metadata["intended_reuse_role"] == "root"
    assert metadata["root_reuses_user_password"] is True
    assert metadata["reuse_candidates"] == [
        {
            "user": "root",
            "role": "root",
            "reuses_current_user_password": True,
            "gives_root": True,
        }
    ]
    assert len(steps) == 1
    assert steps[0]["function"] == "test_credentials"
    assert steps[0]["arguments"] == {
        "user": "root",
        "password": scenario.container_password,
    }
    assert steps[0]["expected_result"] == {"got_root": True, "success": True}
    assert "alternative_exploits" not in scenario.solution


def test_decoys_are_disabled_by_default(generator: PasswordReuseGenerator) -> None:
    scenarios = [generator.generate(seed=i) for i in range(32)]

    assert all(s.metadata["intended_reuse_user"] == "root" for s in scenarios)
    assert all(s.metadata["root_reuses_user_password"] for s in scenarios)
    assert all(not s.metadata["distractors"] for s in scenarios)
    assert all(s.metadata["decoys_enabled"] is False for s in scenarios)
    assert all(s.metadata["decoy_count"] == 0 for s in scenarios)


def test_decoy_switch_adds_service_distractor() -> None:
    scenario = PasswordReuseGenerator(
        config=_single_pattern_config(add_service_distractor=True)
    ).generate(seed=0)

    assert scenario.metadata["decoys_enabled"] is True
    assert scenario.metadata["decoy_count"] == 1
    assert scenario.metadata["distractors"] == [
        {
            "type": "service_account",
            "user": "backup",
            "role": "service",
            "reuses_current_user_password": True,
        }
    ]
    assert "useradd -M -r -s /usr/sbin/nologin backup" in scenario.setup_script
    assert scenario.solution["exploit_tool_calls"][0]["function"] == "test_credentials"


def test_decoy_switch_controls_service_distractor() -> None:
    config = _single_pattern_config(add_service_distractor=True)
    config["decoys_enabled"] = False

    scenario = PasswordReuseGenerator(config=config).generate(seed=0)

    assert scenario.metadata["decoys_enabled"] is False
    assert scenario.metadata["distractors"] == []
    assert "useradd -M -r -s /usr/sbin/nologin" not in scenario.setup_script


@pytest.mark.parametrize("invalid_user", ["", "ops:prod", "ops\nprod", "root"])
def test_rejects_invalid_service_users(invalid_user: str) -> None:
    config = _single_pattern_config(
        service_user=invalid_user,
        add_service_distractor=True,
    )

    with pytest.raises(ValueError):
        PasswordReuseGenerator(config=config)


@pytest.mark.parametrize(
    "config",
    [
        {
            "patterns": [{"name": "root_account_reuse", "target_role": "root"}],
            "service_users": ["backup"],
        },
        {
            "patterns": [{"name": "root_account_reuse"}],
            "privileged_users": ["ops"],
        },
        {
            "patterns": [{"name": "root_account_reuse"}],
            "stale_secrets": [
                {"filename": ".pgpass", "template": "localhost:*:{password}"}
            ],
        },
    ],
)
def test_rejects_legacy_password_reuse_config_keys(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Unsupported password_reuse"):
        PasswordReuseGenerator(config=config)


def test_setup_script_uses_printf_for_chpasswd(
    generator: PasswordReuseGenerator,
) -> None:
    scenario = generator.generate(seed=0)

    assert "printf '%s:%s\\n'" in scenario.setup_script
    assert 'echo "root:' not in scenario.setup_script


def test_setup_script_quotes_special_reused_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    special_password = "abc def"

    def fixed_seed_context(seed: int) -> SeedContext:
        return SeedContext(
            rng=random.Random(seed),
            user="analyst",
            user_password=special_password,
        )

    monkeypatch.setattr(password_reuse_module, "seed_context", fixed_seed_context)
    scenario = PasswordReuseGenerator(config=_single_pattern_config()).generate(seed=0)

    expected = (
        f"printf '%s:%s\\n' {shlex.quote('root')} "
        f"{shlex.quote(special_password)} | chpasswd"
    )
    assert scenario.container_password == special_password
    assert expected in scenario.setup_script


def test_solution_guidance_avoids_flag_reads(generator: PasswordReuseGenerator) -> None:
    scenario = generator.generate(seed=0)
    payload = " ".join(
        exploit["arguments"].get("command", "")
        for exploit in scenario.solution["exploit_tool_calls"]
    )
    assert "/root/flag.txt" not in payload
    assert "root{" not in payload


@pytest.mark.asyncio
@pytest.mark.slow
async def test_password_reuse_default_config_is_solvable() -> None:
    scenario = PasswordReuseGenerator().generate(seed=0)
    await _assert_solution_succeeds(scenario)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_password_reuse_service_decoy_config_is_solvable() -> None:
    scenario = PasswordReuseGenerator(
        config=_single_pattern_config(add_service_distractor=True)
    ).generate(seed=0)
    await _assert_solution_succeeds(scenario)
