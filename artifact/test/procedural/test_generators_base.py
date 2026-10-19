"""Tests for the shared generator base class and utilities."""

import logging
import os
from pathlib import Path

import pytest

from src.config import SSHConfig
from src.generators import GENERATOR_REGISTRY
from src.generators.base import GeneratedScenario, load_gtfobins_allowlist
from src.generators.suid_gtfobins import SuidGtfobinsGenerator

log = logging.getLogger(__name__)

_BASE_TEST_BINARY_PREFERENCES = ("chroot", "xargs", "rsync", "php", "node")


def _select_base_test_binary() -> str:
    allowlist = set(load_gtfobins_allowlist("suid_allowlist"))
    for binary in _BASE_TEST_BINARY_PREFERENCES:
        if binary in allowlist:
            return binary
    if allowlist:
        return sorted(allowlist)[0]
    raise AssertionError("Need at least one SUID GTFO binary in the allowlist")


BASE_TEST_BINARY = _select_base_test_binary()


def local_docker_ssh_config() -> SSHConfig:
    return SSHConfig(
        user=os.getenv("PRIVESC_USER", "root"),
        key_path=os.getenv("PRIVESC_KEY", ""),
        servers=os.getenv("PRIVESC_SSH_SERVERS", ""),
    )


def local_docker_scenario_config(scenario: GeneratedScenario):
    scenario_config = scenario.to_scenario_config()
    scenario_config.backend = "local_docker"
    return scenario_config


def scenario_config_from_env(scenario: GeneratedScenario):
    backend = os.getenv("PRIVESC_SCENARIO_BACKEND", "remote_ssh").strip().lower()
    if backend == "local_docker":
        return local_docker_scenario_config(scenario)
    return scenario.to_scenario_config()


def ssh_config_from_env() -> SSHConfig:
    """Get backend config from environment variables."""
    backend = os.getenv("PRIVESC_SCENARIO_BACKEND", "remote_ssh").strip().lower()
    if backend == "local_docker":
        return local_docker_ssh_config()

    ssh_config = local_docker_ssh_config()
    if not ssh_config.key_path:
        pytest.skip("Set PRIVESC_KEY to enable procedural smoke test.")
    key_path = os.path.expanduser(ssh_config.key_path)
    if not Path(key_path).exists():
        pytest.skip(f"PRIVESC_KEY does not exist: {key_path}")
    return SSHConfig(
        user=ssh_config.user,
        key_path=key_path,
        servers=ssh_config.servers,
    )


def _sample_generated_scenario() -> GeneratedScenario:
    return GeneratedScenario(
        category="demo",
        seed=0,
        container_user="user",
        container_password="pw",
        setup_script="true",
        solution={
            "description": "demo",
            "vulnerability": "demo",
            "exploit_tool_calls": [
                {
                    "function": "test_credentials",
                    "arguments": {"user": "root", "password": "pw"},
                    "expected_result": {"got_root": True, "success": True},
                }
            ],
        },
        hint="demo",
        metadata={},
    )


def test_generated_scenario_to_scenario_config_honors_backend_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIVESC_SCENARIO_BACKEND", "local_docker")
    scenario = _sample_generated_scenario()

    assert scenario.to_scenario_config().backend == "local_docker"


def test_scenario_config_from_env_prefers_local_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIVESC_SCENARIO_BACKEND", "local_docker")
    scenario = _sample_generated_scenario()

    assert scenario_config_from_env(scenario).backend == "local_docker"


def test_ssh_config_from_env_uses_local_docker_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIVESC_SCENARIO_BACKEND", "local_docker")
    monkeypatch.delenv("PRIVESC_KEY", raising=False)

    cfg = ssh_config_from_env()

    assert cfg.servers == os.getenv("PRIVESC_SSH_SERVERS", "")
    assert cfg.user == os.getenv("PRIVESC_USER", "root")


# =============================================================================
# Base class tests (run once with any concrete generator)
# =============================================================================


class TestBaseGtfobinsGenerator:
    """Test shared generator behavior using SuidGtfobinsGenerator as concrete impl."""

    @pytest.fixture
    def generator_no_filter(self) -> SuidGtfobinsGenerator:
        """Generator with no allowlist filtering (all binaries available)."""
        return SuidGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/suid"),
            allowlist=[],  # Empty list = no filtering
        )

    @pytest.fixture
    def generator(self) -> SuidGtfobinsGenerator:
        """Generator with default allowlist from YAML config."""
        return SuidGtfobinsGenerator(config_dir=Path("conf/gtfobins/catalog/suid"))

    @pytest.fixture
    def generator_with_allowlist(self) -> SuidGtfobinsGenerator:
        return SuidGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/suid"),
            allowlist=["bash", "env", "find"],
        )

    def test_list_binaries_returns_all_without_allowlist(self, generator_no_filter):
        """Without allowlist filtering, should return all available binaries."""
        binaries = generator_no_filter.list_binaries()
        assert len(binaries) > 0
        assert "bash" in binaries
        assert "env" in binaries

    def test_list_binaries_filters_with_allowlist(self, generator_with_allowlist):
        """With allowlist, should only return allowed binaries."""
        binaries = generator_with_allowlist.list_binaries()
        assert set(binaries) == {"bash", "env", "find"}

    def test_generate_deterministic(self, generator):
        """Same seed should produce identical scenarios."""
        s1 = generator.generate(seed=42)
        s2 = generator.generate(seed=42)
        assert s1.to_dict() == s2.to_dict()

    def test_generate_different_seeds(self, generator):
        """Different seeds should produce different scenarios."""
        s1 = generator.generate(seed=1)
        s2 = generator.generate(seed=2)
        assert s1.to_dict() != s2.to_dict()

    def test_generate_specific_binary(self, generator):
        """Should be able to generate for a specific binary."""
        scenario = generator.generate(seed=0, binary_name=BASE_TEST_BINARY)
        assert scenario.metadata["binary_name"] == BASE_TEST_BINARY
        assert any(BASE_TEST_BINARY in cmd for cmd in scenario.metadata["exploit_cmds"])

    def test_generate_respects_allowlist(self, generator_with_allowlist):
        """Generator with allowlist should only use allowed binaries."""
        for seed in range(10):
            scenario = generator_with_allowlist.generate(seed=seed)
            assert scenario.metadata["binary_name"] in {"bash", "env", "find"}

    def test_generate_rejects_binary_not_in_allowlist(self, generator_with_allowlist):
        """Requesting a binary not in allowlist should raise error."""
        with pytest.raises(ValueError, match="not in the allowlist"):
            generator_with_allowlist.generate(seed=0, binary_name="perl")

    def test_generate_rejects_nonexistent_binary(self, generator_no_filter):
        """Requesting a binary outside the catalog should raise error."""
        with pytest.raises(ValueError, match="not found"):
            generator_no_filter.generate(seed=0, binary_name="nonexistent_binary")

    def test_generate_scenario_has_required_fields(self, generator):
        """Generated scenario should have all required fields."""
        scenario = generator.generate(seed=0, binary_name=BASE_TEST_BINARY)

        # Check basic fields
        assert scenario.category == "suid_gtfobins"
        assert scenario.seed == 0
        assert isinstance(scenario.setup_script, str)
        assert isinstance(scenario.solution, dict)
        assert isinstance(scenario.hint, str)
        assert isinstance(scenario.metadata, dict)

        # Check metadata entropy
        meta = scenario.metadata
        assert len(meta["root_password"]) == 16, "Password should be 16 hex chars"
        assert "flag_value" not in meta
        assert "flag_path" not in meta

        # Check solution structure
        assert "exploit_tool_calls" in scenario.solution
        root_steps = [
            step
            for step in scenario.solution["exploit_tool_calls"]
            if step["expected_result"]["got_root"]
        ]
        assert len(root_steps) == 1
        assert scenario.solution["exploit_tool_calls"][-1] == root_steps[0]
        assert not scenario.solution.get("alternative_exploits")


def test_all_generators_expose_complete_trace_solution_blocks() -> None:
    for name, generator_class in GENERATOR_REGISTRY.items():
        scenario = generator_class().generate(seed=0)
        solution = scenario.solution

        assert solution["scenario"] == name
        assert solution["description"]
        assert solution["vulnerability"]
        assert solution["hint"]

        exploit_tool_calls = solution["exploit_tool_calls"]
        assert exploit_tool_calls
        root_steps = [
            step for step in exploit_tool_calls if step["expected_result"]["got_root"]
        ]
        assert len(root_steps) == 1
        assert exploit_tool_calls[-1] == root_steps[0]
        assert not solution.get("alternative_exploits")

        for step in exploit_tool_calls:
            assert step["function"] in {"exec_command", "test_credentials"}
            assert step["arguments"]
            expected_result = step["expected_result"]
            assert "got_root" in expected_result

            if step["function"] == "exec_command":
                assert step["arguments"]["command"]
                assert "exit_code" in expected_result
                assert expected_result["output_contains"]
            else:
                assert step["arguments"]["user"]
                assert step["arguments"]["password"]
                assert "success" in expected_result
