"""Tests for procedural file capabilities GTFOBins generator."""

from pathlib import Path

import pytest

from src.generators.capabilities_gtfobins import (
    CapabilitiesGtfobinsGenerator,
    _setcap_spec,
)


def test_setcap_spec_from_cap_setuid():
    assert _setcap_spec(["CAP_SETUID"]) == "cap_setuid+ep"


def test_setcap_spec_multiple_caps():
    result = _setcap_spec(["CAP_SETUID", "CAP_NET_RAW"])
    assert result == "cap_setuid,cap_net_raw+ep"


def test_setcap_spec_deduplicates():
    assert _setcap_spec(["CAP_SETUID", "CAP_SETUID"]) == "cap_setuid+ep"


def test_setcap_spec_empty_defaults_to_setuid():
    assert _setcap_spec([]) == "cap_setuid+ep"


@pytest.fixture
def generator() -> CapabilitiesGtfobinsGenerator:
    return CapabilitiesGtfobinsGenerator(
        config_dir=Path("conf/gtfobins/catalog/capabilities")
    )


def test_category_is_capabilities(generator):
    scenario = generator.generate(seed=0)
    assert scenario.category == "capabilities_gtfobins"


def test_setup_script_applies_correct_cap_spec(generator):
    scenario = generator.generate(seed=0, binary_name="perl")
    assert "cap_setuid+ep" in scenario.setup_script
    assert "chmod 4755" not in scenario.setup_script
    assert "chmod 0755" not in scenario.setup_script


def test_setup_script_resolves_symlinks(generator):
    scenario = generator.generate(seed=0)
    assert "readlink -f" in scenario.setup_script


def test_each_binary_gets_its_own_cap_spec(generator):
    for binary in generator.list_binaries():
        scenario = generator.generate(seed=0, binary_name=binary)
        assert "+ep" in scenario.setup_script, (
            f"{binary}: setup script missing setcap spec"
        )


def test_default_allowlist_matches_training_config(generator):
    assert generator.list_binaries() == ["perl", "php", "ruby"]


def test_solution_uses_single_preferred_exploit_step(generator):
    scenario = generator.generate(seed=0, binary_name="perl")
    steps = scenario.solution["exploit_tool_calls"]

    assert len(steps) == 1
    assert steps[0]["function"] == "exec_command"
    assert steps[-1]["expected_result"]["got_root"] is True
    assert scenario.metadata["preferred_exploit_cmd"] == steps[-1]["arguments"][
        "command"
    ]
