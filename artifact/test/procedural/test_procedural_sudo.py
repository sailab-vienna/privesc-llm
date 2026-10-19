"""Tests for procedural Sudo GTFOBins generator."""

import logging
from pathlib import Path
from typing import cast

import pytest
from omegaconf import OmegaConf

from src.generators.sudo_gtfobins import SudoGtfobinsGenerator, load_sudo_allowlist
from src.generators.base import get_generator_default
from src.gym.scenario import PrivEscScenario

from test.solution_utils import run_exploit
from .test_generators_base import ssh_config_from_env

log = logging.getLogger(__name__)
logging.getLogger("asyncssh").setLevel(logging.WARNING)

_LIVE_SUDO_SMOKE_BINARIES = ("mawk", "rsync")


def _root_steps(scenario) -> list[dict]:
    return [
        step
        for step in scenario.solution["exploit_tool_calls"]
        if step["expected_result"]["got_root"]
    ]


def _training_sudo_config(**overrides) -> dict[str, object]:
    config = dict(get_generator_default("sudo_gtfobins", {}))
    config.update(overrides)
    return config


def _sudo_generator(
    *,
    allowlist: list[str] | None = None,
    config: dict[str, object] | None = None,
) -> SudoGtfobinsGenerator:
    return SudoGtfobinsGenerator(
        config_dir=Path("conf/gtfobins/catalog/sudo"),
        allowlist=allowlist,
        config=config if config is not None else _training_sudo_config(),
    )


async def _assert_solution_succeeds(scenario) -> None:
    ssh_config = ssh_config_from_env()
    async with PrivEscScenario(ssh_config, scenario.to_scenario_config()) as sc:
        for exploit in scenario.solution["exploit_tool_calls"]:
            await run_exploit(sc, exploit, scenario.category, is_alternative=False)


# =============================================================================
# Sudo-specific tests
# =============================================================================


class TestSudoGtfobinsGenerator:
    """Test Sudo-specific generator behavior."""

    @pytest.fixture
    def generator(self) -> SudoGtfobinsGenerator:
        return _sudo_generator(config=_training_sudo_config(grant_subjects=["user"]))

    def test_category_is_sudo(self, generator):
        scenario = generator.generate(seed=0)
        assert scenario.category == "sudo_gtfobins"

    def test_load_sudo_allowlist_uses_flat_training_config(self, generator):
        expected = get_generator_default("sudo_allowlist", [])
        assert load_sudo_allowlist() == expected

    def test_standalone_constructor_uses_training_config(self):
        generator = SudoGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/sudo"),
            allowlist=["mawk"],
        )
        expected_grant_subjects = set(
            get_generator_default("sudo_gtfobins", {})["grant_subjects"]
        )

        scenarios = [
            generator.generate(seed=seed, binary_name="mawk") for seed in range(20)
        ]
        observed_grant_subjects = {
            scenario.metadata["grant_subject"] for scenario in scenarios
        }

        assert observed_grant_subjects <= expected_grant_subjects
        assert observed_grant_subjects - {"user"}
        assert {scenario.metadata["decoys_enabled"] for scenario in scenarios} == {
            False
        }

    def test_setup_script_configures_sudoers(self, generator):
        scenario = generator.generate(seed=0)
        assert scenario.metadata["sudoers_file"].startswith("/etc/sudoers.d/")
        assert "gtfo" not in scenario.metadata["sudoers_file"].lower()
        assert f"sudoers_file={scenario.metadata['sudoers_file']}" in (
            scenario.setup_script
        )
        assert "chmod 0440" in scenario.setup_script
        assert "NOPASSWD" in scenario.setup_script

    def test_group_grant_subject_configures_membership(self):
        generator = SudoGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/sudo"),
            allowlist=["mawk"],
            config={"decoys_enabled": False, "grant_subjects": ["%sudo"]},
        )
        scenario = generator.generate(seed=0, binary_name="mawk")

        assert scenario.metadata["grant_subject"] == "%sudo"
        assert scenario.metadata["sudoers_subject"] == "%sudo"
        assert scenario.metadata["sudoers_group"] == "sudo"
        assert "groupadd -f \"$grant_group\"" in scenario.setup_script
        assert f"usermod -aG \"$grant_group\" {scenario.container_user}" in (
            scenario.setup_script
        )
        assert "${grant_subject" not in scenario.setup_script
        assert "$sudoers_subject ALL=(root) NOPASSWD: /usr/bin/mawk" in (
            scenario.setup_script
        )
        resolved = cast(
            dict[str, object],
            OmegaConf.to_container(
                OmegaConf.create({"setup_script": scenario.setup_script}),
                resolve=True,
            ),
        )
        assert resolved["setup_script"] == scenario.setup_script

    def test_solution_uses_single_preferred_exploit_step(self, generator):
        scenario = generator.generate(seed=0, binary_name="mawk")
        steps = scenario.solution["exploit_tool_calls"]

        assert len(steps) == 1
        assert steps[0]["function"] == "exec_command"
        assert steps[0]["arguments"]["command"] == scenario.metadata[
            "preferred_exploit_cmd"
        ]
        assert steps[0]["expected_result"]["got_root"] is True

    def test_exploit_cmd_has_sudo_prefix(self, generator):
        """Sudo exploits should have sudo prefix."""
        scenario = generator.generate(seed=0, binary_name="mawk")
        command = _root_steps(scenario)[0]["arguments"]["command"]
        assert command.startswith("sudo ")

    def test_uses_real_gtfo_binary_path_in_sudoers(self, generator):
        scenario = generator.generate(seed=0, binary_name="mawk")
        binary_path = scenario.metadata["binary_path"]

        assert binary_path == "/usr/bin/mawk"
        assert f"NOPASSWD: {binary_path}" in scenario.setup_script
        assert "install_target_wrapper" not in scenario.setup_script

    def test_default_training_has_no_benign_sudo_decoys(self, generator):
        scenario = generator.generate(seed=0, binary_name="mawk")
        assert scenario.metadata["decoys_enabled"] is False
        assert scenario.metadata["decoy_wrapper_paths"] == []
        assert scenario.metadata["sudo_l_paths"] == [scenario.metadata["binary_path"]]

    def test_decoy_enabled_setup_script_adds_benign_sudo_decoys(self):
        generator = _sudo_generator(
            allowlist=["mawk"],
            config=_training_sudo_config(
                decoys_enabled=True,
                grant_subjects=["user"],
            ),
        )
        scenario = generator.generate(seed=0, binary_name="mawk")
        decoy_paths = scenario.metadata["decoy_wrapper_paths"]
        decoy_names = {Path(path).name for path in decoy_paths}
        layout = get_generator_default("sudo_layout", {})

        assert (
            int(layout.get("decoy_count_min", 0))
            <= len(decoy_paths)
            <= int(layout.get("decoy_count_max", 0))
        )
        assert decoy_names <= set(layout.get("decoy_wrappers", []))
        assert all(
            path.startswith("/usr/local/bin/") or path.startswith("/usr/local/sbin/")
            for path in decoy_paths
        )
        assert len(scenario.metadata["sudo_l_paths"]) == 1 + len(decoy_paths)
        assert scenario.metadata["binary_path"] in scenario.metadata["sudo_l_paths"]
        for path in decoy_paths:
            name = Path(path).name
            assert f"{name}: routine maintenance status OK" in scenario.setup_script

    def test_keeps_real_binary_path_in_exploit_commands(self, generator):
        scenario = generator.generate(seed=0, binary_name="rsync")
        binary_path = scenario.metadata["binary_path"]
        commands = [
            tool_call["arguments"]["command"]
            for tool_call in scenario.solution["exploit_tool_calls"]
        ]

        assert binary_path == "/usr/bin/rsync"
        assert all(binary_path in command for command in commands)
        assert all("/usr/local/" not in command for command in commands)
        assert any(command.startswith(f"sudo {binary_path} ") for command in commands)

    def test_decoy_wrapper_paths_vary_across_seeds(self):
        generator = _sudo_generator(
            allowlist=["mawk"],
            config=_training_sudo_config(
                decoys_enabled=True,
                grant_subjects=["user"],
            ),
        )
        observed = set()
        for seed in range(12):
            scenario = generator.generate(seed=seed, binary_name="mawk")
            observed.update(scenario.metadata["decoy_wrapper_paths"])

        assert len(observed) > 1

    def test_sudo_l_order_varies_across_seeds(self):
        generator = _sudo_generator(
            allowlist=["mawk"],
            config=_training_sudo_config(
                decoys_enabled=True,
                grant_subjects=["user"],
            ),
        )
        positions = set()
        for seed in range(12):
            scenario = generator.generate(seed=seed, binary_name="mawk")
            positions.add(
                scenario.metadata["sudo_l_paths"].index(
                    scenario.metadata["binary_path"]
                )
            )

        assert len(positions) > 1

    def test_sudoers_file_name_varies_across_seeds(self):
        generator = _sudo_generator(
            allowlist=["mawk"],
            config=_training_sudo_config(
                decoys_enabled=False,
                grant_subjects=["user"],
            ),
        )
        observed = {
            generator.generate(seed=seed, binary_name="mawk").metadata["sudoers_file"]
            for seed in range(20)
        }

        assert len(observed) > 1
        assert all(path.startswith("/etc/sudoers.d/") for path in observed)
        assert all("gtfo" not in path.lower() for path in observed)

    @pytest.mark.parametrize(
        "name",
        ("ops.local", "backup~", "ops job", ".hidden", "../ops"),
    )
    def test_rejects_sudoers_file_names_ignored_by_sudo(self, name):
        with pytest.raises(ValueError, match="sudo_layout.sudoers_file_names"):
            SudoGtfobinsGenerator(
                config_dir=Path("conf/gtfobins/catalog/sudo"),
                allowlist=["mawk"],
                config=_training_sudo_config(),
                layout={
                    **get_generator_default("sudo_layout", {}),
                    "sudoers_file_names": [name],
                },
            )

    def test_allowlist_entries_exist_in_catalog(self):
        available = {
            path.stem for path in Path("conf/gtfobins/catalog/sudo").glob("*.yaml")
        }
        allowlist = set(load_sudo_allowlist())

        assert allowlist <= available

    def test_live_yaml_pools_exclude_tar_and_docker(self):
        allowlist = set(load_sudo_allowlist())

        for binary in ("tar", "docker"):
            assert binary not in allowlist

    def test_generated_scenarios_have_exactly_one_root_exploit_step(self, generator):
        for binary in load_sudo_allowlist():
            scenario = generator.generate(seed=0, binary_name=binary)
            root_steps = _root_steps(scenario)

            assert len(root_steps) == 1, binary
            assert root_steps[0]["arguments"]["command"] == scenario.metadata[
                "preferred_exploit_cmd"
            ]

    def test_multi_command_catalog_entries_use_one_preferred_exploit(self, generator):
        scenario = generator.generate(seed=0, binary_name="php")
        root_steps = _root_steps(scenario)

        assert len(scenario.metadata["exploit_cmds"]) > 1
        assert len(root_steps) == 1
        assert root_steps[0]["arguments"]["command"] == scenario.metadata[
            "exploit_cmds"
        ][0]

    def test_rejects_explicit_tar_binary(self, generator):
        with pytest.raises(ValueError, match="tar is intentionally excluded"):
            generator.generate(seed=0, binary_name="tar")

    def test_default_sampling_uses_flat_allowlist(self, generator):
        allowlist = set(load_sudo_allowlist())
        sampled = [
            generator.generate(seed=seed).metadata["binary_name"] for seed in range(200)
        ]

        assert set(sampled) <= allowlist
        assert len(set(sampled)) > 1

    def test_rejects_explicit_docker_binary(self, generator):
        with pytest.raises(ValueError, match="docker is intentionally excluded"):
            generator.generate(seed=0, binary_name="docker")


# =============================================================================
# Integration smoke test
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_procedural_sudo_bare_sudo_l_matches_solution_guidance():
    generator = _sudo_generator(config=_training_sudo_config(grant_subjects=["user"]))
    scenario = generator.generate(seed=0, binary_name="mawk")
    meta = scenario.metadata
    scenario_config = scenario.to_scenario_config()
    ssh_config = ssh_config_from_env()

    async with PrivEscScenario(ssh_config, scenario_config) as sc:
        result = await sc.exec_command("sudo -l")
        assert result.exit_code == 0, result.output
        assert meta["binary_path"] in result.output
        assert meta["decoy_wrapper_paths"] == []


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.parametrize("binary", _LIVE_SUDO_SMOKE_BINARIES)
async def test_procedural_sudo_selected_binaries_smoke(binary):
    generator = _sudo_generator(config=_training_sudo_config(grant_subjects=["user"]))
    await _assert_solution_succeeds(generator.generate(seed=0, binary_name=binary))


@pytest.mark.asyncio
@pytest.mark.slow
async def test_procedural_sudo_smoke():
    """Test Sudo exploits: verify exploit spawns root shell, password works, and flag is readable."""
    ssh_config = ssh_config_from_env()

    allowlist = load_sudo_allowlist()
    generator = _sudo_generator(
        allowlist=allowlist,
        config=_training_sudo_config(grant_subjects=["user"]),
    )
    available_binaries = set(generator.list_binaries())
    binaries = [
        binary for binary in _LIVE_SUDO_SMOKE_BINARIES if binary in available_binaries
    ]
    if not binaries:
        pytest.skip("No Sudo GTFOBins configs available.")

    log.info(f"Testing {len(binaries)} Sudo binaries: {binaries}")

    for binary in binaries:
        scenario = generator.generate(seed=0, binary_name=binary)
        meta = scenario.metadata
        exploit_tool_calls = scenario.solution["exploit_tool_calls"]
        log.info(f"[{binary}] exploit_tool_calls: {exploit_tool_calls}")

        scenario_config = scenario.to_scenario_config()

        async with PrivEscScenario(ssh_config, scenario_config) as sc:
            log.info(f"[{binary}] Checking sudo -l output...")
            sudo_l = await sc.exec_command(
                f"echo '{scenario.container_password}' | sudo -S -l"
            )
            assert sudo_l.exit_code == 0, (
                f"sudo -l failed for {binary}: {sudo_l.output}"
            )
            assert meta["binary_path"] in sudo_l.output, (
                f"Real GTFO entry missing from sudo -l for {binary}"
            )
            for decoy_path in meta["decoy_wrapper_paths"]:
                assert decoy_path in sudo_l.output, (
                    f"Decoy entry missing from sudo -l for {binary}: {decoy_path}"
                )
            assert "/usr/bin/tar" not in sudo_l.output, (
                f"tar leaked into sudo -l for {binary}"
            )
            log.info(f"[{binary}] ✓ sudo -l shows real entry")

            log.info(f"[{binary}] Testing Sudo exploits...")
            for exploit in exploit_tool_calls:
                await run_exploit(sc, exploit, scenario.category, is_alternative=False)
            log.info(f"[{binary}] ✓ Sudo exploit succeeded")

            log.info(f"[{binary}] Testing test_credentials...")
            creds = await sc.test_credentials("root", meta["root_password"])
            assert creds.success and creds.got_root, (
                f"test_credentials failed for {binary}"
            )
            log.info(f"[{binary}] ✓ test_credentials succeeded")

            log.info(f"[{binary}] Reading flag...")
            flag_result = await sc.exec_command(
                f"echo '{meta['root_password']}' | su -c 'cat /root/flag.txt' root"
            )
            assert flag_result.got_root, f"Root proof read failed for {binary}"
            log.info(f"[{binary}] ✓ Root proof verified")
