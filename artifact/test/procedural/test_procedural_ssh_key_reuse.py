"""Tests for procedural SSH Key Reuse generator."""

import logging

import pytest

from src.generators.ssh_key_reuse import SshKeyReuseGenerator
from src.gym.scenario import PrivEscScenario

from .test_generators_base import ssh_config_from_env

log = logging.getLogger(__name__)
logging.getLogger("asyncssh").setLevel(logging.WARNING)

_SSH_GENERATOR = SshKeyReuseGenerator()
SSH_KEY_CONFIG_LIST = _SSH_GENERATOR.ssh_key_configs
USER_SSH_SUFFIXES = _SSH_GENERATOR.user_ssh_dir_suffixes
SHARED_SSH_LOCATIONS = _SSH_GENERATOR.shared_ssh_dirs


# =============================================================================
# Unit tests
# =============================================================================


class TestSshKeyReuseGenerator:
    """Test SSH Key Reuse generator."""

    @pytest.fixture
    def generator(self) -> SshKeyReuseGenerator:
        return SshKeyReuseGenerator()

    def test_category(self, generator):
        scenario = generator.generate(seed=0)
        assert scenario.category == "ssh_key_reuse"

    def test_deterministic(self, generator):
        s1 = generator.generate(seed=42)
        s2 = generator.generate(seed=42)
        assert s1.to_dict() == s2.to_dict()

    def test_different_seeds_different_scenarios(self, generator):
        s1 = generator.generate(seed=1)
        s2 = generator.generate(seed=2)
        assert s1.to_dict() != s2.to_dict()

    def test_different_seeds_vary_configs(self, generator):
        """Different seeds should produce varied key types/names/locations."""
        scenarios = [generator.generate(seed=i) for i in range(30)]
        key_types = {s.metadata["key_type"] for s in scenarios}
        key_names = {s.metadata["key_name"] for s in scenarios}
        ssh_dirs = {s.metadata["ssh_dir"] for s in scenarios}
        assert len(key_types) > 1 and len(key_names) > 1 and len(ssh_dirs) > 1

    def test_holdout_key_not_in_configs(self, generator):
        """Benchmark uses 'id_rsa' - should not be in training configs."""
        key_names = [config[2] for config in SSH_KEY_CONFIG_LIST]
        assert "id_rsa" not in key_names

    def test_paths_consistent(self, generator):
        """Key paths should be correctly constructed."""
        scenario = generator.generate(seed=0)
        meta = scenario.metadata
        assert meta["key_path"] == f"{meta['ssh_dir']}/{meta['key_name']}"
        assert meta["key_path"] in meta["exploit_cmd"]

    def test_ssh_target_comes_from_config(self, generator):
        scenario = generator.generate(seed=0)
        assert scenario.metadata["ssh_target"] == generator.ssh_target

    def test_solution_lists_key_directory_then_ssh(self, generator):
        scenario = generator.generate(seed=0)
        steps = scenario.solution["exploit_tool_calls"]
        assert len(steps) == 2
        assert all(step["function"] == "exec_command" for step in steps)
        assert steps[0]["arguments"]["command"] == (
            f"ls -la {scenario.metadata['ssh_dir']}"
        )
        assert steps[1]["arguments"]["command"] == scenario.metadata["exploit_cmd"]

    def test_setup_creates_authorized_keys(self, generator):
        """Setup script should configure SSH access."""
        scenario = generator.generate(seed=0)
        assert "authorized_keys" in scenario.setup_script
        assert "/root/.ssh" in scenario.setup_script


# =============================================================================
# Integration smoke test
# =============================================================================


@pytest.mark.parametrize("ssh_key_config", SSH_KEY_CONFIG_LIST)
def test_procedural_ssh_key_reuse_configs_generate_paths(
    ssh_key_config: tuple[str, str, str],
):
    generator = SshKeyReuseGenerator(
        ssh_key_configs=[ssh_key_config],
        user_ssh_dir_suffixes=[USER_SSH_SUFFIXES[0]]
        if USER_SSH_SUFFIXES
        else ["/.ssh"],
        shared_ssh_dirs=[],
    )

    scenario = generator.generate(seed=0)
    meta = scenario.metadata
    key_type, keygen_args, key_name = ssh_key_config

    assert meta["key_type"] == key_type
    assert meta["key_name"] == key_name
    assert meta["key_path"].endswith(f"/{key_name}")
    assert keygen_args in scenario.setup_script
    assert meta["key_path"] in meta["exploit_cmd"]


@pytest.mark.parametrize("ssh_dir", [*USER_SSH_SUFFIXES, *SHARED_SSH_LOCATIONS])
def test_procedural_ssh_key_reuse_dirs_generate_paths(ssh_dir: str):
    if ssh_dir in USER_SSH_SUFFIXES:
        generator = SshKeyReuseGenerator(
            ssh_key_configs=[SSH_KEY_CONFIG_LIST[0]],
            user_ssh_dir_suffixes=[ssh_dir],
            shared_ssh_dirs=[],
        )
    else:
        generator = SshKeyReuseGenerator(
            ssh_key_configs=[SSH_KEY_CONFIG_LIST[0]],
            user_ssh_dir_suffixes=[],
            shared_ssh_dirs=[ssh_dir],
        )

    scenario = generator.generate(seed=0)
    meta = scenario.metadata

    if ssh_dir in USER_SSH_SUFFIXES:
        expected_dir = f"/home/{scenario.container_user}{ssh_dir}"
    else:
        expected_dir = ssh_dir
    assert meta["ssh_dir"] == expected_dir
    assert meta["key_path"].startswith(f"{expected_dir}/")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_procedural_ssh_key_reuse_solution_smoke():
    ssh_config = ssh_config_from_env()
    generator = SshKeyReuseGenerator(
        ssh_key_configs=[SSH_KEY_CONFIG_LIST[0]],
        user_ssh_dir_suffixes=[USER_SSH_SUFFIXES[0]]
        if USER_SSH_SUFFIXES
        else ["/.ssh"],
        shared_ssh_dirs=[],
    )

    scenario = generator.generate(seed=0)
    meta = scenario.metadata
    log.info(f"Testing SSH key reuse with {meta['key_path']} ({meta['key_type']})")

    scenario_config = scenario.to_scenario_config()

    async with PrivEscScenario(ssh_config, scenario_config) as sc:
        key_head = await sc.exec_command(f"head -n 1 {meta['key_path']}")
        assert "-----BEGIN" in key_head.output, "Should be a valid private key"
        ssh_result = await sc.exec_command(meta["exploit_cmd"])
        assert ssh_result.got_root, "SSH should grant root access"
        flag_result = await sc.exec_command(
            "ssh -o StrictHostKeyChecking=no "
            f"-i {meta['key_path']} root@{meta['ssh_target']} cat /root/flag.txt"
        )
        assert flag_result.got_root, "Root proof read failed"
