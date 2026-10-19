"""Tests for procedural Cron scenario generators.

Note: Cron integration tests require roughly one minute per scenario.
"""

import logging

import pytest

from src.generators.cron_wildcard import CronWildcardGenerator
from src.generators.cron_writable_script import CronWritableScriptGenerator
from src.generators.base import get_generator_default
from src.gym.scenario import PrivEscScenario

from .test_generators_base import ssh_config_from_env

log = logging.getLogger(__name__)
logging.getLogger("asyncssh").setLevel(logging.WARNING)

CRON_WILDCARD_DIRS = CronWildcardGenerator().backup_dirs
_CRON_WRITABLE = CronWritableScriptGenerator()
CRON_SCRIPT_NAMES = _CRON_WRITABLE.script_names


async def _assert_cron_writable_script_solution_succeeds(scenario) -> None:
    ssh_config = ssh_config_from_env()
    meta = scenario.metadata

    async with PrivEscScenario(ssh_config, scenario.to_scenario_config()) as sc:
        script_path = meta["script_path"]

        log.info("script_path: %s", script_path)

        log.info("Checking cron job...")
        cron_check = await sc.exec_command(f"cat {meta['job_path']}")
        assert script_path in cron_check.output, "Cron job not found"
        log.info("✓ Cron job exists")

        log.info("Checking script permissions...")
        write_check = await sc.exec_command(f"test -w {script_path}")
        ls_result = await sc.exec_command(f"ls -la {script_path}")
        assert write_check.exit_code == 0, f"Script not writable: {ls_result.output}"
        assert meta["script_root"] == f"/home/{meta['user']}"
        assert script_path.startswith(f"/home/{meta['user']}/")
        assert meta["user"] in ls_result.output, ls_result.output
        log.info("✓ Script is writable")

        log.info("Overwriting script...")
        result = await sc.exec_command(meta["exploit_cmd"])
        assert result.exit_code == 0, f"Failed to overwrite script: {result.output}"
        log.info("✓ Script overwritten")

        log.info("Waiting %s seconds for cron to execute...", meta["wait_seconds"])
        wait_result = await sc.exec_command(
            f"sleep {meta['wait_seconds']} && ls -la /bin/bash"
        )
        assert "rws" in wait_result.output, (
            f"SUID not set on bash: {wait_result.output}"
        )
        log.info("✓ SUID bit set on /bin/bash")

        log.info("Testing root shell...")
        root_result = await sc.exec_command("/bin/bash -p -c 'whoami'")
        assert "root" in root_result.output, f"Not root: {root_result.output}"
        log.info("✓ Got root shell")

        log.info("Reading flag...")
        flag_result = await sc.exec_command("/bin/bash -p -c 'cat /root/flag.txt'")
        assert flag_result.got_root, "Root proof read failed"
        log.info("✓ Root proof verified")


# =============================================================================
# Unit tests - CronWildcardGenerator
# =============================================================================


class TestCronWildcardGenerator:
    """Test Cron Wildcard generator."""

    @pytest.fixture
    def generator(self) -> CronWildcardGenerator:
        return CronWildcardGenerator()

    def test_category(self, generator):
        scenario = generator.generate(seed=0)
        assert scenario.category == "cron_wildcard"

    def test_deterministic(self, generator):
        s1 = generator.generate(seed=42)
        s2 = generator.generate(seed=42)
        assert s1.to_dict() == s2.to_dict()

    def test_different_seeds_different_dirs(self, generator):
        # With enough seeds, we should get different backup dirs
        dirs = {generator.generate(seed=i).metadata["backup_dir"] for i in range(20)}
        assert len(dirs) > 1, "Should use multiple backup directories across seeds"

    def test_setup_script_creates_cron_job(self, generator):
        scenario = generator.generate(seed=0)
        assert scenario.metadata["job_path"] in scenario.setup_script
        assert "tar" in scenario.setup_script

    def test_job_name_varies_across_seeds(self, generator):
        job_names = {
            generator.generate(seed=i).metadata["job_name"] for i in range(30)
        }
        assert len(job_names) > 1
        assert job_names <= set(generator.job_names)

    @pytest.mark.parametrize(
        "name",
        ("nightly.backup", "backup~", "backup job", ".hidden", "../backup"),
    )
    def test_rejects_job_names_ignored_by_cron(self, name):
        config = {
            **get_generator_default("cron_wildcard", {}),
            "job_names": [name],
        }

        with pytest.raises(ValueError, match="cron_wildcard.job_names"):
            CronWildcardGenerator(config=config)

    def test_rejects_legacy_job_path_ignored_by_cron(self):
        config = {
            **get_generator_default("cron_wildcard", {}),
            "job_names": [],
            "job_path": "/etc/cron.d/nightly.backup",
        }

        with pytest.raises(ValueError, match="cron_wildcard.job_path"):
            CronWildcardGenerator(config=config)

    def test_cron_command_has_wildcard(self, generator):
        scenario = generator.generate(seed=0)
        cron_cmd = scenario.metadata["cron_command"]
        assert "*" in cron_cmd, "Cron command must use wildcard for exploit"
        assert "tar" in cron_cmd

    def test_solution_has_exploit_steps(self, generator):
        scenario = generator.generate(seed=0)
        steps = scenario.solution["exploit_tool_calls"]
        assert len(steps) == 2
        assert steps[0]["arguments"]["command"] == scenario.metadata["exploit_cmd"]
        assert "--checkpoint" in steps[0]["arguments"]["command"]
        assert steps[1]["arguments"]["command"] == scenario.metadata["wait_cmd"]
        assert f"sleep {generator.wait_seconds}" in steps[1]["arguments"]["command"]
        assert steps[1]["arguments"]["command"].endswith("&& /bin/bash -p")


# =============================================================================
# Unit tests - CronWritableScriptGenerator
# =============================================================================


class TestCronWritableScriptGenerator:
    """Test Cron Writable Script generator."""

    @pytest.fixture
    def generator(self) -> CronWritableScriptGenerator:
        return CronWritableScriptGenerator()

    def test_category(self, generator):
        scenario = generator.generate(seed=0)
        assert scenario.category == "cron_writable_script"

    def test_deterministic(self, generator):
        s1 = generator.generate(seed=42)
        s2 = generator.generate(seed=42)
        assert s1.to_dict() == s2.to_dict()

    def test_different_seeds_different_scripts(self, generator):
        scripts = {
            generator.generate(seed=i).metadata["script_path"] for i in range(20)
        }
        assert len(scripts) > 1, "Should use multiple script names across seeds"

    def test_setup_script_creates_writable_file(self, generator):
        scenario = generator.generate(seed=0)
        assert f"chmod {generator.script_mode}" in scenario.setup_script
        assert (
            f"chown {scenario.metadata['user']}:{scenario.metadata['user']}"
            in scenario.setup_script
        )
        assert "groupadd" not in scenario.setup_script
        assert scenario.metadata["job_path"] in scenario.setup_script

    def test_job_name_varies_across_seeds(self, generator):
        job_names = {
            generator.generate(seed=i).metadata["job_name"] for i in range(30)
        }
        assert len(job_names) > 1
        assert job_names <= set(generator.job_names)

    @pytest.mark.parametrize(
        "name",
        ("nightly.backup", "backup~", "backup job", ".hidden", "../backup"),
    )
    def test_rejects_job_names_ignored_by_cron(self, name):
        config = {
            **get_generator_default("cron_writable_script", {}),
            "job_names": [name],
        }

        with pytest.raises(ValueError, match="cron_writable_script.job_names"):
            CronWritableScriptGenerator(config=config)

    def test_rejects_legacy_job_path_ignored_by_cron(self):
        config = {
            **get_generator_default("cron_writable_script", {}),
            "job_names": [],
            "job_path": "/etc/cron.d/nightly.backup",
        }

        with pytest.raises(ValueError, match="cron_writable_script.job_path"):
            CronWritableScriptGenerator(config=config)

    def test_script_path_is_user_home_by_default(self, generator):
        scenario = generator.generate(seed=0)
        assert scenario.metadata["script_path"].startswith(
            f"/home/{scenario.metadata['user']}/"
        )

    def test_solution_has_exploit_steps(self, generator):
        scenario = generator.generate(seed=0)
        steps = scenario.solution["exploit_tool_calls"]
        assert len(steps) == 2
        assert "chmod u+s /bin/bash" in steps[0]["arguments"]["command"]
        assert steps[0]["arguments"]["command"] == scenario.metadata["exploit_cmd"]
        assert steps[1]["arguments"]["command"] == scenario.metadata["wait_cmd"]
        assert f"sleep {generator.wait_seconds}" in steps[1]["arguments"]["command"]
        assert steps[1]["arguments"]["command"].endswith("&& /bin/bash -p")
        assert scenario.hint == "Check cron jobs for user-writable scripts."


# =============================================================================
# Integration smoke tests
# =============================================================================


@pytest.mark.parametrize("backup_dir", CRON_WILDCARD_DIRS)
@pytest.mark.asyncio
@pytest.mark.slow
async def test_procedural_cron_wildcard_smoke(backup_dir: str):
    """Test Cron Wildcard: verify tar checkpoint injection works.

    This test takes ~70 seconds due to cron timing.
    """
    ssh_config = ssh_config_from_env()

    generator = CronWildcardGenerator(backup_dirs=[backup_dir])
    scenario = generator.generate(seed=0)
    meta = scenario.metadata

    log.info(f"backup_dir: {meta['backup_dir']}")
    log.info(f"cron_command: {meta['cron_command']}")

    scenario_config = scenario.to_scenario_config()

    async with PrivEscScenario(ssh_config, scenario_config) as sc:
        backup_dir = meta["backup_dir"]

        # 1. Verify cron job exists
        log.info("Checking cron job...")
        cron_check = await sc.exec_command(f"cat {meta['job_path']}")
        assert "tar" in cron_check.output, "Cron job not found"
        log.info("✓ Cron job exists")

        # 2. Create exploit files
        log.info("Creating exploit files...")
        result = await sc.exec_command(meta["exploit_cmd"])
        assert result.exit_code == 0, f"Failed to create exploit files: {result.output}"
        log.info("✓ Exploit files created")

        # 3. Verify files exist
        ls_result = await sc.exec_command(f"ls -la {backup_dir}/")
        log.info(f"Backup dir contents:\n{ls_result.output}")
        assert "--checkpoint=1" in ls_result.output

        # 4. Wait for cron to execute.
        log.info("Waiting %s seconds for cron to execute...", meta["wait_seconds"])
        wait_result = await sc.exec_command(
            f"sleep {meta['wait_seconds']} && ls -la /bin/bash"
        )
        assert "rws" in wait_result.output, (
            f"SUID not set on bash: {wait_result.output}"
        )
        log.info("✓ SUID bit set on /bin/bash")

        # 5. Get root shell
        log.info("Testing root shell...")
        root_result = await sc.exec_command("/bin/bash -p -c 'whoami'")
        assert "root" in root_result.output, f"Not root: {root_result.output}"
        log.info("✓ Got root shell")

        # 6. Read flag
        log.info("Reading flag...")
        flag_result = await sc.exec_command("/bin/bash -p -c 'cat /root/flag.txt'")
        assert flag_result.got_root, "Root proof read failed"
        log.info("✓ Root proof verified")


@pytest.mark.parametrize("script_name", CRON_SCRIPT_NAMES)
@pytest.mark.asyncio
@pytest.mark.slow
async def test_procedural_cron_writable_script_smoke(script_name: str):
    """Each configured script name should remain solvable."""
    scenario = CronWritableScriptGenerator(script_names=[script_name]).generate(seed=0)
    await _assert_cron_writable_script_solution_succeeds(scenario)
