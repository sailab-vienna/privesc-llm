"""Tests for procedural SUID GTFOBins generator."""

import json
import logging
import shlex
from pathlib import Path

import pytest

from src.generators.base import get_generator_default, load_gtfobins_allowlist
from src.generators.suid_gtfobins import SuidGtfobinsGenerator
from src.gym.scenario import PrivEscScenario

from test.solution_utils import run_exploit
from .test_generators_base import ssh_config_from_env

log = logging.getLogger(__name__)
logging.getLogger("asyncssh").setLevel(logging.WARNING)

FORBIDDEN_DECOY_NAMES = {
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
_PREFERRED_SUID_TEST_BINARIES = ("chroot", "xargs", "rsync", "php", "node")
_HEAD_LIMIT_SUID_TEST_BINARIES = (
    "start-stop-daemon",
    "chroot",
    "softlimit",
    "xargs",
)


def _select_suid_test_binary(
    preferred_binaries: tuple[str, ...] = _PREFERRED_SUID_TEST_BINARIES,
) -> str:
    allowlist = set(load_gtfobins_allowlist("suid_allowlist"))
    for binary in preferred_binaries:
        if binary in allowlist:
            return binary
    raise AssertionError("Need at least one preferred SUID test binary in allowlist")


SUID_TEST_BINARY = _select_suid_test_binary()


# =============================================================================
# SUID-specific tests
# =============================================================================


class TestSuidGtfobinsGenerator:
    """Test SUID-specific generator behavior."""

    @pytest.fixture
    def generator(self) -> SuidGtfobinsGenerator:
        return SuidGtfobinsGenerator(config_dir=Path("conf/gtfobins/catalog/suid"))

    def test_category_is_suid(self, generator):
        scenario = generator.generate(seed=0)
        assert scenario.category == "suid_gtfobins"

    def test_setup_script_sets_suid_bit(self, generator):
        scenario = generator.generate(seed=0)
        assert "chmod 4755" in scenario.setup_script
        assert "chown root:root" in scenario.setup_script

    def test_exploit_cmd_not_prefixed(self, generator):
        """SUID exploits shouldn't have sudo prefix."""
        scenario = generator.generate(seed=0, binary_name=SUID_TEST_BINARY)
        command = scenario.solution["exploit_tool_calls"][-1]["arguments"]["command"]
        assert not command.startswith("sudo ")

    def test_uses_catalog_binary_path_by_default(self, generator):
        scenario = generator.generate(seed=0, binary_name=SUID_TEST_BINARY)
        binary_path = scenario.metadata["binary_path"]

        assert binary_path.endswith(f"/{SUID_TEST_BINARY}")
        assert not binary_path.startswith("/usr/local/bin/")
        assert "install_suid_copy" not in scenario.setup_script
        assert scenario.metadata["decoys_enabled"] is False
        assert scenario.metadata["decoy_count"] == 0

    def test_decoys_keep_catalog_target_path(self):
        generator = SuidGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/suid"),
            config={"decoys_enabled": True},
        )
        for seed in range(12):
            scenario = generator.generate(seed=seed, binary_name=SUID_TEST_BINARY)
            binary_path = scenario.metadata["binary_path"]

            assert binary_path.endswith(f"/{SUID_TEST_BINARY}")
            assert not binary_path.startswith("/usr/local/bin/")
            assert "readlink -f" not in scenario.setup_script
            assert "install_suid_copy" in scenario.setup_script
            assert "install_suid_existing" not in scenario.setup_script
            assert binary_path in scenario.solution["vulnerability"]

    def test_setup_script_adds_realistic_suid_decoys_through_one_path(self):
        generator = SuidGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/suid"),
            config={"decoys_enabled": True},
        )
        scenario = generator.generate(seed=0, binary_name=SUID_TEST_BINARY)
        decoy_paths = scenario.metadata["decoy_paths"]

        layout = get_generator_default("suid_layout", {})
        helper_names = set(layout.get("decoy_binaries", []))
        assert (
            int(layout.get("decoy_count_min", 0))
            <= len(decoy_paths)
            <= int(layout.get("decoy_count_max", 0))
        )
        assert scenario.metadata["decoy_paths"] == decoy_paths
        assert len(scenario.metadata["decoy_source_paths"]) == len(decoy_paths)
        assert all(path.startswith(("/opt/", "/var/lib/")) for path in decoy_paths)
        assert all(Path(path).name in helper_names for path in decoy_paths)
        assert scenario.metadata["binary_path"] not in decoy_paths
        assert "system_decoy_paths" not in scenario.metadata
        assert "install_suid_existing" not in scenario.setup_script

    def test_suid_decoys_install_before_target(self):
        generator = SuidGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/suid"),
            config={"decoys_enabled": True},
        )
        scenario = generator.generate(seed=0, binary_name=SUID_TEST_BINARY)
        decoy_paths = scenario.metadata["decoy_paths"]

        layout = get_generator_default("suid_layout", {})
        assert (
            int(layout.get("decoy_count_min", 0))
            <= len(decoy_paths)
            <= int(layout.get("decoy_count_max", 0))
        )
        target_setup = (
            f"chown root:root {scenario.metadata['binary_path']} "
            f"&& chmod 4755 {scenario.metadata['binary_path']}"
        )
        assert scenario.setup_script.index(target_setup) > scenario.setup_script.index(
            decoy_paths[-1]
        )

    def test_decoy_names_are_helper_style(self):
        generator = SuidGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/suid"),
            config={"decoys_enabled": True},
        )
        scenario = generator.generate(seed=0, binary_name=SUID_TEST_BINARY)
        layout = get_generator_default("suid_layout", {})
        helper_names = set(layout.get("decoy_binaries", []))
        installed_names = {Path(path).name for path in scenario.metadata["decoy_paths"]}

        assert all(name in helper_names for name in installed_names)
        assert not any(
            name == forbidden or name.endswith(f"-{forbidden}")
            for name in installed_names
            for forbidden in FORBIDDEN_DECOY_NAMES
        )

    def test_decoys_keep_catalog_exploit_commands(self):
        generator = SuidGtfobinsGenerator(
            config_dir=Path("conf/gtfobins/catalog/suid"),
            config={"decoys_enabled": True},
        )
        scenario = generator.generate(seed=0, binary_name="xargs")
        binary_path = scenario.metadata["binary_path"]
        commands = scenario.metadata["exploit_cmds"]
        root_command = scenario.solution["exploit_tool_calls"][-1]["arguments"][
            "command"
        ]

        assert binary_path == "/usr/bin/xargs"
        assert all("/usr/bin/xargs" in command for command in commands)
        assert any(
            command.startswith(f"echo x | {binary_path}") for command in commands
        )
        assert root_command == scenario.metadata["preferred_exploit_cmd"]

    def test_forbidden_benchmark_strings_absent(self, generator):
        forbidden_strings = (
            '"binary_name": "find"',
            '"binary_name": "python"',
            '"binary_name": "python3"',
            '"binary_name": "python3.11"',
            "/usr/bin/find",
            "/usr/bin/python",
            "/usr/bin/python3",
            "/usr/bin/python3.11",
            "os.setuid(0)",
            "os.execl('/bin/sh', 'sh', '-c', 'whoami')",
        )

        for seed in range(10):
            payload = json.dumps(
                generator.generate(seed=seed).to_dict(), sort_keys=True
            )
            for forbidden in forbidden_strings:
                assert forbidden not in payload, (
                    f"Found forbidden benchmark string: {forbidden}"
                )


# =============================================================================
# Integration smoke test
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_procedural_suid_generated_solution_smoke():
    ssh_config = ssh_config_from_env()
    generator = SuidGtfobinsGenerator(config_dir=Path("conf/gtfobins/catalog/suid"))
    scenario = generator.generate(seed=0, binary_name=SUID_TEST_BINARY)

    async with PrivEscScenario(ssh_config, scenario.to_scenario_config()) as sc:
        for exploit in scenario.solution["exploit_tool_calls"]:
            await run_exploit(sc, exploit, scenario.category, is_alternative=False)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_procedural_suid_does_not_shadow_recon_commands():
    ssh_config = ssh_config_from_env()
    generator = SuidGtfobinsGenerator(config_dir=Path("conf/gtfobins/catalog/suid"))
    expected_paths = {
        "whoami": "/usr/bin/whoami",
        "id": "/usr/bin/id",
        "groups": "/usr/bin/groups",
        "find": "/usr/bin/find",
        "printenv": "/usr/bin/printenv",
        "stat": "/usr/bin/stat",
        "uname": "/usr/bin/uname",
    }

    for seed in range(2):
        scenario = generator.generate(seed=seed, binary_name=SUID_TEST_BINARY)
        async with PrivEscScenario(ssh_config, scenario.to_scenario_config()) as sc:
            for command, expected_path in expected_paths.items():
                result = await sc.exec_command(f"command -v {command}")
                assert result.exit_code == 0, (
                    f"Failed to resolve {command} for seed {seed}: {result.output}"
                )
                assert result.output.strip() == expected_path, (
                    f"{command} shadowed for seed {seed}: {result.output.strip()}"
                )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_procedural_suid_smoke():
    """Test SUID exploits: verify exploit spawns root shell, password works, and flag is readable."""
    ssh_config = ssh_config_from_env()

    allowlist = load_gtfobins_allowlist("suid_allowlist")
    generator = SuidGtfobinsGenerator(
        config_dir=Path("conf/gtfobins/catalog/suid"),
        allowlist=allowlist,
    )
    binaries = [
        binary
        for binary in _PREFERRED_SUID_TEST_BINARIES
        if binary in generator.list_binaries()
    ][:2]
    if not binaries:
        pytest.skip("No SUID GTFOBins configs available.")

    log.info(f"Testing {len(binaries)} SUID binaries: {binaries}")

    for binary in binaries:
        scenario = generator.generate(seed=0, binary_name=binary)
        meta = scenario.metadata
        exploit_tool_calls = scenario.solution["exploit_tool_calls"]
        log.info(f"[{binary}] exploit_tool_calls: {exploit_tool_calls}")

        scenario_config = scenario.to_scenario_config()

        async with PrivEscScenario(ssh_config, scenario_config) as sc:
            log.info(f"[{binary}] Testing SUID exploits...")
            for exploit in exploit_tool_calls:
                await run_exploit(sc, exploit, scenario.category, is_alternative=False)
            log.info(f"[{binary}] ✓ SUID exploit succeeded")

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


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.parametrize("binary_name", _HEAD_LIMIT_SUID_TEST_BINARIES)
async def test_procedural_suid_decoys_keep_target_out_of_head_limited_find(
    binary_name: str,
):
    ssh_config = ssh_config_from_env()
    generator = SuidGtfobinsGenerator(
        config_dir=Path("conf/gtfobins/catalog/suid"),
        config={"decoys_enabled": True},
    )

    scenario = generator.generate(seed=0, binary_name=binary_name)
    binary_path = str(scenario.metadata["binary_path"])

    async with PrivEscScenario(ssh_config, scenario.to_scenario_config()) as sc:
        present = await sc.exec_command(f"test -u {shlex.quote(binary_path)}")
        assert present.exit_code == 0, f"Missing SUID target: {binary_path}"

        resolved = await sc.exec_command(f"readlink -f {shlex.quote(binary_path)}")
        assert resolved.exit_code == 0, f"Could not resolve target: {binary_path}"
        reported_path = resolved.output.strip()

        find_result = await sc.exec_command(
            "find / -perm -4000 -type f 2>/dev/null "
            f"| grep -Fx -- {shlex.quote(reported_path)}"
        )
        assert find_result.exit_code == 0, (
            f"find did not report target: {reported_path}"
        )

        head_result = await sc.exec_command(
            "find / -perm -4000 -type f 2>/dev/null | head -n 10"
        )
        first_ten = [
            line.strip() for line in head_result.output.splitlines() if line.strip()
        ]
        assert reported_path not in first_ten, (
            f"Real target appeared in first 10 SUID results: {first_ten}"
        )
