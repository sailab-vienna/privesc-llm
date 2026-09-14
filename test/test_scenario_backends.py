import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import cast

import asyncssh
import pytest

from src.config import ScenarioConfig, SSHConfig
from src.gym.backends import (
    AuthOutcome,
    LocalDockerBackend,
    RemoteSshDockerBackend,
    TIMEOUT_EXIT_CODE,
    build_backend,
    calculate_command_timeout,
)
from src.gym.backends import local_docker as local_docker_module
from src.gym.backends import remote_ssh as remote_ssh_module
from src.gym.backends.base import (
    root_proof_install_script,
    run_ssh_command_with_root_proof,
    test_credentials_on_port as _test_credentials_on_port,
)
from src.gym.backends.host_ssh_pool import HostSSHConnectionPool, HostSSHKey
from src.gym.scenario import AuthResult, ExecResult, PrivEscScenario
from src.scenarios.static import StaticScenarioSource


def _docker_available() -> bool:
    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _image_available(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _ssh_for_tests() -> SSHConfig:
    home = str(Path.home())
    return SSHConfig(
        user=os.getenv("PRIVESC_USER", "root"),
        key_path=os.getenv("PRIVESC_KEY", f"{home}/.ssh/id_ed25519_privesc"),
        servers=os.getenv("PRIVESC_SSH_SERVERS", "localhost:22"),
        host_path=os.getenv(
            "PRIVESC_HOST_PATH",
            f"{home}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        ),
    )


async def _no_sleep(_seconds: float) -> None:
    return None


def test_build_backend_selects_remote_and_local() -> None:
    ssh_cfg = _ssh_for_tests()
    scenario = ScenarioConfig(
        name="01_vuln_suid_gtfo", image="privesc_01_vuln_suid_gtfo"
    )

    scenario.backend = "remote_ssh"
    backend = build_backend(ssh_cfg, scenario, lambda _: None)
    assert isinstance(backend, RemoteSshDockerBackend)

    scenario.backend = "local_docker"
    backend = build_backend(ssh_cfg, scenario, lambda _: None)
    assert LocalDockerBackend is not None
    assert isinstance(backend, LocalDockerBackend)


def test_remote_backend_preferred_host_round_robins_ssh_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remote_ssh_module, "_INSTANCE_COUNTER", 0)
    ssh_cfg = SSHConfig(
        user="root",
        key_path="/tmp/key",
        servers="host1:1001,host2:1002,host3:1003",
    )

    keys = []
    for i in range(4):
        backend = RemoteSshDockerBackend(
            ssh_cfg,
            ScenarioConfig(name=f"demo{i}", image="privesc_demo"),
            lambda _: None,
        )
        candidates = backend._host_key_candidates()
        keys.append(candidates[backend._instance_index % len(candidates)])

    assert [(key.host, key.port) for key in keys] == [
        ("host1", 1001),
        ("host2", 1002),
        ("host3", 1003),
        ("host1", 1001),
    ]


def test_calculate_command_timeout_respects_sleep_and_slow_commands() -> None:
    scenario = ScenarioConfig(
        name="01_vuln_suid_gtfo",
        image="privesc_01_vuln_suid_gtfo",
        base_command_timeout=2,
        slow_command_timeout=7,
        max_command_timeout=9,
    )
    assert calculate_command_timeout("id", scenario) == 2
    assert calculate_command_timeout("sleep 3 && id", scenario) == 5
    assert calculate_command_timeout("docker ps", scenario) == 7
    assert calculate_command_timeout("ssh -o StrictHostKeyChecking=no root@localhost", scenario) == 7
    assert calculate_command_timeout("docker ps && sleep 10", scenario) == 9


def test_auth_result_requires_verified_root_proof() -> None:
    result = AuthResult(
        user="monitoring",
        password="monitoring",
        success=True,
        uid=0,
    )

    assert result.got_root is False

    verified = AuthResult(
        user="monitoring",
        password="monitoring",
        success=True,
        uid=0,
        root_verified=True,
    )

    assert verified.got_root is True


def test_tool_result_schema_excludes_hidden_root_proof_state() -> None:
    result = ExecResult(
        command="cat /root/flag.txt",
        output="proof-value",
        exit_code=0,
        root_verified=True,
    )

    assert asdict(result) == {
        "got_root": True,
        "command": "cat /root/flag.txt",
        "output": "proof-value",
        "exit_code": 0,
        "timed_out": False,
    }


def test_root_proof_install_script_sets_root_only_flag() -> None:
    script = root_proof_install_script("proof-value")

    assert "proof-value" in script
    assert "/root/flag.txt" in script
    assert "chown root:root /root/flag.txt" in script
    assert "chmod 0600 /root/flag.txt" in script


@pytest.mark.asyncio
async def test_timed_out_command_uses_hidden_probe_for_root_proof() -> None:
    class _Stdin:
        def __init__(self, process: "_TimedOutProcess") -> None:
            self._process = process

        def write(self, data: str) -> None:
            self._process.writes.append(data)

        async def drain(self) -> None:
            return None

    class _TimedOutProcess:
        def __init__(self) -> None:
            self.stdin = _Stdin(self)
            self.writes: list[str] = []
            self.terminated = False

        async def wait(self, check: bool = False, timeout: int | None = None):
            raise asyncssh.TimeoutError(
                None,
                "docker run --rm -it -v /:/mnt alpine chroot /mnt /bin/sh",
                None,
                None,
                None,
                None,
                "root@container:/#",
                "",
            )

        def terminate(self) -> None:
            self.terminated = True

        def collect_output(self) -> tuple[str, str]:
            write = self.writes[-1]
            begin = write.split("printf '\\n", 1)[1].split("\\n'", 1)[0]
            end = write.rsplit("printf '\\n", 1)[1].split("\\n'", 1)[0]
            return f"{begin}\nproof-value\n{end}", ""

        def kill(self) -> None:
            return None

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    class _Conn:
        def __init__(self) -> None:
            self.process = _TimedOutProcess()

        async def create_process(self, *_args, **_kwargs):
            return self.process

    conn = _Conn()

    result = await run_ssh_command_with_root_proof(
        cast(asyncssh.SSHClientConnection, conn),
        "docker run --rm -it -v /:/mnt alpine chroot /mnt /bin/sh",
        1,
        term_cols=80,
        term_rows=24,
        expected_root_proof="proof-value",
    )

    assert result.timed_out is True
    assert result.root_verified is True
    assert "proof-value" not in result.output
    assert conn.process.terminated is True


@pytest.mark.asyncio
async def test_nested_docker_prompt_does_not_verify_root_proof() -> None:
    class _Completed:
        stdout = "root@nested:/#"
        returncode = 0

    class _Process:
        async def wait(self, check: bool = False, timeout: int | None = None):
            return _Completed()

    class _Conn:
        async def create_process(self, *_args, **_kwargs):
            return _Process()

    result = await run_ssh_command_with_root_proof(
        cast(asyncssh.SSHClientConnection, _Conn()),
        "docker run --rm -it alpine /bin/sh",
        1,
        term_cols=80,
        term_rows=24,
        expected_root_proof="proof-value",
    )

    assert result.root_verified is False


@pytest.mark.asyncio
async def test_local_backend_reset_installs_random_root_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert LocalDockerBackend is not None
    backend = LocalDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(
            name="demo",
            image="privesc_demo",
            backend="local_docker",
            setup_script="echo setup",
        ),
        lambda _: None,
    )
    events: list[str] = []
    scripts: list[str] = []

    async def _fake_cleanup_runtime() -> None:
        events.append("cleanup")

    async def _fake_start_container() -> None:
        events.append("start")

    async def _fake_run_setup_script() -> None:
        events.append("setup")

    async def _fake_exec_as_root(script: str) -> tuple[int, str]:
        events.append("proof")
        scripts.append(script)
        return 0, ""

    async def _fake_connect_to_container() -> None:
        events.append("connect")

    monkeypatch.setattr(backend, "_cleanup_runtime", _fake_cleanup_runtime)
    monkeypatch.setattr(backend, "_start_container", _fake_start_container)
    monkeypatch.setattr(backend, "_run_setup_script", _fake_run_setup_script)
    monkeypatch.setattr(backend, "_exec_as_root", _fake_exec_as_root)
    monkeypatch.setattr(backend, "_connect_to_container", _fake_connect_to_container)

    await backend.reset()
    first_proof = backend._root_proof
    await backend.reset()

    assert events[:5] == ["cleanup", "start", "setup", "proof", "connect"]
    assert first_proof is not None
    assert backend._root_proof is not None
    assert backend._root_proof != first_proof
    assert first_proof in scripts[0]
    assert backend._root_proof in scripts[1]


@pytest.mark.asyncio
async def test_remote_backend_installs_root_proof_via_stdin() -> None:
    backend = RemoteSshDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo", image="privesc_demo", backend="remote_ssh"),
        lambda _: None,
    )
    backend._cname = "demo-container"
    backend._root_proof = "proof-value"
    calls: list[tuple[str, str | None]] = []

    async def _fake_run_host(
        cmd: str,
        timeout: int = 120,
        input_data: str | None = None,
    ):
        calls.append((cmd, input_data))

        class _Result:
            stdout = ""

        return _Result()

    backend._run_host = _fake_run_host  # type: ignore[method-assign]

    await backend._install_root_proof()

    assert len(calls) == 1
    assert calls[0][0] == "docker exec -i demo-container /bin/sh"
    assert calls[0][1] is not None
    assert "proof-value" not in calls[0][0]
    assert "proof-value" in calls[0][1]


@pytest.mark.asyncio
@pytest.mark.slow
async def test_local_backend_smoke_exec_command() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon is not available")
    if not _image_available("privesc_01_vuln_suid_gtfo:latest"):
        pytest.skip("Scenario image privesc_01_vuln_suid_gtfo:latest is missing")

    instance = StaticScenarioSource(scenarios=["01_vuln_suid_gtfo"]).build(0)
    instance.config.backend = "local_docker"
    ssh_cfg = _ssh_for_tests()

    async with PrivEscScenario(ssh_cfg, instance.config) as sc:
        result = await sc.exec_command("whoami")
        assert result.exit_code == 0
        assert "lowpriv" in result.output


@pytest.mark.asyncio
@pytest.mark.slow
async def test_local_backend_smoke_hidden_probe_verifies_suid_shell() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon is not available")
    if not _image_available("privesc_01_vuln_suid_gtfo:latest"):
        pytest.skip("Scenario image privesc_01_vuln_suid_gtfo:latest is missing")

    instance = StaticScenarioSource(scenarios=["01_vuln_suid_gtfo"]).build(0)
    instance.config.backend = "local_docker"
    ssh_cfg = _ssh_for_tests()

    async with PrivEscScenario(ssh_cfg, instance.config) as sc:
        result = await sc.exec_command(
            "/usr/bin/find . -exec /bin/sh -p \\; -quit"
        )

    assert result.timed_out is True
    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert result.got_root is True
    assert "privesc-root-proof-" not in result.output


@pytest.mark.asyncio
@pytest.mark.slow
async def test_remote_backend_smoke_exec_command() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon is not available")
    if not _image_available("privesc_01_vuln_suid_gtfo:latest"):
        pytest.skip("Scenario image privesc_01_vuln_suid_gtfo:latest is missing")

    key_path = os.getenv("PRIVESC_KEY")
    if not key_path or not Path(key_path).exists():
        pytest.skip("Set PRIVESC_KEY to a valid SSH private key (use source .env)")

    instance = StaticScenarioSource(scenarios=["01_vuln_suid_gtfo"]).build(0)
    instance.config.backend = "remote_ssh"
    ssh_cfg = _ssh_for_tests()

    async with PrivEscScenario(ssh_cfg, instance.config) as sc:
        result = await sc.exec_command("whoami")
        assert result.exit_code == 0
        assert "lowpriv" in result.output


@pytest.mark.asyncio
async def test_remote_backend_reconnect_rebuilds_forwarded_route() -> None:
    backend = RemoteSshDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo", image="privesc_demo", backend="remote_ssh"),
        lambda _: None,
    )

    events: list[str] = []

    class _Conn:
        def close(self) -> None:
            events.append("close_cont_ssh")

        async def wait_closed(self) -> None:
            events.append("wait_closed_cont_ssh")

    class _Fwd:
        def close(self) -> None:
            events.append("close_forward")

    backend._cont_ssh = cast(asyncssh.SSHClientConnection, _Conn())
    backend._fwd_server = cast(asyncssh.SSHListener, _Fwd())

    async def _fake_get_container_port() -> int:
        events.append("get_container_port")
        return 2222

    async def _fake_forward_port(port: int) -> None:
        events.append(f"forward_port:{port}")
        backend._local_port = 12345

    async def _fake_connect_to_container() -> None:
        events.append("connect_to_container")

    backend._get_container_port = _fake_get_container_port
    backend._forward_port = _fake_forward_port  # type: ignore[method-assign]
    backend._connect_to_container = _fake_connect_to_container

    await backend._reconnect_container()

    assert events == [
        "close_cont_ssh",
        "wait_closed_cont_ssh",
        "close_forward",
        "get_container_port",
        "forward_port:2222",
        "connect_to_container",
    ]


@pytest.mark.asyncio
async def test_remote_backend_test_credentials_refreshes_route_after_connection_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = RemoteSshDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo", image="privesc_demo", backend="remote_ssh"),
        lambda _: None,
    )
    backend._local_port = 45678

    attempts = {"count": 0}
    reconnects = {"count": 0}

    async def _fake_test_credentials_on_port(
        port: int | None,
        user: str,
        password: str,
        connect_timeout: int,
        expected_root_proof: str | None = None,
    ) -> AuthOutcome:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise asyncssh.ConnectionLost("boom")
        return AuthOutcome(
            success=True, message=f"ok:{port}:{user}:{password}:{connect_timeout}"
        )

    async def _fake_reconnect_container() -> None:
        reconnects["count"] += 1
        backend._local_port = 56789

    monkeypatch.setattr(
        "src.gym.backends.remote_ssh.test_credentials_on_port",
        _fake_test_credentials_on_port,
    )
    backend._reconnect_container = _fake_reconnect_container  # type: ignore[method-assign]

    result = await backend.test_credentials("root", "root")

    assert result.success is True
    assert reconnects["count"] == 1
    assert attempts["count"] == 2
    assert "56789" in result.message


@pytest.mark.asyncio
async def test_remote_backend_run_command_raises_after_connection_retry_exhaustion() -> (
    None
):
    backend = RemoteSshDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo", image="privesc_demo", backend="remote_ssh"),
        lambda _: None,
    )

    async def _fake_reconnect_container() -> None:
        raise asyncssh.ConnectionLost("boom")

    backend._reconnect_container = _fake_reconnect_container  # type: ignore[method-assign]
    backend._cont_ssh = None

    with pytest.raises(asyncssh.ConnectionLost):
        await backend.run_command("id", 1)


@pytest.mark.asyncio
async def test_local_backend_test_credentials_refreshes_route_after_connection_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert LocalDockerBackend is not None
    backend = LocalDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo", image="privesc_demo", backend="local_docker"),
        lambda _: None,
    )
    backend._local_port = 45678

    attempts = {"count": 0}
    reconnects = {"count": 0}

    async def _fake_test_credentials_on_port(
        port: int | None,
        user: str,
        password: str,
        connect_timeout: int,
        expected_root_proof: str | None = None,
    ) -> AuthOutcome:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise asyncssh.ConnectionLost("boom")
        return AuthOutcome(
            success=True, message=f"ok:{port}:{user}:{password}:{connect_timeout}"
        )

    async def _fake_reconnect_container() -> None:
        reconnects["count"] += 1
        backend._local_port = 56789

    monkeypatch.setattr(
        "src.gym.backends.local_docker.test_credentials_on_port",
        _fake_test_credentials_on_port,
    )
    backend._reconnect_container = _fake_reconnect_container  # type: ignore[method-assign]

    result = await backend.test_credentials("root", "root")

    assert result.success is True
    assert reconnects["count"] == 1
    assert attempts["count"] == 2
    assert "56789" in result.message


@pytest.mark.asyncio
async def test_local_backend_shares_docker_client_across_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert LocalDockerBackend is not None

    original_client = local_docker_module._DOCKER_CLIENT
    original_refs = local_docker_module._DOCKER_CLIENT_REFS
    local_docker_module._DOCKER_CLIENT = None
    local_docker_module._DOCKER_CLIENT_REFS = 0

    calls = {"from_env": 0, "close": 0}

    class _Client:
        def close(self) -> None:
            calls["close"] += 1

    client = _Client()

    def _fake_from_env():
        calls["from_env"] += 1
        return client

    monkeypatch.setattr(local_docker_module.docker, "from_env", _fake_from_env)

    backend1 = LocalDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo1", image="privesc_demo", backend="local_docker"),
        lambda _: None,
    )
    backend2 = LocalDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo2", image="privesc_demo", backend="local_docker"),
        lambda _: None,
    )

    try:
        await backend1.start()
        await backend2.start()

        assert backend1._client is client
        assert backend2._client is client
        assert calls["from_env"] == 1
        assert local_docker_module._DOCKER_CLIENT_REFS == 2

        await backend1.close()
        assert calls["close"] == 0
        assert local_docker_module._DOCKER_CLIENT_REFS == 1

        await backend2.close()
        assert calls["close"] == 1
        assert local_docker_module._DOCKER_CLIENT_REFS == 0
        assert local_docker_module._DOCKER_CLIENT is None
    finally:
        local_docker_module._DOCKER_CLIENT = original_client
        local_docker_module._DOCKER_CLIENT_REFS = original_refs


@pytest.mark.asyncio
async def test_remote_ready_command_failure_is_not_backend_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", _no_sleep)

    backend = RemoteSshDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(
            name="demo",
            image="privesc_demo",
            backend="remote_ssh",
            ready_command="check-ready",
            max_command_timeout=7,
        ),
        lambda _: None,
    )

    attempts = {"count": 0}

    class _Conn:
        async def run(self, *_args, **_kwargs):
            attempts["count"] += 1

            class _Result:
                exit_status = 1
                stdout = "not ready"
                stderr = ""

            return _Result()

    backend._cont_ssh = cast(asyncssh.SSHClientConnection, _Conn())

    with pytest.raises(RuntimeError, match="Ready command failed"):
        await backend._run_ready_command()

    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_local_ready_command_failure_is_not_backend_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", _no_sleep)

    assert LocalDockerBackend is not None
    backend = LocalDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(
            name="demo",
            image="privesc_demo",
            backend="local_docker",
            ready_command="check-ready",
            max_command_timeout=7,
        ),
        lambda _: None,
    )

    attempts = {"count": 0}

    class _Conn:
        async def run(self, *_args, **_kwargs):
            attempts["count"] += 1

            class _Result:
                exit_status = 1
                stdout = "not ready"
                stderr = ""

            return _Result()

    backend._cont_ssh = cast(asyncssh.SSHClientConnection, _Conn())

    with pytest.raises(RuntimeError, match="Ready command failed"):
        await backend._run_ready_command()

    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_remote_ready_command_reconnects_after_connection_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", _no_sleep)

    backend = RemoteSshDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(
            name="demo",
            image="privesc_demo",
            backend="remote_ssh",
            ready_command="check-ready",
            max_command_timeout=7,
        ),
        lambda _: None,
    )

    reconnects = {"count": 0}

    class _WorkingConn:
        async def run(self, *_args, **_kwargs):
            class _Result:
                exit_status = 0
                stdout = "ready"
                stderr = ""

            return _Result()

    class _FlakyConn:
        async def run(self, *_args, **_kwargs):
            raise asyncssh.ConnectionLost("boom")

    async def _fake_reconnect_container() -> None:
        reconnects["count"] += 1
        backend._cont_ssh = cast(asyncssh.SSHClientConnection, _WorkingConn())

    backend._cont_ssh = cast(asyncssh.SSHClientConnection, _FlakyConn())
    backend._reconnect_container = _fake_reconnect_container  # type: ignore[method-assign]

    await backend._run_ready_command()

    assert reconnects["count"] == 1


@pytest.mark.asyncio
async def test_local_ready_command_reconnects_after_connection_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", _no_sleep)

    assert LocalDockerBackend is not None
    backend = LocalDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(
            name="demo",
            image="privesc_demo",
            backend="local_docker",
            ready_command="check-ready",
            max_command_timeout=7,
        ),
        lambda _: None,
    )

    reconnects = {"count": 0}

    class _WorkingConn:
        async def run(self, *_args, **_kwargs):
            class _Result:
                exit_status = 0
                stdout = "ready"
                stderr = ""

            return _Result()

    class _FlakyConn:
        async def run(self, *_args, **_kwargs):
            raise asyncssh.ConnectionLost("boom")

    async def _fake_reconnect_container() -> None:
        reconnects["count"] += 1
        backend._cont_ssh = cast(asyncssh.SSHClientConnection, _WorkingConn())

    backend._cont_ssh = cast(asyncssh.SSHClientConnection, _FlakyConn())
    backend._reconnect_container = _fake_reconnect_container  # type: ignore[method-assign]

    await backend._run_ready_command()

    assert reconnects["count"] == 1


@pytest.mark.asyncio
async def test_remote_get_container_port_retries_until_mapping_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.utils.retry.asyncio.sleep", _no_sleep)

    backend = RemoteSshDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo", image="privesc_demo", backend="remote_ssh"),
        lambda _: None,
    )
    backend._cname = "demo"

    attempts = {"count": 0}

    class _Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    async def _fake_run_host(_cmd: str):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return _Result("")
        return _Result("0.0.0.0:2222")

    backend._run_host = _fake_run_host  # type: ignore[method-assign]

    port = await backend._get_container_port()

    assert port == 2222
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_test_credentials_on_port_disables_ssh_config_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    closed = {"count": 0}

    class _Result:
        def __init__(self, stdout: str) -> None:
            self.exit_status = 0
            self.stdout = stdout

    class _Conn:
        async def run(
            self,
            command: str,
            check: bool = False,
            timeout: int | None = None,
        ) -> _Result:
            assert check is False
            if command == "id -u":
                return _Result("0\n")
            assert command == "cat /root/flag.txt 2>/dev/null"
            assert timeout == 5
            return _Result("proof-value")

        def close(self) -> None:
            closed["count"] += 1

        async def wait_closed(self) -> None:
            return None

    async def _fake_connect(*_args, **kwargs):
        captured.update(kwargs)
        return _Conn()

    monkeypatch.setattr("src.gym.backends.base.asyncssh.connect", _fake_connect)

    result = await _test_credentials_on_port(2222, "root", "pw", 5, "proof-value")

    assert result.success is True
    assert result.uid == 0
    assert result.root_verified is True
    assert captured["config"] is None
    assert captured["client_keys"] is None
    assert captured["agent_path"] is None
    assert captured["host_based_auth"] is False
    assert captured["public_key_auth"] is False
    assert captured["kbdint_auth"] is False
    assert captured["password_auth"] is True
    assert captured["preferred_auth"] == "password"
    assert closed["count"] == 1


@pytest.mark.asyncio
async def test_local_backend_connect_to_container_disables_ssh_config_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert LocalDockerBackend is not None
    backend = LocalDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo", image="privesc_demo", backend="local_docker"),
        lambda _: None,
    )
    backend._local_port = 2222
    captured: dict[str, object] = {}

    class _Conn:
        pass

    async def _fake_connect(*_args, **kwargs):
        captured.update(kwargs)
        return _Conn()

    monkeypatch.setattr("src.gym.backends.local_docker.asyncssh.connect", _fake_connect)

    await backend._connect_to_container()

    assert captured["config"] is None
    assert captured["agent_path"] is None
    assert captured["port"] == 2222
    assert captured["username"] == backend._scen_cfg.container_user


@pytest.mark.asyncio
async def test_remote_backend_connect_to_container_disables_ssh_config_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = RemoteSshDockerBackend(
        _ssh_for_tests(),
        ScenarioConfig(name="demo", image="privesc_demo", backend="remote_ssh"),
        lambda _: None,
    )
    backend._local_port = 2222
    captured: dict[str, object] = {}

    class _Conn:
        pass

    async def _fake_connect(*_args, **kwargs):
        captured.update(kwargs)
        return _Conn()

    monkeypatch.setattr("src.gym.backends.remote_ssh.asyncssh.connect", _fake_connect)

    await backend._connect_to_container()

    assert captured["config"] is None
    assert captured["port"] == 2222
    assert captured["username"] == backend._scen_cfg.container_user


@pytest.mark.asyncio
async def test_host_ssh_pool_open_connection_disables_ssh_config_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = HostSSHConnectionPool()
    key = HostSSHKey(host="host", port=22, user="root", key_path="/tmp/key")
    captured: dict[str, object] = {}

    class _Conn:
        pass

    async def _fake_connect(*_args, **kwargs):
        captured.update(kwargs)
        return _Conn()

    monkeypatch.setattr("src.gym.backends.host_ssh_pool.asyncssh.connect", _fake_connect)

    await pool._open_connection(key)

    assert captured["config"] is None
    assert captured["client_keys"] == ["/tmp/key"]
