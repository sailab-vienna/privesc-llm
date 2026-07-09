from __future__ import annotations

import asyncio
import asyncssh
import contextlib
import docker
import os
import subprocess
import uuid
from asyncio import Lock
from docker.errors import APIError, NotFound
from docker.models.containers import Container
from typing import Callable, Optional

from src.config import ScenarioConfig, SSHConfig, resolve_default_host_path
from src.gym.backends.base import (
    AuthOutcome,
    CommandOutcome,
    KEEPALIVE_COUNT_MAX,
    KEEPALIVE_INTERVAL,
    MAX_CONNECTION_RETRIES,
    ReadyCommandError,
    TRANSIENT_CONN_ERRORS,
    TIMEOUT_EXIT_CODE,
    build_setup_script,
    new_root_proof,
    resolve_image_ref,
    root_proof_install_script,
    run_ssh_command_with_root_proof,
    test_credentials_on_port,
)
from src.utils.retry import retry_async


_DOCKER_CLIENT: docker.DockerClient | None = None
_DOCKER_CLIENT_REFS = 0
_DOCKER_CLIENT_LOCK = Lock()


async def _acquire_docker_client() -> docker.DockerClient:
    global _DOCKER_CLIENT, _DOCKER_CLIENT_REFS
    async with _DOCKER_CLIENT_LOCK:
        if _DOCKER_CLIENT is None:
            _DOCKER_CLIENT = await asyncio.to_thread(docker.from_env)
        _DOCKER_CLIENT_REFS += 1
        return _DOCKER_CLIENT


async def _release_docker_client() -> None:
    global _DOCKER_CLIENT, _DOCKER_CLIENT_REFS
    async with _DOCKER_CLIENT_LOCK:
        if _DOCKER_CLIENT_REFS <= 0:
            return
        _DOCKER_CLIENT_REFS -= 1
        if _DOCKER_CLIENT_REFS != 0:
            return
        client = _DOCKER_CLIENT
        _DOCKER_CLIENT = None
    if client is not None:
        await asyncio.to_thread(client.close)


class LocalDockerBackend:
    def __init__(
        self,
        ssh_cfg: SSHConfig,
        scen_cfg: ScenarioConfig,
        log: Callable[[str], None],
    ) -> None:
        self._ssh_cfg = ssh_cfg
        self._scen_cfg = scen_cfg
        self._log = log
        self._cname_prefix = f"privesc_{scen_cfg.name}"
        self._cname: Optional[str] = None
        self._local_port: Optional[int] = None
        self._client: Optional[docker.DockerClient] = None
        self._cont_ssh: Optional[asyncssh.SSHClientConnection] = None
        self._reconnect_lock = Lock()
        self._uses_shared_client = False
        self._root_proof: str | None = None

    def _auth_connect_timeout(self) -> int:
        return int(getattr(self._scen_cfg, "auth_connect_timeout", 15))

    @property
    def local_port(self) -> int | None:
        return self._local_port

    async def start(self) -> None:
        if self._client is not None:
            return
        self._client = await _acquire_docker_client()
        self._uses_shared_client = True

    async def reset(self) -> None:
        await self._cleanup_runtime()
        await self._start_container()
        if self._scen_cfg.setup_script:
            self._log("[dim]  Running setup script[/]")
            await self._run_setup_script()
        self._root_proof = new_root_proof()
        await self._install_root_proof()
        await self._connect_to_container()
        if self._scen_cfg.ready_command:
            await self._run_ready_command()

    async def close(self) -> None:
        await self._cleanup_runtime()
        if self._uses_shared_client:
            self._uses_shared_client = False
            self._client = None
            await _release_docker_client()
            return
        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None

    def _host_env_with_path(self) -> dict[str, str]:
        env = dict(os.environ)
        host_path_raw = self._ssh_cfg.host_path or resolve_default_host_path()
        host_path = os.path.expandvars(host_path_raw)
        if "$HOME" in host_path:
            host_path = resolve_default_host_path()
        env["PATH"] = f"{host_path}:{env.get('PATH', '')}"
        return env

    async def _run_pre_command(self) -> None:
        command = self._scen_cfg.pre_command
        if not command:
            return

        def _run() -> None:
            subprocess.run(
                ["/bin/bash", "-lc", command],
                check=True,
                env=self._host_env_with_path(),
            )

        await asyncio.to_thread(_run)

    @retry_async(
        attempts=MAX_CONNECTION_RETRIES,
        exceptions=(APIError, OSError),
        base_delay=0.5,
        factor=2.0,
    )
    async def _ensure_registry_mirror(self) -> None:
        if self._client is None:
            raise RuntimeError("Docker client not initialized")

        def _ensure() -> None:
            assert self._client is not None
            try:
                self._client.networks.get("privesc-net")
            except NotFound:
                self._client.networks.create("privesc-net")

            try:
                mirror = self._client.containers.get("privesc-registry-mirror")
                if mirror.status != "running":
                    mirror.start()
                return
            except NotFound:
                pass

            self._client.containers.run(
                "registry:3",
                detach=True,
                restart_policy={"Name": "always"},
                name="privesc-registry-mirror",
                network="privesc-net",
                ports={"5000/tcp": 5000},
                environment={
                    "REGISTRY_PROXY_REMOTEURL": "https://registry-1.docker.io"
                },
                volumes={
                    "/var/cache/privesc-registry": {
                        "bind": "/var/lib/registry",
                        "mode": "rw",
                    }
                },
            )

        await asyncio.to_thread(_ensure)

    def _parse_extra_args(self) -> tuple[bool, str | None]:
        privileged = False
        network_name: str | None = None
        idx = 0
        args = list(self._scen_cfg.extra_args)
        while idx < len(args):
            arg = args[idx]
            if arg == "--privileged":
                privileged = True
                idx += 1
                continue
            if arg == "--network" and idx + 1 < len(args):
                network_name = args[idx + 1]
                idx += 2
                continue
            raise ValueError(f"Unsupported extra_args entry for local backend: {arg}")
        return privileged, network_name

    async def _start_container(self) -> None:
        if self._client is None:
            raise RuntimeError("Docker client not initialized")

        self._cname = f"{self._cname_prefix}_{uuid.uuid4().hex[:6]}"
        if self._scen_cfg.log_message:
            self._log(f"[dim]  {self._scen_cfg.log_message}[/]")
        await self._run_pre_command()

        privileged, network_name = self._parse_extra_args()
        if self._scen_cfg.use_registry_mirror:
            await self._ensure_registry_mirror()
            if network_name is None:
                network_name = "privesc-net"

        command_in_container = self._scen_cfg.command
        if command_in_container and not self._scen_cfg.manual_sshd_start:
            command_in_container += " && exec /usr/sbin/sshd -D -e"

        image = resolve_image_ref(self._scen_cfg.image)
        cname = self._cname
        if cname is None:
            raise RuntimeError("Container name not initialized")
        command_arg: list[str] | None = None
        if command_in_container:
            command_arg = ["sh", "-c", command_in_container]
        privileged_flag = bool(privileged)

        def _run_container() -> Container:
            assert self._client is not None
            if network_name:
                return self._client.containers.run(
                    image,
                    command=command_arg,
                    detach=True,
                    remove=True,
                    name=cname,
                    ports={"22/tcp": None},
                    privileged=privileged_flag,
                    network=network_name,
                )
            return self._client.containers.run(
                image,
                command=command_arg,
                detach=True,
                remove=True,
                name=cname,
                ports={"22/tcp": None},
                privileged=privileged_flag,
            )

        self._log(f"[dim]  Launching container {self._cname}[/]")
        container = await asyncio.to_thread(_run_container)
        self._local_port = await self._wait_for_port(container)

    @retry_async(
        attempts=MAX_CONNECTION_RETRIES,
        exceptions=(APIError, OSError),
        base_delay=0.5,
        factor=2.0,
    )
    async def _wait_for_port(self, container: Container) -> int:
        for _ in range(100):
            try:
                await asyncio.to_thread(container.reload)
            except NotFound:
                raise RuntimeError("Container exited before SSH port was ready")
            attrs = container.attrs if isinstance(container.attrs, dict) else {}
            net = attrs.get("NetworkSettings")
            net_dict = net if isinstance(net, dict) else {}
            ports = net_dict.get("Ports")
            ports_dict = ports if isinstance(ports, dict) else {}
            bindings = ports_dict.get("22/tcp")
            if bindings and isinstance(bindings, list) and bindings[0].get("HostPort"):
                return int(bindings[0]["HostPort"])
            await asyncio.sleep(0.1)
        raise RuntimeError("Container SSH port mapping not available")

    @retry_async(
        attempts=MAX_CONNECTION_RETRIES,
        exceptions=(asyncssh.Error, OSError),
        base_delay=0.5,
        factor=2.0,
    )
    async def _connect_to_container(self) -> None:
        if self._local_port is None:
            raise RuntimeError("Container SSH port not available")
        self._cont_ssh = await asyncssh.connect(
            "127.0.0.1",
            port=self._local_port,
            username=self._scen_cfg.container_user,
            password=self._scen_cfg.container_password,
            known_hosts=None,
            config=None,
            connect_timeout=self._auth_connect_timeout(),
            keepalive_interval=KEEPALIVE_INTERVAL,
            keepalive_count_max=KEEPALIVE_COUNT_MAX,
        )
        self._log(
            f"[green]  Ready: ssh {self._scen_cfg.container_user}@127.0.0.1 -p {self._local_port}[/]"
        )

    async def _reconnect_container(self) -> None:
        async with self._reconnect_lock:
            if self._cont_ssh:
                with contextlib.suppress(Exception):
                    self._cont_ssh.close()
                    await self._cont_ssh.wait_closed()
            self._cont_ssh = None
            await self._connect_to_container()

    def _get_container(self) -> Container:
        if self._client is None or self._cname is None:
            raise RuntimeError("Container not started")
        return self._client.containers.get(self._cname)

    async def _exec_as_root(self, script: str) -> tuple[int, str]:
        container = self._get_container()

        def _run() -> tuple[int, str]:
            exit_code, output = container.exec_run(
                cmd=["sh", "-lc", script],
                user="root",
                tty=True,
            )
            if isinstance(output, bytes):
                return int(exit_code), output.decode("utf-8", errors="replace")
            return int(exit_code), str(output)

        return await asyncio.to_thread(_run)

    async def _run_setup_script(self) -> None:
        inner = build_setup_script(
            container_user=self._scen_cfg.container_user,
            container_password=self._scen_cfg.container_password,
            setup_script=self._scen_cfg.setup_script or "",
        )
        exit_code, output = await self._exec_as_root(inner)
        if exit_code != 0:
            raise RuntimeError(f"Setup script failed with exit {exit_code}: {output}")

    async def _install_root_proof(self) -> None:
        if self._root_proof is None:
            raise RuntimeError("Root proof not initialized")
        exit_code, output = await self._exec_as_root(
            root_proof_install_script(self._root_proof)
        )
        if exit_code != 0:
            raise RuntimeError(
                f"Root proof setup failed with exit {exit_code}: {output}"
            )

    @retry_async(
        attempts=MAX_CONNECTION_RETRIES,
        exceptions=(asyncssh.Error, OSError),
        base_delay=0.5,
        factor=2.0,
    )
    async def _run_ready_command(self) -> None:
        if self._cont_ssh is None:
            await self._reconnect_container()
        if self._cont_ssh is None:
            raise RuntimeError("Container SSH connection not available")
        timeout = self._scen_cfg.ready_command_timeout or self._scen_cfg.max_command_timeout
        try:
            result = await self._cont_ssh.run(
                self._scen_cfg.ready_command,
                timeout=timeout,
                check=False,
            )
            if result.exit_status != 0:
                exit_status = result.exit_status
                output = str(result.stdout or result.stderr or "").strip()
                raise ReadyCommandError(
                    f"Ready command failed with exit {exit_status}: {output}",
                    exit_status=exit_status if isinstance(exit_status, int) else None,
                )
        except asyncssh.TimeoutError as timeout_err:
            output = str(timeout_err.stdout or "").strip()
            raise ReadyCommandError(
                f"Ready command timed out after {timeout}s: {output}",
                exit_status=TIMEOUT_EXIT_CODE,
            )
        except TRANSIENT_CONN_ERRORS:
            self._cont_ssh = None
            raise

    async def run_command(self, command: str, timeout: int) -> CommandOutcome:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_CONNECTION_RETRIES + 1):
            if self._cont_ssh is None:
                try:
                    await self._reconnect_container()
                except Exception as reconnect_err:
                    last_error = reconnect_err
                    if attempt >= MAX_CONNECTION_RETRIES:
                        break
                    continue

            try:
                if self._cont_ssh is None:
                    raise RuntimeError("Container SSH connection not available")
                return await run_ssh_command_with_root_proof(
                    self._cont_ssh,
                    command,
                    timeout=timeout,
                    term_cols=self._scen_cfg.term_cols,
                    term_rows=self._scen_cfg.term_rows,
                    expected_root_proof=self._root_proof,
                )
            except TRANSIENT_CONN_ERRORS as conn_err:
                last_error = conn_err
                self._cont_ssh = None
                if attempt >= MAX_CONNECTION_RETRIES:
                    break

        if last_error is not None:
            raise last_error
        raise OSError("Container connection not available")

    async def test_credentials(self, user: str, password: str) -> AuthOutcome:
        last_error: Exception | None = None
        for attempt in range(1, MAX_CONNECTION_RETRIES + 1):
            if self._local_port is None:
                try:
                    await self._reconnect_container()
                except Exception as reconnect_err:
                    last_error = reconnect_err
                    if attempt >= MAX_CONNECTION_RETRIES:
                        break
                    continue

            try:
                return await test_credentials_on_port(
                    self._local_port,
                    user,
                    password,
                    self._auth_connect_timeout(),
                    self._root_proof,
                )
            except TRANSIENT_CONN_ERRORS as conn_err:
                last_error = conn_err
                self._cont_ssh = None
                if attempt >= MAX_CONNECTION_RETRIES:
                    break
                try:
                    await self._reconnect_container()
                except Exception as reconnect_err:
                    last_error = reconnect_err

        if last_error is not None:
            raise last_error
        raise OSError("Container connection not available")

    async def _cleanup_runtime(self) -> None:
        self._root_proof = None
        if self._cont_ssh:
            with contextlib.suppress(Exception):
                self._cont_ssh.close()
                await self._cont_ssh.wait_closed()
            self._cont_ssh = None
        if self._client is None:
            self._cname = None
            self._local_port = None
            return

        if self._cname is None:
            self._local_port = None
            return

        cname = self._cname

        def _remove() -> None:
            assert self._client is not None
            try:
                container = self._client.containers.get(cname)
                container.remove(force=True)
            except (NotFound, APIError):
                return

        await asyncio.to_thread(_remove)
        self._cname = None
        self._local_port = None
