from __future__ import annotations

import asyncssh
import contextlib
import os
import shlex
import threading
import uuid
from asyncio import Lock
from typing import Callable, Optional

from src.config import (
    ScenarioConfig,
    SSHConfig,
    resolve_default_host_path,
    resolve_ssh_endpoints,
)
from src.gym.backends.base import (
    AuthOutcome,
    CommandOutcome,
    HOST_CMD_TIMEOUT,
    KEEPALIVE_COUNT_MAX,
    KEEPALIVE_INTERVAL,
    MAX_CONNECTION_RETRIES,
    ReadyCommandError,
    TRANSIENT_CONN_ERRORS,
    TIMEOUT_EXIT_CODE,
    build_setup_script,
    free_port,
    new_root_proof,
    resolve_image_ref,
    root_proof_install_script,
    run_ssh_command_with_root_proof,
    test_credentials_on_port,
)
from src.gym.backends.host_ssh_pool import HOST_SSH_POOL, HostSSHKey
from src.utils.retry import retry_async


_INSTANCE_COUNTER = 0
_INSTANCE_COUNTER_LOCK = threading.Lock()
_DOCKER_CONFIG_ERROR_MARKERS = (
    "invalid reference format",
    "unknown flag",
    "requires at least",
    "requires exactly",
)


def _next_instance_index() -> int:
    global _INSTANCE_COUNTER
    with _INSTANCE_COUNTER_LOCK:
        value = _INSTANCE_COUNTER
        _INSTANCE_COUNTER += 1
        return value


def _read_int_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    if not raw:
        return default
    return max(minimum, int(raw))


class RemoteSshDockerBackend:
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
        self._host_ssh: Optional[asyncssh.SSHClientConnection] = None
        self._host_ssh_key: Optional[HostSSHKey] = None
        self._runtime_host_key: Optional[HostSSHKey] = None
        self._cont_ssh: Optional[asyncssh.SSHClientConnection] = None
        self._fwd_server: Optional[asyncssh.SSHListener] = None
        self._local_port: Optional[int] = None
        self._reconnect_lock = Lock()
        self._instance_index = _next_instance_index()
        self._root_proof: str | None = None
        self._host_ssh_shards = _read_int_env(
            "PRIVESC_HOST_SSH_SHARDS", default=20, minimum=1
        )

    def _host_key_candidates(self) -> list[HostSSHKey]:
        endpoints = resolve_ssh_endpoints(self._ssh_cfg)
        shard = self._instance_index % self._host_ssh_shards
        return [
            HostSSHKey(
                host=endpoint.host,
                port=endpoint.port,
                user=self._ssh_cfg.user,
                key_path=self._ssh_cfg.key_path,
                shard=shard,
            )
            for endpoint in endpoints
        ]

    def _auth_connect_timeout(self) -> int:
        return int(getattr(self._scen_cfg, "auth_connect_timeout", 15))

    @property
    def local_port(self) -> int | None:
        return self._local_port

    async def start(self) -> None:
        await self._acquire_any_host_connection()

    async def reset(self) -> None:
        await self._cleanup_runtime()

        await self._start_container()
        await self._forward_port(await self._get_container_port())
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
        await self._release_host_connection()

    async def _acquire_any_host_connection(self) -> None:
        if self._host_ssh is not None:
            return
        candidates = self._host_key_candidates()
        preferred_index = self._instance_index % len(candidates)
        lease = await HOST_SSH_POOL.acquire_any(candidates, preferred_index)
        self._host_ssh = lease.connection
        self._host_ssh_key = lease.key
        self._log(f"[dim]  SSH to host {lease.key.host}:{lease.key.port}[/]")

    async def _acquire_pinned_host_connection(self, key: HostSSHKey) -> None:
        if self._host_ssh is not None and self._host_ssh_key == key:
            return
        await self._release_host_connection()
        self._host_ssh = await HOST_SSH_POOL.acquire(key)
        self._host_ssh_key = key
        self._log(f"[dim]  SSH to pinned host {key.host}:{key.port}[/]")

    async def _acquire_host_connection(self) -> None:
        if self._runtime_host_key is not None:
            await self._acquire_pinned_host_connection(self._runtime_host_key)
            return
        await self._acquire_any_host_connection()

    async def _release_host_connection(self) -> None:
        key = self._host_ssh_key
        self._host_ssh = None
        self._host_ssh_key = None
        if key is None:
            return
        await HOST_SSH_POOL.release(key)

    async def _mark_current_host_unhealthy(self, reason: str) -> None:
        key = self._host_ssh_key
        if key is None:
            return
        self._log(
            f"[yellow]  Marking host {key.host}:{key.port} unhealthy: {reason}[/]"
        )
        await HOST_SSH_POOL.mark_unhealthy(key, reason)
        await self._release_host_connection()

    async def _invalidate_current_host_connection(self) -> None:
        key = self._host_ssh_key
        self._host_ssh = None
        self._host_ssh_key = None
        if key is None:
            return
        await HOST_SSH_POOL.invalidate(key)
        await HOST_SSH_POOL.release(key)

    async def _run_host(
        self,
        cmd: str,
        timeout: int = HOST_CMD_TIMEOUT,
        input_data: str | None = None,
    ) -> asyncssh.SSHCompletedProcess:
        host_path = self._ssh_cfg.host_path or resolve_default_host_path()
        prefixed = f"PATH={host_path}:$PATH {cmd}"
        last_error: Exception | None = None
        for attempt in range(1, MAX_CONNECTION_RETRIES + 1):
            try:
                if self._host_ssh is None:
                    await self._acquire_host_connection()
                if self._host_ssh is None:
                    raise RuntimeError("Host SSH connection not established")
                return await self._host_ssh.run(
                    prefixed,
                    check=True,
                    timeout=timeout,
                    input=input_data,
                )
            except asyncssh.ProcessError as proc_err:
                self._log(
                    f"[yellow]  Host command failed exit={proc_err.exit_status}: {cmd}[/]"
                )
                stderr = str(proc_err.stderr or "").strip()
                if stderr:
                    self._log(f"[yellow]  Host stderr: {stderr}[/]")
                raise
            except TRANSIENT_CONN_ERRORS as conn_err:
                last_error = conn_err
                if self._runtime_host_key is not None:
                    await self._invalidate_current_host_connection()
                elif self._host_ssh_key is not None:
                    await self._mark_current_host_unhealthy(str(conn_err))
                if attempt >= MAX_CONNECTION_RETRIES:
                    break
                await self._acquire_host_connection()
        raise RuntimeError(
            f"Host connection error after {MAX_CONNECTION_RETRIES} attempts: {last_error}"
        )

    @retry_async(
        attempts=MAX_CONNECTION_RETRIES,
        exceptions=(asyncssh.ProcessError, asyncssh.Error, OSError, ValueError),
        base_delay=0.5,
        factor=2.0,
    )
    async def _ensure_registry_mirror(self) -> None:
        try:
            await self._run_host("docker network create privesc-net")
        except asyncssh.ProcessError:
            pass

        inspect_cmd = "docker inspect -f '{{.State.Running}}|{{.Config.Env}}' privesc-registry-mirror"
        try:
            res = await self._run_host(inspect_cmd)
            stdout = str(res.stdout or "")
            is_running_str, env_str = stdout.strip().split("|", 1)
            is_running = is_running_str == "true"
            is_configured = (
                "REGISTRY_PROXY_REMOTEURL=https://registry-1.docker.io" in env_str
            )
            if not is_configured:
                await self._run_host("docker rm -f privesc-registry-mirror")
            else:
                try:
                    net_check = await self._run_host(
                        "docker inspect -f '{{json .NetworkSettings.Networks}}' privesc-registry-mirror"
                    )
                    net_stdout = str(net_check.stdout or "")
                    if '"privesc-net":' not in net_stdout:
                        await self._run_host(
                            "docker network connect privesc-net privesc-registry-mirror"
                        )
                except (asyncssh.ProcessError, asyncssh.Error):
                    pass
                if not is_running:
                    await self._run_host("docker start privesc-registry-mirror")
                return
        except (asyncssh.ProcessError, ValueError):
            pass

        start_cmd = (
            "docker run -d --restart=always --name privesc-registry-mirror "
            "--network privesc-net "
            "-p 5000:5000 "
            "-e REGISTRY_PROXY_REMOTEURL=https://registry-1.docker.io "
            "-v /var/cache/privesc-registry:/var/lib/registry "
            "registry:3"
        )
        await self._run_host(start_cmd)

    async def _start_container(self) -> None:
        self._cname = f"{self._cname_prefix}_{uuid.uuid4().hex[:6]}"
        run_args = ["-d", "--rm", "--name", self._cname, "-P"]

        if self._scen_cfg.use_registry_mirror:
            await self._ensure_registry_mirror()
            run_args.extend(["--network", "privesc-net"])

        if self._scen_cfg.log_message:
            self._log(f"[dim]  {self._scen_cfg.log_message}[/]")
        if self._scen_cfg.pre_command:
            await self._run_host(self._scen_cfg.pre_command)
        if self._scen_cfg.extra_args:
            run_args.extend(self._scen_cfg.extra_args)

        command_in_container = self._scen_cfg.command
        if command_in_container and not self._scen_cfg.manual_sshd_start:
            command_in_container += " && exec /usr/sbin/sshd -D -e"

        run_args.append(resolve_image_ref(self._scen_cfg.image))
        if command_in_container:
            run_args.extend(["sh", "-c", command_in_container])

        self._log(f"[dim]  Launching container {self._cname}[/]")
        try:
            await self._run_host(f"docker run {shlex.join(run_args)}", timeout=300)
            self._runtime_host_key = self._host_ssh_key
            if self._runtime_host_key is not None:
                self._log(
                    f"[dim]  Container {self._cname} pinned to SSH host "
                    f"{self._runtime_host_key.host}:{self._runtime_host_key.port}[/]"
                )
        except asyncssh.ProcessError as proc_err:
            stderr = str(proc_err.stderr or "").lower()
            is_config_error = any(
                marker in stderr for marker in _DOCKER_CONFIG_ERROR_MARKERS
            )
            if proc_err.exit_status == 125 and not is_config_error:
                await self._mark_current_host_unhealthy("docker run exited with 125")
            raise

    @retry_async(
        attempts=MAX_CONNECTION_RETRIES,
        exceptions=(asyncssh.ProcessError, asyncssh.Error, OSError, ValueError),
        base_delay=0.5,
        factor=2.0,
    )
    async def _get_container_port(self) -> int:
        if self._cname is None:
            raise RuntimeError("Container not started")
        result = await self._run_host(f"docker port {self._cname} 22/tcp")
        if not result.stdout:
            raise ValueError("docker port command returned no output")
        return int(str(result.stdout).strip().rsplit(":", 1)[-1])

    async def _forward_port(self, rport: int) -> None:
        last_error: Exception | None = None
        for attempt in range(1, MAX_CONNECTION_RETRIES + 1):
            try:
                if self._host_ssh is None:
                    await self._acquire_host_connection()
                if self._host_ssh is None:
                    raise RuntimeError("Host SSH connection not established")

                self._local_port = free_port()
                self._fwd_server = await self._host_ssh.forward_local_port(
                    "127.0.0.1", self._local_port, "127.0.0.1", rport
                )
                return
            except TRANSIENT_CONN_ERRORS as conn_err:
                last_error = conn_err
                if self._runtime_host_key is not None:
                    await self._invalidate_current_host_connection()
                elif self._host_ssh_key is not None:
                    await self._mark_current_host_unhealthy(str(conn_err))
                if attempt >= MAX_CONNECTION_RETRIES:
                    break

        raise RuntimeError(
            f"Host port-forward error after {MAX_CONNECTION_RETRIES} attempts: {last_error}"
        )

    async def _close_forward_server(self) -> None:
        if self._fwd_server is None:
            return
        self._fwd_server.close()
        self._fwd_server = None

    async def _refresh_container_route(self) -> None:
        await self._close_forward_server()
        self._local_port = None
        await self._forward_port(await self._get_container_port())

    @retry_async(
        attempts=MAX_CONNECTION_RETRIES,
        exceptions=(asyncssh.Error, OSError),
        base_delay=0.5,
        factor=2.0,
    )
    async def _connect_to_container(self) -> None:
        if self._local_port is None:
            raise RuntimeError("Container SSH port forward not available")
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
            await self._refresh_container_route()
            await self._connect_to_container()

    async def _run_setup_script(self) -> None:
        if not self._cname:
            raise RuntimeError("Container not started yet")
        inner = build_setup_script(
            container_user=self._scen_cfg.container_user,
            container_password=self._scen_cfg.container_password,
            setup_script=self._scen_cfg.setup_script or "",
        )
        await self._run_host(
            f"docker exec -i {self._cname} /bin/sh -c {shlex.quote(inner)}"
        )

    async def _install_root_proof(self) -> None:
        if not self._cname:
            raise RuntimeError("Container not started yet")
        if self._root_proof is None:
            raise RuntimeError("Root proof not initialized")
        inner = root_proof_install_script(self._root_proof)
        await self._run_host(
            f"docker exec -i {self._cname} /bin/sh",
            input_data=inner,
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
                if attempt >= MAX_CONNECTION_RETRIES:
                    break
                try:
                    await self._reconnect_container()
                except Exception as reconnect_err:
                    last_error = reconnect_err

        if last_error is not None:
            raise last_error
        return AuthOutcome(success=False, message="Container connection not available")

    async def _cleanup_runtime(self) -> None:
        self._root_proof = None
        if self._cont_ssh:
            with contextlib.suppress(Exception):
                self._cont_ssh.close()
                await self._cont_ssh.wait_closed()
            self._cont_ssh = None
        if self._fwd_server:
            await self._close_forward_server()
        if self._cname:
            self._log(f"[dim]  Removing container {self._cname}[/]")
            with contextlib.suppress(Exception):
                await self._run_host(f"docker rm -f {self._cname}")
            self._cname = None
            self._runtime_host_key = None
            self._local_port = None
