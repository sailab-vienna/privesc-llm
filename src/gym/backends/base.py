from __future__ import annotations

import asyncio
import asyncssh
import contextlib
import re
import secrets
import shlex
import socket
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from src.utils.root_detection import (
    ROOT_PROOF_PATH,
    marked_root_proof_matches,
    root_proof_matches,
)

TIMEOUT_EXIT_CODE = 124
CONNECTION_ERROR_EXIT_CODE = 255

SLOW_COMMANDS = ("docker", "find /", "ssh ")
SLEEP_RE = re.compile(r"sleep\s+(\d+)")

MAX_CONNECTION_RETRIES = 3
HOST_CMD_TIMEOUT = 120
KEEPALIVE_INTERVAL = 30
KEEPALIVE_COUNT_MAX = 3
ROOT_PROOF_PROBE_TIMEOUT = 1.0
TRANSIENT_CONN_ERRORS = (
    asyncssh.ConnectionLost,
    asyncssh.DisconnectError,
    asyncssh.ChannelOpenError,
    OSError,
)


@dataclass
class CommandOutcome:
    output: str
    exit_code: int
    timed_out: bool = False
    root_verified: bool = False


@dataclass
class AuthOutcome:
    success: bool
    message: str = ""
    uid: int | None = None
    root_verified: bool = False


class ReadyCommandError(RuntimeError):
    def __init__(self, message: str, *, exit_status: int | None = None) -> None:
        super().__init__(message)
        self.exit_status = exit_status


class ScenarioBackend(Protocol):
    async def start(self) -> None: ...

    async def reset(self) -> None: ...

    async def run_command(self, command: str, timeout: int) -> CommandOutcome: ...

    async def test_credentials(self, user: str, password: str) -> AuthOutcome: ...

    async def close(self) -> None: ...

    @property
    def local_port(self) -> int | None: ...


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def resolve_image_ref(image: str | None) -> str:
    value = image or ""
    if ":" in value.split("/")[-1]:
        return value
    return f"{value}:latest"


def build_setup_script(
    container_user: str,
    container_password: str,
    setup_script: str,
) -> str:
    password_reset = f"echo '{container_user}:{container_password}' | chpasswd"
    ensure_user = (
        f"id -u {container_user} >/dev/null 2>&1 || "
        f"useradd -m -s /bin/bash {container_user}"
    )
    return "\n".join(
        [
            "set -eu",
            ensure_user,
            password_reset,
            "cat <<'EOF' > /tmp/setup.sh",
            setup_script,
            "EOF",
            "chmod 700 /tmp/setup.sh",
            "/bin/sh /tmp/setup.sh",
            "rm -f /tmp/setup.sh",
        ]
    )


def new_root_proof() -> str:
    return f"privesc-root-proof-{secrets.token_urlsafe(32)}"


def root_proof_install_script(expected_proof: str) -> str:
    flag_path = shlex.quote(ROOT_PROOF_PATH)
    return "\n".join(
        [
            "set -eu",
            "umask 077",
            f"printf '%s' {shlex.quote(expected_proof)} > {flag_path}",
            f"chown root:root {flag_path}",
            f"chmod 0600 {flag_path}",
        ]
    )


def _output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


async def _probe_timed_out_process_for_root(
    process: Any,
    expected_proof: str | None,
) -> bool:
    if not expected_proof:
        return False

    nonce = uuid.uuid4().hex
    begin_marker = f"__PRIVESC_ROOT_PROOF_BEGIN_{nonce}__"
    end_marker = f"__PRIVESC_ROOT_PROOF_END_{nonce}__"
    probe = (
        "\n"
        f"printf '\\n{begin_marker}\\n'; "
        f"cat {shlex.quote(ROOT_PROOF_PATH)} 2>/dev/null; "
        f"printf '\\n{end_marker}\\n'\n"
    )

    try:
        process.stdin.write(probe)
        await process.stdin.drain()
    except (OSError, asyncssh.Error):
        return False

    deadline = asyncio.get_running_loop().time() + ROOT_PROOF_PROBE_TIMEOUT
    chunks: list[str] = []
    while True:
        try:
            stdout, _stderr = process.collect_output()
        except (OSError, asyncssh.Error):
            return False
        if stdout:
            chunks.append(_output_text(stdout))
            output = "".join(chunks)
            if marked_root_proof_matches(
                output,
                expected_proof,
                begin_marker,
                end_marker,
            ):
                return True
            if end_marker in output:
                return False
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.05)


async def _terminate_process(process: Any) -> None:
    with contextlib.suppress(Exception):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait_closed(), timeout=0.5)
        return
    except (OSError, asyncssh.Error, asyncio.TimeoutError):
        pass

    with contextlib.suppress(Exception):
        process.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait_closed(), timeout=0.5)
    with contextlib.suppress(Exception):
        process.close()


async def run_ssh_command_with_root_proof(
    conn: asyncssh.SSHClientConnection,
    command: str,
    timeout: int,
    *,
    term_cols: int,
    term_rows: int,
    expected_root_proof: str | None,
) -> CommandOutcome:
    process = await conn.create_process(
        command,
        term_type="xterm",
        term_size=(term_cols, term_rows),
        stderr=asyncssh.STDOUT,
    )
    try:
        completed = await process.wait(check=False, timeout=timeout)
        stdout = _output_text(completed.stdout)
        return CommandOutcome(
            output=stdout,
            exit_code=completed.returncode or 0,
            timed_out=False,
            root_verified=root_proof_matches(stdout, expected_root_proof),
        )
    except asyncssh.TimeoutError as timeout_err:
        stdout = _output_text(getattr(timeout_err, "stdout", None))
        try:
            root_verified = root_proof_matches(stdout, expected_root_proof)
            if not root_verified:
                root_verified = await _probe_timed_out_process_for_root(
                    process,
                    expected_root_proof,
                )
        finally:
            await _terminate_process(process)
        return CommandOutcome(
            output=stdout,
            exit_code=TIMEOUT_EXIT_CODE,
            timed_out=True,
            root_verified=root_verified,
        )


async def test_credentials_on_port(
    port: int | None,
    user: str,
    password: str,
    connect_timeout: int,
    expected_root_proof: str | None = None,
) -> AuthOutcome:
    if port is None:
        return AuthOutcome(success=False, message="Container connection not available")

    conn = None
    try:
        conn = await asyncssh.connect(
            "127.0.0.1",
            port=port,
            username=user,
            password=password,
            known_hosts=None,
            config=None,
            client_keys=None,
            agent_path=None,
            host_based_auth=False,
            public_key_auth=False,
            kbdint_auth=False,
            password_auth=True,
            preferred_auth="password",
            connect_timeout=connect_timeout,
        )
        uid = None
        result = await conn.run("id -u", check=False)
        if result.exit_status == 0:
            uid_text = str(result.stdout or "").strip()
            if uid_text.isdecimal():
                uid = int(uid_text)
        root_verified = False
        if expected_root_proof:
            proof_result = await conn.run(
                f"cat {shlex.quote(ROOT_PROOF_PATH)} 2>/dev/null",
                check=False,
                timeout=connect_timeout,
            )
            if proof_result.exit_status == 0:
                root_verified = root_proof_matches(
                    _output_text(proof_result.stdout),
                    expected_root_proof,
                )
        return AuthOutcome(
            success=True,
            message="Authentication successful",
            uid=uid,
            root_verified=root_verified,
        )
    except asyncssh.PermissionDenied:
        return AuthOutcome(success=False, message="Permission denied")
    finally:
        if conn is not None:
            conn.close()
            await conn.wait_closed()


def calculate_command_timeout(
    command: str,
    base_command_timeout: int,
    slow_command_timeout: int,
    max_command_timeout: int,
) -> int:
    cmd_lower = command.lower()
    base_timeout = (
        slow_command_timeout
        if any(slow_cmd in cmd_lower for slow_cmd in SLOW_COMMANDS)
        else base_command_timeout
    )
    total_sleep = sum(int(s) for s in SLEEP_RE.findall(command))
    return min(base_timeout + total_sleep, max_command_timeout)
