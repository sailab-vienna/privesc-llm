"""Test for Docker cleanup SSH connection."""

import asyncio
import os

import asyncssh
import pytest

from src.config import SSHConfig, resolve_ssh_endpoints
from src.gym.backends.host_ssh_pool import HostSSHConnectionPool, HostSSHKey
from src.utils.docker_cleanup import cleanup_command, count_lines


@pytest.mark.asyncio
async def test_cleanup_pool_disables_ssh_config_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    pool = HostSSHConnectionPool()
    key = HostSSHKey(host="host", port=22, user="root", key_path="/tmp/key", shard=-1)

    class _Conn:
        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    async def _fake_connect(*_args, **kwargs):
        captured.update(kwargs)
        return _Conn()

    monkeypatch.setattr("src.gym.backends.host_ssh_pool.asyncssh.connect", _fake_connect)

    conn = await pool.acquire(key)

    assert conn is not None
    assert captured["config"] is None
    assert captured["client_keys"] == ["/tmp/key"]
    await pool.release(key)


async def acquire_cleanup_connection(
    host: str, port: int, user: str, key_path: str
) -> tuple[HostSSHConnectionPool, HostSSHKey, asyncssh.SSHClientConnection]:
    pool = HostSSHConnectionPool()
    key = HostSSHKey(host=host, port=port, user=user, key_path=key_path, shard=-1)
    conn = await pool.acquire(key)
    return pool, key, conn


def get_ssh_config_from_env() -> tuple[str, int, str, str]:
    """Get SSH config from environment variables."""
    user = os.getenv("PRIVESC_USER", "root")
    key_path = os.getenv("PRIVESC_KEY", "")
    if not key_path:
        raise ValueError("Set PRIVESC_KEY environment variable.")
    endpoints = resolve_ssh_endpoints(
        SSHConfig(
            user=user,
            key_path=key_path,
            servers=os.getenv("PRIVESC_SSH_SERVERS", ""),
        )
    )
    endpoint = endpoints[0]
    return endpoint.host, endpoint.port, user, key_path


def output_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="ignore")
    return output


def skip_if_not_configured():
    """Skip test if SSH environment not configured."""
    try:
        return get_ssh_config_from_env()
    except ValueError as e:
        pytest.skip(f"Environment not configured: {e}")


@pytest.mark.asyncio
async def test_cleanup_ssh_connection():
    """Test that the cleanup SSH connection works."""
    host, port, user, key_path = skip_if_not_configured()

    pool, key, conn = await acquire_cleanup_connection(host, port, user, key_path)
    try:
        result = await conn.run("echo hello", check=True)
        assert output_text(result.stdout).strip() == "hello"
    finally:
        await pool.release(key)


@pytest.mark.asyncio
async def test_cleanup_docker_command():
    """Test that the cleanup Docker commands work (dry run - just list containers)."""
    host, port, user, key_path = skip_if_not_configured()

    pool, key, conn = await acquire_cleanup_connection(host, port, user, key_path)
    try:
        result = await conn.run(
            "docker ps -aq --filter 'name=privesc_' | wc -l",
            check=False,
        )
        count = int(output_text(result.stdout).strip())
        assert count >= 0
        print(f"Found {count} privesc containers on host")
    finally:
        await pool.release(key)


@pytest.mark.asyncio
async def test_cleanup_removes_old_containers():
    """Integration test: spawn containers (including privesc), wait briefly, verify cleanup removes them."""
    host, port, user, key_path = skip_if_not_configured()
    test_prefix = "test_cleanup_"
    privesc_prefix = "privesc_test_"

    pool, key, conn = await acquire_cleanup_connection(host, port, user, key_path)
    try:
        # Clean up any leftover test containers first
        await conn.run(
            f"docker ps -aq --filter 'name={test_prefix}' --filter 'name={privesc_prefix}' | xargs -r docker rm -f",
            check=False,
        )

        # Spawn 2 standard test containers
        await conn.run(
            f"docker run -d --name {test_prefix}1 alpine sleep 300", check=False
        )
        await conn.run(
            f"docker run -d --name {test_prefix}2 alpine sleep 300", check=False
        )
        # Spawn 1 privesc test container
        await conn.run(
            f"docker run -d --name {privesc_prefix}1 alpine sleep 300", check=False
        )

        # Verify they exist
        list_result = await conn.run(
            f"docker ps -aq --filter 'name={test_prefix}' --filter 'name={privesc_prefix}' | wc -l",
            check=False,
        )
        initial_count = int(list_result.stdout.strip()) if list_result.stdout else 0
        assert initial_count == 3, f"Expected 3 test containers, got {initial_count}"

        # Wait 3 seconds so containers are older than 2 seconds
        await asyncio.sleep(3)

        # Run cleanup for standard test containers
        cleanup_result = await conn.run(
            cleanup_command(f"name={test_prefix}", max_age_seconds=2), check=False
        )
        removed_std = count_lines(cleanup_result.stdout)

        # Run cleanup for privesc test containers
        cleanup_privesc_result = await conn.run(
            cleanup_command(f"name={privesc_prefix}", max_age_seconds=2), check=False
        )
        removed_privesc = count_lines(cleanup_privesc_result.stdout)

        print(
            f"Cleanup removed {removed_std} standard + {removed_privesc} privesc containers"
        )

        # Verify containers are gone
        final_result = await conn.run(
            f"docker ps -aq --filter 'name={test_prefix}' --filter 'name={privesc_prefix}' | wc -l",
            check=False,
        )
        final_count = int(final_result.stdout.strip()) if final_result.stdout else 0

        assert removed_std >= 2, (
            f"Expected to remove at least 2 standard containers, removed {removed_std}"
        )
        assert removed_privesc >= 1, (
            f"Expected to remove at least 1 privesc container, removed {removed_privesc}"
        )
        assert final_count == 0, f"Expected 0 remaining containers, got {final_count}"
    finally:
        await pool.release(key)
