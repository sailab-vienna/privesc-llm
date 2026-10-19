import asyncio
from typing import cast

import asyncssh
import pytest

import src.gym.backends.host_ssh_pool as host_ssh_pool
from src.gym.backends.host_ssh_pool import HostSSHConnectionPool, HostSSHKey


class _FakeSSHConn:
    def __init__(self) -> None:
        self.close_calls = 0
        self.wait_closed_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    async def wait_closed(self) -> None:
        self.wait_closed_calls += 1


def _key(host: str = "host", shard: int = 0) -> HostSSHKey:
    return HostSSHKey(host=host, port=22, user="user", key_path="/tmp/key", shard=shard)


@pytest.mark.asyncio
async def test_acquire_reuses_and_release_closes_last_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = HostSSHConnectionPool()
    created: list[_FakeSSHConn] = []

    async def fake_connect(
        *args: object, **kwargs: object
    ) -> asyncssh.SSHClientConnection:
        conn = _FakeSSHConn()
        created.append(conn)
        return cast(asyncssh.SSHClientConnection, conn)

    monkeypatch.setattr(asyncssh, "connect", fake_connect)

    key = _key()
    first = cast(_FakeSSHConn, await pool.acquire(key))
    second = cast(_FakeSSHConn, await pool.acquire(key))

    assert first is second
    assert len(created) == 1

    await pool.release(key)
    assert first.close_calls == 0

    await pool.release(key)
    assert first.close_calls == 1
    assert first.wait_closed_calls == 1


@pytest.mark.asyncio
async def test_invalidate_forces_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = HostSSHConnectionPool()
    created: list[_FakeSSHConn] = []

    async def fake_connect(
        *args: object, **kwargs: object
    ) -> asyncssh.SSHClientConnection:
        conn = _FakeSSHConn()
        created.append(conn)
        return cast(asyncssh.SSHClientConnection, conn)

    monkeypatch.setattr(asyncssh, "connect", fake_connect)

    key = _key()
    first = cast(_FakeSSHConn, await pool.acquire(key))
    await pool.invalidate(key)

    second = cast(_FakeSSHConn, await pool.acquire(key))
    assert first is not second
    assert first.close_calls == 1
    assert len(created) == 2

    await pool.release(key)
    await pool.release(key)
    assert second.close_calls == 1


@pytest.mark.asyncio
async def test_parallel_acquire_connects_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = HostSSHConnectionPool()
    calls = 0

    async def fake_connect(
        *args: object, **kwargs: object
    ) -> asyncssh.SSHClientConnection:
        nonlocal calls
        calls += 1
        return cast(asyncssh.SSHClientConnection, _FakeSSHConn())

    monkeypatch.setattr(asyncssh, "connect", fake_connect)

    key = _key()
    one, two, three = await asyncio.gather(
        pool.acquire(key),
        pool.acquire(key),
        pool.acquire(key),
    )

    assert one is two is three
    assert calls == 1

    await pool.release(key)
    await pool.release(key)
    await pool.release(key)


@pytest.mark.asyncio
async def test_parallel_acquire_stops_waiters_after_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = HostSSHConnectionPool()
    calls = 0

    async def fake_connect(
        *args: object, **kwargs: object
    ) -> asyncssh.SSHClientConnection:
        nonlocal calls
        calls += 1
        raise OSError("connection refused")

    monkeypatch.setattr(asyncssh, "connect", fake_connect)

    results = await asyncio.gather(
        pool.acquire(_key()),
        pool.acquire(_key()),
        return_exceptions=True,
    )

    assert all(isinstance(result, OSError) for result in results)
    assert calls == host_ssh_pool.MAX_CONNECTION_RETRIES


@pytest.mark.asyncio
async def test_acquire_any_skips_unhealthy_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setenv("PRIVESC_HOST_SSH_UNHEALTHY_TTL", "60")
    monkeypatch.setattr(host_ssh_pool.time, "monotonic", lambda: now)
    pool = HostSSHConnectionPool()

    unhealthy_key = _key(host="unhealthy")
    healthy_key = _key(host="healthy")
    created: list[str] = []

    async def fake_connect(
        host: str, *args: object, **kwargs: object
    ) -> asyncssh.SSHClientConnection:
        created.append(host)
        return cast(asyncssh.SSHClientConnection, _FakeSSHConn())

    monkeypatch.setattr(asyncssh, "connect", fake_connect)

    await pool.mark_unhealthy(unhealthy_key)

    lease = await pool.acquire_any([unhealthy_key, healthy_key], 0)

    assert lease.key == healthy_key
    assert created == ["healthy"]
    await pool.release(healthy_key)


@pytest.mark.asyncio
async def test_acquire_any_quarantines_failed_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = HostSSHConnectionPool()
    failing_key = _key(host="failing")
    healthy_key = _key(host="healthy")

    async def fake_connect(
        host: str, *args: object, **kwargs: object
    ) -> asyncssh.SSHClientConnection:
        if host == "failing":
            raise OSError("connection refused")
        return cast(asyncssh.SSHClientConnection, _FakeSSHConn())

    monkeypatch.setattr(asyncssh, "connect", fake_connect)

    lease = await pool.acquire_any([failing_key, healthy_key], 0)

    assert lease.key == healthy_key
    await pool.release(healthy_key)


@pytest.mark.asyncio
async def test_unhealthy_host_recovers_after_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setenv("PRIVESC_HOST_SSH_UNHEALTHY_TTL", "1")
    monkeypatch.setattr(host_ssh_pool.time, "monotonic", lambda: now)
    pool = HostSSHConnectionPool()

    recovered_key = _key(host="recovered")
    fallback_key = _key(host="fallback")

    await pool.mark_unhealthy(recovered_key)
    created: list[str] = []

    async def fake_connect(
        host: str, *args: object, **kwargs: object
    ) -> asyncssh.SSHClientConnection:
        created.append(host)
        return cast(asyncssh.SSHClientConnection, _FakeSSHConn())

    monkeypatch.setattr(asyncssh, "connect", fake_connect)

    first = await pool.acquire_any([recovered_key, fallback_key], 0)
    assert first.key == fallback_key
    await pool.release(fallback_key)

    now = 102.0
    second = await pool.acquire_any([recovered_key, fallback_key], 0)
    assert second.key == recovered_key
    assert created == ["fallback", "recovered"]
    await pool.release(recovered_key)


@pytest.mark.asyncio
async def test_mark_unhealthy_closes_all_host_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = HostSSHConnectionPool()
    created: list[_FakeSSHConn] = []

    async def fake_connect(
        *args: object, **kwargs: object
    ) -> asyncssh.SSHClientConnection:
        conn = _FakeSSHConn()
        created.append(conn)
        return cast(asyncssh.SSHClientConnection, conn)

    monkeypatch.setattr(asyncssh, "connect", fake_connect)

    first_key = _key(host="host", shard=0)
    second_key = _key(host="host", shard=1)
    await pool.acquire(first_key)
    await pool.acquire(second_key)

    await pool.mark_unhealthy(first_key)

    assert [conn.close_calls for conn in created] == [1, 1]

    await pool.release(first_key)
    await pool.release(second_key)
