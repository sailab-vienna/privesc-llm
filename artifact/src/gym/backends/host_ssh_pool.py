from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import asyncssh

from src.gym.backends.base import (
    KEEPALIVE_COUNT_MAX,
    KEEPALIVE_INTERVAL,
    MAX_CONNECTION_RETRIES,
)
from src.utils.retry import retry_async


log = logging.getLogger(__name__)


class HostSSHUnavailableError(OSError):
    pass


@dataclass(frozen=True)
class HostSSHKey:
    host: str
    port: int
    user: str
    key_path: str
    shard: int = 0

    def host_identity(self) -> _HostIdentity:
        return _HostIdentity(
            host=self.host,
            port=self.port,
            user=self.user,
            key_path=self.key_path,
        )


@dataclass(frozen=True)
class _HostIdentity:
    host: str
    port: int
    user: str
    key_path: str


@dataclass
class _PoolEntry:
    connection: asyncssh.SSHClientConnection | None = None
    ref_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class HostSSHLease:
    key: HostSSHKey
    connection: asyncssh.SSHClientConnection


class HostSSHConnectionPool:
    def __init__(self) -> None:
        self._entries: dict[HostSSHKey, _PoolEntry] = {}
        self._unhealthy_until: dict[_HostIdentity, float] = {}
        self._unhealthy_ttl = max(
            1.0, float(os.getenv("PRIVESC_HOST_SSH_UNHEALTHY_TTL", "300"))
        )
        self._connect_timeout = max(
            1.0, float(os.getenv("PRIVESC_HOST_SSH_CONNECT_TIMEOUT", "15"))
        )
        self._pool_lock = asyncio.Lock()

    async def acquire_any(
        self, keys: list[HostSSHKey], preferred_index: int
    ) -> HostSSHLease:
        if not keys:
            raise ValueError("At least one SSH host key is required")

        last_error: Exception | None = None
        for key in await self._healthy_keys(keys, preferred_index):
            try:
                return HostSSHLease(key=key, connection=await self.acquire(key))
            except (asyncssh.Error, OSError) as conn_err:
                last_error = conn_err

        if last_error is not None:
            raise last_error
        raise HostSSHUnavailableError("No healthy SSH host endpoint available")

    async def acquire(self, key: HostSSHKey) -> asyncssh.SSHClientConnection:
        async with self._pool_lock:
            self._clear_recovered_hosts(time.monotonic())
            unhealthy_until = self._unhealthy_until.get(key.host_identity())
            if unhealthy_until is not None:
                raise HostSSHUnavailableError(
                    f"SSH host is unhealthy: {key.host}:{key.port}"
                )
            entry = self._entries.get(key)
            if entry is None:
                entry = _PoolEntry()
                self._entries[key] = entry
            entry.ref_count += 1

        try:
            return await self._ensure_connected(entry, key)
        except (asyncssh.Error, OSError) as conn_err:
            await self.mark_unhealthy(key, str(conn_err))
            await self.release(key)
            raise
        except Exception:
            await self.release(key)
            raise

    async def release(self, key: HostSSHKey) -> None:
        async with self._pool_lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.ref_count -= 1
            if entry.ref_count > 0:
                return
            self._entries.pop(key, None)

        await self._close_entry_connection(entry)

    async def invalidate(self, key: HostSSHKey) -> None:
        async with self._pool_lock:
            entry = self._entries.get(key)
        if entry is None:
            return
        await self._close_entry_connection(entry)

    async def mark_unhealthy(self, key: HostSSHKey, reason: str | None = None) -> None:
        await self._set_unhealthy(key, reason)
        identity = key.host_identity()
        async with self._pool_lock:
            entries = [
                entry
                for entry_key, entry in self._entries.items()
                if entry_key.host_identity() == identity
            ]

        await asyncio.gather(
            *(self._close_entry_connection(entry) for entry in entries)
        )

    async def _set_unhealthy(
        self, key: HostSSHKey, reason: str | None = None
    ) -> None:
        identity = key.host_identity()
        async with self._pool_lock:
            was_healthy = identity not in self._unhealthy_until
            self._unhealthy_until[identity] = time.monotonic() + self._unhealthy_ttl

        if was_healthy:
            msg = f"SSH host unhealthy for {self._unhealthy_ttl:.0f}s: {key.host}:{key.port}"
            if reason:
                msg = f'{msg} reason="{reason}"'
            log.warning(msg)

    async def _healthy_keys(
        self, keys: list[HostSSHKey], preferred_index: int
    ) -> list[HostSSHKey]:
        now = time.monotonic()
        start = preferred_index % len(keys)
        ordered = keys[start:] + keys[:start]
        async with self._pool_lock:
            self._clear_recovered_hosts(now)
            return [
                key
                for key in ordered
                if key.host_identity() not in self._unhealthy_until
            ]

    async def _ensure_connected(
        self,
        entry: _PoolEntry,
        key: HostSSHKey,
    ) -> asyncssh.SSHClientConnection:
        async with entry.lock:
            async with self._pool_lock:
                self._clear_recovered_hosts(time.monotonic())
                if key.host_identity() in self._unhealthy_until:
                    raise HostSSHUnavailableError(
                        f"SSH host is unhealthy: {key.host}:{key.port}"
                    )

            conn = entry.connection
            if conn is not None:
                return conn

            try:
                conn = await self._open_connection(key)
            except (asyncssh.Error, OSError) as conn_err:
                await self._set_unhealthy(key, str(conn_err))
                raise
            entry.connection = conn
            return conn

    @retry_async(
        attempts=MAX_CONNECTION_RETRIES,
        exceptions=(asyncssh.Error, OSError),
        base_delay=0.5,
        factor=2.0,
    )
    async def _open_connection(self, key: HostSSHKey) -> asyncssh.SSHClientConnection:
        return await asyncssh.connect(
            key.host,
            port=key.port,
            username=key.user,
            client_keys=[key.key_path],
            known_hosts=None,
            config=None,
            connect_timeout=self._connect_timeout,
            keepalive_interval=KEEPALIVE_INTERVAL,
            keepalive_count_max=KEEPALIVE_COUNT_MAX,
        )

    def _clear_recovered_hosts(self, now: float) -> None:
        recovered = [
            identity
            for identity, unhealthy_until in self._unhealthy_until.items()
            if unhealthy_until <= now
        ]
        for identity in recovered:
            self._unhealthy_until.pop(identity, None)
            log.info("SSH host cooldown expired: %s:%s", identity.host, identity.port)

    async def _close_entry_connection(self, entry: _PoolEntry) -> None:
        conn = None
        async with entry.lock:
            conn = entry.connection
            entry.connection = None
        if conn is None:
            return
        conn.close()
        await conn.wait_closed()


HOST_SSH_POOL = HostSSHConnectionPool()
