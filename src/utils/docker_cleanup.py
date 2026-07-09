"""Background Docker cleanup for long-running training jobs."""

import asyncio
import logging

import asyncssh

from src.gym.backends.host_ssh_pool import (
    HostSSHConnectionPool,
    HostSSHKey,
    HostSSHUnavailableError,
)

log = logging.getLogger("docker_cleanup")


def cleanup_command(filter_expr: str, max_age_seconds: int) -> str:
    """Generate command to remove containers older than max_age_seconds.

    Uses docker inspect to get creation time since 'until' filter isn't universally supported.
    The generated script listing containers, checking their age, and removing expired ones.
    """
    # We use a readable multi-line shell script for clarity.
    return f"""
    for id in $(docker ps -aq --filter "{filter_expr}"); do
        created_ts=$(docker inspect -f '{{{{.Created}}}}' "$id" 2>/dev/null | cut -d. -f1)
        
        if [ -n "$created_ts" ]; then
            created_epoch=$(date -d "$created_ts" +%s 2>/dev/null || echo 0)
            current_epoch=$(date +%s)
            age=$((current_epoch - created_epoch))
            
            if [ "$age" -gt {max_age_seconds} ]; then
                docker rm -f "$id" 2>/dev/null && echo "$id"
            fi
        fi
    done
    """


def count_lines(output: str | bytes | None) -> int:
    """Count non-empty lines in command output."""
    if output is None:
        return 0
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="ignore")
    return len(output.strip().split()) if output.strip() else 0


async def run_periodic_cleanup(
    host: str,
    port: int,
    user: str,
    key_path: str,
    pool: HostSSHConnectionPool | None = None,
    interval_minutes: int = 5,
    max_age_minutes: int = 30,
) -> None:
    """Background task that cleans up stale containers every interval.

    Only removes containers older than max_age_minutes to avoid killing active rollouts.
    """
    log.info(
        "🧹 Starting cleanup task (every %dm, max age %dm)",
        interval_minutes,
        max_age_minutes,
    )
    max_age_seconds = max_age_minutes * 60
    cleanup_pool = pool or HostSSHConnectionPool()
    key = HostSSHKey(host=host, port=port, user=user, key_path=key_path, shard=-1)

    while True:
        await asyncio.sleep(interval_minutes * 60)
        conn: asyncssh.SSHClientConnection | None = None
        try:
            conn = await cleanup_pool.acquire(key)
            alpine_result = await conn.run(
                cleanup_command("ancestor=alpine", max_age_seconds), check=False
            )
            alpine_removed = count_lines(alpine_result.stdout)

            privesc_result = await conn.run(
                cleanup_command("name=privesc_", max_age_seconds), check=False
            )
            privesc_removed = count_lines(privesc_result.stdout)

            if alpine_removed or privesc_removed:
                log.info(
                    "🧹 Removed %d alpine + %d privesc containers on %s:%d (age > %dm)",
                    alpine_removed,
                    privesc_removed,
                    host,
                    port,
                    max_age_minutes,
                )
            else:
                log.debug("No stale containers to remove on %s:%d", host, port)

            await conn.run("docker container prune -f", check=False)
        except HostSSHUnavailableError as e:
            log.info("Cleanup skipped unhealthy SSH host %s:%d: %s", host, port, e)
        except (asyncssh.Error, OSError) as e:
            await cleanup_pool.mark_unhealthy(key, str(e))
            log.warning("Cleanup failed on %s:%d: %s", host, port, e)
        except Exception as e:
            log.warning("Cleanup failed on %s:%d: %s", host, port, e)
        finally:
            if conn is not None:
                await cleanup_pool.release(key)
