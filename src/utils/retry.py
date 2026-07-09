import asyncio
import logging
import random
from typing import Callable, Tuple, Type

import asyncssh

log = logging.getLogger("privesc_retry")


def retry_async(
    attempts: int = 3,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    base_delay: float = 0.5,
    factor: float = 2.0,
    jitter: float = 0.2,
):
    """Async retry with exponential backoff and optional jitter."""

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == attempts:
                        log.error("Retry: Exhausted attempts for %s", func.__name__)
                        raise
                    sleep_delay = delay
                    if jitter > 0 and sleep_delay > 0:
                        span_low = 1 - jitter
                        span_high = 1 + jitter
                        sleep_delay = sleep_delay * random.uniform(span_low, span_high)
                    if isinstance(e, asyncssh.ProcessError):
                        log.warning(
                            (
                                "Retry %s %d/%d failed:\n"
                                "  cmd: %s\n"
                                "  stderr: %s\n"
                                "  exit: %d\n"
                                "  sleeping %.3fs"
                            ),
                            func.__name__,
                            attempt,
                            attempts,
                            e.command,
                            e.stderr,
                            e.exit_status,
                            sleep_delay,
                        )
                    else:
                        log.warning(
                            "Retry %s %d/%d failed (%s). Sleeping %.3fs",
                            func.__name__,
                            attempt,
                            attempts,
                            e,
                            sleep_delay,
                        )
                    await asyncio.sleep(max(0.001, sleep_delay))
                    delay *= factor

            # should not reach here
            if last_exc:
                raise last_exc

        return wrapper

    return decorator
