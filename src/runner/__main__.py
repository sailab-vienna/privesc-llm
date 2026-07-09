"""CLI entry point for runner."""

import asyncio
import logging

import hydra

from src.config import AppConfig, register_with_hydra
from src.paths import model_dir_name
from src.tui import console

from .single import run_single
from .multi import run_multi


async def run(cfg: AppConfig) -> None:
    workers = int(cfg.runner.workers)
    if workers > 1:
        await run_multi(cfg)
    else:
        await run_single(cfg)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: AppConfig) -> None:
    from rich.logging import RichHandler

    logging.basicConfig(
        level=cfg.log_level,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            RichHandler(
                console=console, rich_tracebacks=True, show_time=True, show_path=False
            )
        ],
        force=True,
    )

    if cfg.log_level == "DEBUG":
        logging.getLogger("privesc_agent").setLevel(logging.DEBUG)
        logging.getLogger("privesc_tools").setLevel(logging.DEBUG)
        logging.getLogger("privesc_scenario").setLevel(logging.DEBUG)

    for lib in ["asyncssh", "httpcore", "httpx", "openai"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    model_short = model_dir_name(cfg.agent.model)
    cap = cfg.runner.max_runs
    cap_text = f", cap={cap}" if cap is not None else ""
    console.print(
        "[bold]Starting runner[/] "
        f"[dim]model={model_short}, runs_per_item={cfg.runner.runs_per_item}{cap_text}[/]"
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    register_with_hydra()
    main()
