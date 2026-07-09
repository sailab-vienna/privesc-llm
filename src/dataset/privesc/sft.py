#!/usr/bin/env python3
"""Assemble SFT traces into a filtered dataset."""

import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from rich import get_console
from rich.logging import RichHandler

from src.config import AppConfig, register_with_hydra, sft_teacher_model
from src.dataset.privesc.sft_assembly import DatasetAssembler
from src.dataset.privesc.sft_quality import QualityConfig, TraceQualityFilter
from src.dataset.privesc.sft_reporting import print_filter_stats, save_outputs
from src.paths import resolve_traces_dir

console = get_console()


def _configure_logging() -> None:
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        omit_repeated_times=False,
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        if isinstance(existing, logging.StreamHandler) and not isinstance(
            existing, logging.FileHandler
        ):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

__all__ = [
    "QualityConfig",
    "TraceQualityFilter",
    "DatasetAssembler",
    "run_pipeline",
]


def _quality_filter(cfg: AppConfig) -> TraceQualityFilter:
    quality_dict = OmegaConf.to_container(cfg.datasets.sft.quality, resolve=True)
    if not isinstance(quality_dict, dict):
        raise ValueError("datasets.sft.quality must be a mapping")
    return TraceQualityFilter(QualityConfig(**{str(k): v for k, v in quality_dict.items()}))


def run_pipeline(cfg: AppConfig) -> None:
    _configure_logging()
    teacher_model = sft_teacher_model(cfg)
    console.print(
        f"[bold green]Starting SFT Dataset Assembly[/] - Teacher: {teacher_model}"
    )

    # Step 1: build the pipeline components.
    orig_cwd = Path(get_original_cwd())
    quality_filter = _quality_filter(cfg)
    assembler = DatasetAssembler(cfg, quality_filter)
    prune_rejected = bool(getattr(cfg.datasets.sft, "prune", False))
    trace_root = getattr(cfg.datasets.sft, "trace_root", None)
    source_dir = str(getattr(cfg.datasets.sft, "source_dir", "training"))
    exclude_source_dir = getattr(cfg.datasets.sft, "exclude_source_dir", None)
    max_per_generator = getattr(cfg.datasets.sft, "max_per_generator", None)
    fail_on_split_collision = bool(
        getattr(cfg.datasets.sft, "fail_on_split_collision", False)
    )

    # Step 2: show which raw traces will be processed.
    if prune_rejected:
        console.print(
            "[bold red]PRUNE MODE ENABLED[/]: Rejected traces will be deleted from disk!"
        )
    console.print(f"[bold cyan]Source:[/] {source_dir}")
    if exclude_source_dir:
        console.print(f"[bold cyan]Exclude split:[/] {exclude_source_dir}")
    if fail_on_split_collision:
        console.print("[bold cyan]Split collision policy:[/] fail")
    if max_per_generator is not None:
        console.print(f"[bold cyan]Cap per generator:[/] {max_per_generator}")
    traces_dir = resolve_traces_dir(
        orig_cwd, teacher_model, source_dir, trace_root=trace_root
    )
    console.print(f"[dim]Traces dir:[/] {traces_dir}")
    console.print(f"[dim]Dataset dir:[/] {cfg.datasets.sft.output_dir}")

    # Step 3: load raw traces, normalize prompts, apply quality/audit filters,
    # prune/select duplicates and caps, and convert passing traces into examples.
    examples = assembler.load_traces(
        teacher_model,
        base_dir=orig_cwd,
        prune_rejected=prune_rejected,
        source_dir=source_dir,
        exclude_source_dir=exclude_source_dir,
        trace_root=trace_root,
        max_per_generator=max_per_generator,
        fail_on_split_collision=fail_on_split_collision,
    )

    # Step 4: stop early if nothing passed filtering.
    if not examples:
        console.print("[bold red]No passing traces found![/]")
        print_filter_stats(assembler.stats["filtered"])
        return

    # Step 5: save the assembled dataset, stats, and tool definitions.
    console.print(f"[bold blue]Successfully loaded {len(examples)} traces.[/]")
    save_outputs(
        cfg,
        assembler.preprocessor,
        examples,
        output_dir=cfg.datasets.sft.output_dir,
        filtered_stats=assembler.stats["filtered"],
    )
    console.print("[bold green]Assembly Complete![/]")


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: AppConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    register_with_hydra()
    main()
