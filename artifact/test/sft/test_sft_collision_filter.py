"""Unit tests for SFT split collision filtering."""

import json
from pathlib import Path
from typing import cast

import pytest

from src.config import AppConfig
from src.dataset.privesc.sft import DatasetAssembler, QualityConfig, TraceQualityFilter
from src.paths import TRACE_COLLECTION_BASE


def _long_text() -> str:
    return (
        "Plan: I will enumerate the system, validate findings with commands, and escalate privileges safely. "
        "I will check identity, sudo permissions, SUID/SGID binaries, writable paths, cron jobs, and file capabilities."
    )


def _trace(*, generator: str, seed: int, item_run_ordinal: int | None = None) -> dict:
    metadata = {
        "benchmark_eligible": True,
        "seed": seed,
        "generator_name": generator,
        "prompt_vars": {
            "user": "u",
            "password": "p",
            "max_turns": 5,
            "term_cols": 80,
            "term_rows": 24,
        },
    }
    if item_run_ordinal is not None:
        metadata["item_run_ordinal"] = item_run_ordinal
    return {
        "scenario": generator,
        "mode": "trace_collection",
        "status": "completed",
        "success": True,
        "turns": 2,
        "total_tokens": 500,
        "history": [
            {"role": "system", "content": "HEADER\nSECRET SOLUTION DATA\nFOOTER"},
            {"role": "user", "content": "Start"},
            {
                "role": "assistant",
                "content": _long_text(),
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"command": "id"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "exec_command",
                "tool_call_id": "call_1",
                "content": '{"command": "id", "output": "uid=0(root)", "got_root": true, "exit_code": 0}',
            },
        ],
        "metadata": metadata,
    }


def _block_content_trace(*, generator: str, seed: int) -> dict:
    payload = _trace(generator=generator, seed=seed)
    payload["history"][2]["content"] = [
        {
            "type": "reasoning",
            "content": [
                {
                    "type": "reasoning_text",
                    "text": _long_text(),
                }
            ],
        },
        {
            "type": "text",
            "text": "Visible summary that restates the plan and should not be duplicated.",
        },
    ]
    return payload


def _bad_trace(*, generator: str, seed: int) -> dict:
    payload = _trace(generator=generator, seed=seed)
    payload["history"][2]["content"] = "short"
    return payload


def _audit_rejected_trace(*, generator: str, seed: int) -> dict:
    payload = _trace(generator=generator, seed=seed)
    command = "docker run --rm -it -v /:/mnt alpine chroot /mnt /bin/sh"
    payload["history"][2]["tool_calls"] = [
        {
            "type": "function",
            "id": "call_1",
            "function": {
                "name": "exec_command",
                "arguments": json.dumps({"command": command}),
            },
        }
    ]
    payload["history"][3] = {
        "role": "tool",
        "name": "exec_command",
        "tool_call_id": "call_1",
        "content": json.dumps(
            {
                "command": command,
                "output": "#",
                "got_root": True,
                "exit_code": 124,
            }
        ),
    }
    return payload


def _credential_holdout_trace(*, generator: str, seed: int) -> dict:
    payload = _trace(generator=generator, seed=seed)
    payload["history"][2]["tool_calls"] = [
        {
            "type": "function",
            "id": "call_1",
            "function": {
                "name": "test_credentials",
                "arguments": '{"user": "root", "password": "trustno1"}',
            },
        }
    ]
    payload["history"][3] = {
        "role": "tool",
        "name": "test_credentials",
        "tool_call_id": "call_1",
        "content": (
            '{"user": "root", "password": "trustno1", '
            '"got_root": true, "success": true}'
        ),
    }
    return payload


def _repairable_trace(*, generator: str, seed: int) -> dict:
    payload = _trace(generator=generator, seed=seed)
    payload["history"][2]["content"] = ""
    payload["history"][2]["tool_calls"] = [
        {
            "type": "function",
            "id": "call_1",
            "function": {
                "name": "exec_command",
                "arguments": '{"command": "id"}',
            },
        }
    ]
    return payload


def _non_trace_collection(*, generator: str, seed: int) -> dict:
    payload = _trace(generator=generator, seed=seed)
    payload["mode"] = "evaluation"
    return payload


def _write_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_collection_summary(
    tmp_path: Path,
    *,
    split: str,
    generators: list[str],
    target_per_generator: int,
) -> None:
    path = (
        tmp_path / TRACE_COLLECTION_BASE / split / "stats" / "collection_summary.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "configured_generators": generators,
                "target_per_generator": target_per_generator,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _traces_dir(tmp_path: Path, split: str, model: str) -> Path:
    """Create and return the canonical trace directory for a split + model."""
    d = tmp_path / TRACE_COLLECTION_BASE / split / "traces" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_validation_rejects_generator_seed_overlap(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"
    seed = 123

    train_dir = _traces_dir(tmp_path, "training", model)
    val_dir = _traces_dir(tmp_path, "validation", model)

    _write_trace(train_dir / f"{gen}_a.json", _trace(generator=gen, seed=seed))
    _write_trace(val_dir / f"{gen}_b.json", _trace(generator=gen, seed=seed))

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(
        model,
        base_dir=tmp_path,
        prune_rejected=False,
        source_dir="validation",
        exclude_source_dir="training",
    )
    assert examples == []
    assert assembler.stats["filtered"].get("seed_collision_with_excluded_split") == 1


def test_validation_allows_non_overlapping_seed(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"

    train_dir = _traces_dir(tmp_path, "training", model)
    val_dir = _traces_dir(tmp_path, "validation", model)

    _write_trace(train_dir / f"{gen}_a.json", _trace(generator=gen, seed=1))
    _write_trace(val_dir / f"{gen}_b.json", _trace(generator=gen, seed=2))

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(
        model,
        base_dir=tmp_path,
        prune_rejected=False,
        source_dir="validation",
        exclude_source_dir="training",
    )
    assert len(examples) == 1


def test_assembler_records_sft_token_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "m"
    gen = "suid_gtfobins"
    trace = _trace(generator=gen, seed=1)
    trace["total_tokens"] = 10
    _write_trace(_traces_dir(tmp_path, "training", model) / f"{gen}.json", trace)
    monkeypatch.setattr(
        "src.dataset.privesc.sft_preprocessing.get_num_tokens_from_messages",
        lambda messages, tools=None: 123,
    )

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(model, base_dir=tmp_path, source_dir="training")

    assert len(examples) == 1
    assert examples[0]["num_tokens"] == 123


def test_validation_collision_fails_in_strict_mode(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"
    seed = 123

    train_dir = _traces_dir(tmp_path, "training", model)
    val_dir = _traces_dir(tmp_path, "validation", model)

    _write_trace(train_dir / f"{gen}_a.json", _trace(generator=gen, seed=seed))
    _write_trace(val_dir / f"{gen}_b.json", _trace(generator=gen, seed=seed))

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)

    with pytest.raises(ValueError, match="Detected split collisions"):
        assembler.load_traces(
            model,
            base_dir=tmp_path,
            prune_rejected=False,
            source_dir="validation",
            exclude_source_dir="training",
            fail_on_split_collision=True,
        )


def test_prune_caps_passing_traces_per_generator(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"

    val_dir = _traces_dir(tmp_path, "validation", model)

    first = val_dir / f"{gen}_1.json"
    second = val_dir / f"{gen}_2.json"
    _write_trace(first, _trace(generator=gen, seed=1))
    _write_trace(second, _trace(generator=gen, seed=2))

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(
        model,
        base_dir=tmp_path,
        prune_rejected=True,
        source_dir="validation",
        exclude_source_dir=None,
        max_per_generator=1,
    )
    assert len(examples) == 1
    assert first.exists()
    assert not second.exists()
    assert assembler.stats["filtered"].get("extra_trace_over_cap") == 1


def test_strict_assembly_rejects_missing_trace_collection_ordinal(
    tmp_path: Path,
) -> None:
    model = "m"
    gen = "suid_gtfobins"
    _write_collection_summary(
        tmp_path,
        split="training",
        generators=[gen],
        target_per_generator=2,
    )
    train_dir = _traces_dir(tmp_path, "training", model)
    _write_trace(
        train_dir / f"{gen}_0.json",
        _trace(generator=gen, seed=1, item_run_ordinal=0),
    )

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    with pytest.raises(ValueError, match=r"missing=\[\('suid_gtfobins', 1\)\]"):
        assembler.load_traces(model, base_dir=tmp_path, source_dir="training")


def test_strict_assembly_surfaces_duplicate_trace_collection_ordinal(
    tmp_path: Path,
) -> None:
    model = "m"
    gen = "suid_gtfobins"
    _write_collection_summary(
        tmp_path,
        split="training",
        generators=[gen],
        target_per_generator=1,
    )
    train_dir = _traces_dir(tmp_path, "training", model)
    _write_trace(
        train_dir / f"{gen}_first.json",
        _trace(generator=gen, seed=1, item_run_ordinal=0),
    )
    _write_trace(
        train_dir / f"{gen}_duplicate.json",
        _trace(generator=gen, seed=2, item_run_ordinal=0),
    )

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    with pytest.raises(ValueError, match=r"duplicates=\[\('suid_gtfobins', 0\)\]"):
        assembler.load_traces(model, base_dir=tmp_path, source_dir="training")


def test_prune_rejected_removes_holdout_leakage_trace(
    tmp_path: Path,
) -> None:
    model = "m"
    gen = "password_reuse"

    train_dir = _traces_dir(tmp_path, "training", model)
    adjacent_trace = train_dir / f"{gen}_bad.json"
    _write_trace(adjacent_trace, _credential_holdout_trace(generator=gen, seed=13))

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(
        model,
        base_dir=tmp_path,
        prune_rejected=True,
        source_dir="training",
    )

    assert examples == []
    assert not adjacent_trace.exists()
    assert (
        assembler.stats["filtered"].get("collection_audit:exact_benchmark_credential")
        == 1
    )


def test_assembly_rejects_collection_audit_failures(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"

    train_dir = _traces_dir(tmp_path, "training", model)
    _write_trace(
        train_dir / f"{gen}_audit_rejected.json",
        _audit_rejected_trace(generator=gen, seed=14),
    )

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(model, base_dir=tmp_path, source_dir="training")

    assert examples == []
    assert (
        assembler.stats["filtered"].get(
            "collection_audit:exact_benchmark_exploit_fragment"
        )
        == 1
    )


def test_filters_duplicate_seed_within_split(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"

    train_dir = _traces_dir(tmp_path, "training", model)
    _write_trace(train_dir / f"{gen}_first.json", _trace(generator=gen, seed=7))
    _write_trace(train_dir / f"{gen}_dupe.json", _trace(generator=gen, seed=7))

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(
        model,
        base_dir=tmp_path,
        prune_rejected=False,
        source_dir="training",
    )

    assert len(examples) == 1
    assert assembler.stats["filtered"].get("duplicate_seed_within_split") == 1


def test_duplicate_key_after_failed_trace_is_not_dropped(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"

    train_dir = _traces_dir(tmp_path, "training", model)
    _write_trace(train_dir / f"{gen}_first_bad.json", _bad_trace(generator=gen, seed=9))
    _write_trace(train_dir / f"{gen}_second_good.json", _trace(generator=gen, seed=9))

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(
        model,
        base_dir=tmp_path,
        prune_rejected=False,
        source_dir="training",
    )

    assert len(examples) == 1
    assert assembler.stats["filtered"].get("duplicate_seed_within_split") is None


def test_duplicate_seed_filter_applies_only_to_trace_collection(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"

    train_dir = _traces_dir(tmp_path, "training", model)
    _write_trace(
        train_dir / f"{gen}_eval_1.json",
        _non_trace_collection(generator=gen, seed=11),
    )
    _write_trace(
        train_dir / f"{gen}_eval_2.json",
        _non_trace_collection(generator=gen, seed=11),
    )

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(
        model,
        base_dir=tmp_path,
        prune_rejected=False,
        source_dir="training",
    )

    assert len(examples) == 2
    assert assembler.stats["filtered"].get("duplicate_seed_within_split") is None


def test_load_traces_rejects_missing_reasoning_without_repair(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"
    train_dir = _traces_dir(tmp_path, "training", model)
    trace_path = train_dir / f"{gen}_missing_reasoning.json"
    _write_trace(trace_path, _repairable_trace(generator=gen, seed=12))

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(
        model,
        base_dir=tmp_path,
        prune_rejected=False,
        source_dir="training",
    )

    assert examples == []
    assert (
        json.loads(trace_path.read_text(encoding="utf-8"))["history"][2]["content"]
        == ""
    )
    assert assembler.stats["filtered"].get("too_many_empty_reasoning (1 > 0)") == 1


def test_load_traces_prefers_reasoning_blocks_for_sft_content(tmp_path: Path) -> None:
    model = "m"
    gen = "suid_gtfobins"
    train_dir = _traces_dir(tmp_path, "training", model)
    _write_trace(
        train_dir / f"{gen}_blocks.json",
        _block_content_trace(generator=gen, seed=13),
    )

    filt = TraceQualityFilter(QualityConfig())
    assembler = DatasetAssembler(cfg=cast(AppConfig, object()), quality_filter=filt)
    examples = assembler.load_traces(
        model,
        base_dir=tmp_path,
        prune_rejected=False,
        source_dir="training",
    )

    assert len(examples) == 1
    assert examples[0]["messages"][2]["content"] == _long_text()
