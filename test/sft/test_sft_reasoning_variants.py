import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.dataset.privesc.sft_quality import QualityConfig, TraceQualityFilter
from src.dataset.privesc.sft_reasoning_variants import (
    SHORT_REASONING_REWRITE_MAX_RETRIES,
    SHORT_REASONING_PROMPT_TEMPLATE,
    SHORT_REASONING_STRUCTURED_OUTPUT_METHOD,
    TraceTransformContext,
    _short_reasoning_rewrite_payload,
    _short_reasoning_system_prompt,
    derive_no_reasoning_trace,
    derive_short_reasoning_trace,
    transform_reasoning_variant_split,
)


def _long_reasoning() -> str:
    return (
        "Plan: I will enumerate identity, sudo privileges, SUID binaries, writable paths, and cron jobs before escalating. "
        "I will ground every command in prior evidence and verify root access through an accepted root shell or login."
    )


def _tool_call(command: str = "id") -> dict:
    return {
        "type": "function",
        "id": "call_1",
        "function": {
            "name": "exec_command",
            "arguments": json.dumps({"command": command}),
        },
    }


def _trace(*, seed: int, hidden_reasoning: str | None = None) -> dict:
    reasoning = hidden_reasoning if hidden_reasoning is not None else _long_reasoning()
    return {
        "scenario": "suid_gtfobins",
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
                "content": [
                    {
                        "type": "reasoning",
                        "content": [
                            {
                                "type": "reasoning_text",
                                "text": reasoning,
                            }
                        ],
                    },
                    {"type": "text", "text": "Visible assistant text."},
                ],
                "reasoning_content": "provider hidden reasoning",
                "additional_kwargs": {
                    "reasoning_content": "additional hidden reasoning",
                    "visible_flag": True,
                },
                "tool_calls": [_tool_call()],
            },
            {
                "role": "tool",
                "name": "exec_command",
                "tool_call_id": "call_1",
                "content": json.dumps(
                    {
                        "command": "id",
                        "output": "uid=1000(lowpriv)",
                        "got_root": False,
                        "exit_code": 0,
                    }
                ),
            },
        ],
        "metadata": {
            "benchmark_eligible": True,
            "generator_name": "suid_gtfobins",
            "seed": seed,
            "prompt_vars": {
                "user": "u",
                "password": "p",
                "max_turns": 5,
                "term_cols": 80,
                "term_rows": 24,
            },
        },
    }


def _write_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class _FakeShortReasoningRewriter:
    def __init__(self, mapping: dict[int, str] | None = None) -> None:
        self.mapping = mapping or {
            2: "Check identity and sudo access, then run the matching enumeration command."
        }
        self.calls: list[Path] = []

    def rewrite_trace(self, trace_data: dict, *, source_file: Path) -> dict[int, str]:
        self.calls.append(source_file)
        return dict(self.mapping)


class _SequenceShortReasoningRewriter:
    def __init__(self, mappings: list[dict[int, str]]) -> None:
        self.mappings = mappings
        self.calls: list[Path] = []

    def rewrite_trace(self, trace_data: dict, *, source_file: Path) -> dict[int, str]:
        self.calls.append(source_file)
        index = min(len(self.calls), len(self.mappings)) - 1
        return dict(self.mappings[index])


def _transform_context(
    *,
    source_file: Path,
    source_root: Path,
    output_root: Path,
    variant: str,
    quality_metrics: dict,
    teacher_model: str = "deepseek/deepseek-v4-flash",
    forbidden_short_reasoning_phrases: tuple[str, ...] = (),
) -> TraceTransformContext:
    return TraceTransformContext(
        source_file=source_file,
        source_root=source_root,
        output_root=output_root,
        variant=variant,
        source_variant="long_reasoning",
        quality_metrics=quality_metrics,
        teacher_model=teacher_model,
        forbidden_short_reasoning_phrases=forbidden_short_reasoning_phrases,
    )


def test_remove_assistant_reasoning_preserves_visible_trace_fields(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "long_reasoning"
    output_root = tmp_path / "no_reasoning"
    source_file = source_root / "training" / "traces" / "m" / "trace.json"
    trace = _trace(seed=7)
    original = deepcopy(trace)

    result = derive_no_reasoning_trace(
        trace,
        context=_transform_context(
            source_file=source_file,
            source_root=source_root,
            output_root=output_root,
            variant="no_reasoning",
            quality_metrics={"passed": True},
        ),
    )
    transformed = result.trace

    assert trace == original
    assert [msg["role"] for msg in transformed["history"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assistant = transformed["history"][2]
    assert assistant["content"] == ""
    assert "reasoning_content" not in assistant
    assert assistant["additional_kwargs"] == {"visible_flag": True}
    assert assistant["tool_calls"] == original["history"][2]["tool_calls"]
    assert transformed["history"][3] == original["history"][3]
    assert transformed["success"] is True
    assert transformed["metadata"]["seed"] == 7
    assert transformed["metadata"]["generator_name"] == "suid_gtfobins"
    assert transformed["metadata"]["sft_reasoning_variant"] == {
        "variant": "no_reasoning",
        "source_variant": "long_reasoning",
        "transform": "remove_assistant_reasoning",
        "source_trace_root": source_root.as_posix(),
        "source_trace_path": "training/traces/m/trace.json",
        "output_trace_root": output_root.as_posix(),
        "quality_metrics": {"passed": True},
    }


def test_short_reasoning_uses_one_rewrite_and_preserves_trace_fields(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "long_reasoning"
    output_root = tmp_path / "short_reasoning"
    source_file = source_root / "training" / "traces" / "m" / "trace.json"
    trace = _trace(seed=8)
    original = deepcopy(trace)
    rewriter = _FakeShortReasoningRewriter(
        {
            2: "Enumerate user privileges and run id before choosing the next escalation path."
        }
    )

    result = derive_short_reasoning_trace(
        trace,
        context=_transform_context(
            source_file=source_file,
            source_root=source_root,
            output_root=output_root,
            variant="short_reasoning",
            quality_metrics={"passed": True},
        ),
        rewriter=rewriter,
    )
    transformed = result.trace

    assert trace == original
    assert rewriter.calls == [source_file]
    assert result.assistant_messages_rewritten == 1
    assert result.reasoning_chars_before > result.reasoning_chars_after
    assistant = transformed["history"][2]
    assert (
        assistant["content"]
        == "Enumerate user privileges and run id before choosing the next escalation path."
    )
    assert assistant["reasoning_content"] == assistant["content"]
    assert assistant["additional_kwargs"] == {
        "reasoning_content": assistant["content"],
        "visible_flag": True,
    }
    assert assistant["tool_calls"] == original["history"][2]["tool_calls"]
    assert transformed["history"][3] == original["history"][3]
    assert transformed["metadata"]["seed"] == 8
    assert transformed["metadata"]["sft_reasoning_variant"] == {
        "variant": "short_reasoning",
        "source_variant": "long_reasoning",
        "transform": "rewrite_short_reasoning",
        "source_trace_root": source_root.as_posix(),
        "source_trace_path": "training/traces/m/trace.json",
        "output_trace_root": output_root.as_posix(),
        "quality_metrics": {"passed": True},
        "rewrite_model": "deepseek/deepseek-v4-flash",
        "rewrite_prompt_template": SHORT_REASONING_PROMPT_TEMPLATE,
        "structured_output_method": SHORT_REASONING_STRUCTURED_OUTPUT_METHOD,
    }


def test_short_reasoning_retries_invalid_rewrite_before_success(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "long_reasoning"
    output_root = tmp_path / "short_reasoning"
    source_file = source_root / "training" / "traces" / "m" / "trace.json"
    trace = _trace(seed=13)
    valid_rewrite = "Enumerate privileges, identify cap_setuid, and choose a focused escalation command."
    rewriter = _SequenceShortReasoningRewriter(
        [
            {2: "not shorter " * 100},
            {2: valid_rewrite},
        ]
    )

    result = derive_short_reasoning_trace(
        trace,
        context=_transform_context(
            source_file=source_file,
            source_root=source_root,
            output_root=output_root,
            variant="short_reasoning",
            quality_metrics={"passed": True},
        ),
        rewriter=rewriter,
    )

    assert rewriter.calls == [source_file, source_file]
    assert result.llm_calls == 2
    assert result.trace["history"][2]["content"] == valid_rewrite


def test_short_reasoning_fails_fast_on_guided_leakage_phrase(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "long_reasoning"
    output_root = tmp_path / "short_reasoning"
    source_file = source_root / "training" / "traces" / "m" / "trace.json"
    rewriter = _FakeShortReasoningRewriter(
        {2: "The hint suggests checking sudo first."}
    )

    with pytest.raises(ValueError, match="short_reasoning_forbidden_phrase"):
        derive_short_reasoning_trace(
            _trace(seed=16),
            context=_transform_context(
                source_file=source_file,
                source_root=source_root,
                output_root=output_root,
                variant="short_reasoning",
                quality_metrics={"passed": True},
                forbidden_short_reasoning_phrases=("hint suggests", "as hinted"),
            ),
            rewriter=rewriter,
        )

    assert rewriter.calls == [source_file]


def test_short_reasoning_retries_then_rejects_long_rewrite(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "long_reasoning"
    output_root = tmp_path / "short_reasoning"
    source_file = source_root / "training" / "traces" / "m" / "trace.json"
    trace = _trace(seed=14)
    rewriter = _FakeShortReasoningRewriter({2: "not shorter " * 100})

    with pytest.raises(ValueError, match="short_reasoning_not_shorter"):
        derive_short_reasoning_trace(
            trace,
            context=_transform_context(
                source_file=source_file,
                source_root=source_root,
                output_root=output_root,
                variant="short_reasoning",
                quality_metrics={"passed": True},
            ),
            rewriter=rewriter,
        )

    assert len(rewriter.calls) == SHORT_REASONING_REWRITE_MAX_RETRIES + 1


def test_short_reasoning_missing_source_fails_without_retry(tmp_path: Path) -> None:
    source_root = tmp_path / "long_reasoning"
    output_root = tmp_path / "short_reasoning"
    source_file = source_root / "training" / "traces" / "m" / "trace.json"
    trace = _trace(seed=15, hidden_reasoning="")
    assistant = trace["history"][2]
    assistant["content"] = ""
    assistant.pop("reasoning_content")
    assistant["additional_kwargs"].pop("reasoning_content")
    rewriter = _FakeShortReasoningRewriter({2: "Short rewrite."})

    with pytest.raises(ValueError, match="short_reasoning_missing_source_reasoning"):
        derive_short_reasoning_trace(
            trace,
            context=_transform_context(
                source_file=source_file,
                source_root=source_root,
                output_root=output_root,
                variant="short_reasoning",
                quality_metrics={"passed": True},
            ),
            rewriter=rewriter,
        )

    assert rewriter.calls == []


def test_short_reasoning_prompt_loads_from_canonical_template() -> None:
    prompt = _short_reasoning_system_prompt()

    assert "Rewrite each `assistant_rewrites` item" in prompt
    assert "Use only that item's `source_reasoning` and `tool_calls`" in prompt
    assert "top-level `assistant_reasoning`" in prompt


def test_short_reasoning_payload_excludes_hidden_and_future_context() -> None:
    trace = _trace(seed=9)
    trace["history"].extend(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "content": [
                            {
                                "type": "reasoning_text",
                                "text": "Future reasoning should stay isolated.",
                            }
                        ],
                    },
                    {"type": "text", "text": "Future visible text."},
                ],
                "tool_calls": [_tool_call("whoami")],
            },
            {
                "role": "tool",
                "name": "exec_command",
                "tool_call_id": "future_call",
                "content": json.dumps({"output": "future root evidence"}),
            },
        ]
    )

    payload, assistant_indices = _short_reasoning_rewrite_payload(trace)

    assert assistant_indices == [2, 4]
    serialized = json.dumps(payload)
    assert "SECRET SOLUTION DATA" not in serialized
    assert "future root evidence" not in serialized
    assert "prior_messages" not in serialized
    assert "assistant_action" not in serialized
    assert "Future visible text" not in serialized
    assert '"role": "tool"' not in serialized

    first_item = payload["assistant_rewrites"][0]
    assert first_item["message_index"] == 2
    assert first_item["source_reasoning"] == (
        _long_reasoning()
        + "\n\nprovider hidden reasoning"
        + "\n\nadditional hidden reasoning"
    )
    assert first_item["tool_calls"] == trace["history"][2]["tool_calls"]

    second_item = payload["assistant_rewrites"][1]
    assert second_item["source_reasoning"] == "Future reasoning should stay isolated."
    assert second_item["tool_calls"] == trace["history"][4]["tool_calls"]


def test_transform_split_copies_only_quality_passing_traces(tmp_path: Path) -> None:
    source_root = tmp_path / "standard" / "guided" / "deepseek" / "long_reasoning"
    output_root = tmp_path / "standard" / "guided" / "deepseek" / "no_reasoning"
    traces_dir = source_root / "training" / "traces" / "m"
    good_path = traces_dir / "good.json"
    bad_path = traces_dir / "bad.json"
    good_trace = _trace(seed=1)
    bad_trace = _trace(seed=2, hidden_reasoning="short")
    _write_trace(good_path, good_trace)
    _write_trace(bad_path, bad_trace)

    stats = transform_reasoning_variant_split(
        source_trace_root=source_root,
        output_trace_root=output_root,
        teacher_model="m",
        source_dir="training",
        quality_filter=TraceQualityFilter(
            QualityConfig(reject_tool_call_free_turns=True)
        ),
    )

    assert stats.total == 2
    assert stats.passed_quality == 1
    assert stats.transformed == 1
    assert stats.filtered == {"too_many_empty_reasoning (1 > 0)": 1}
    assert json.loads(good_path.read_text(encoding="utf-8")) == good_trace
    assert json.loads(bad_path.read_text(encoding="utf-8")) == bad_trace

    output_good = output_root / "training" / "traces" / "m" / "good.json"
    output_bad = output_root / "training" / "traces" / "m" / "bad.json"
    assert output_good.exists()
    assert not output_bad.exists()

    transformed = json.loads(output_good.read_text(encoding="utf-8"))
    assert transformed["history"][2]["content"] == ""
    assert (
        transformed["history"][2]["tool_calls"]
        == good_trace["history"][2]["tool_calls"]
    )
    assert transformed["metadata"]["seed"] == 1
    assert (
        transformed["metadata"]["sft_reasoning_variant"]["quality_metrics"]["passed"]
        is True
    )

    stats_path = output_root / "training" / "stats" / "no_reasoning_transform.json"
    assert json.loads(stats_path.read_text(encoding="utf-8"))["transformed"] == 1


def test_transform_split_short_reasoning_calls_rewriter_once_per_passing_trace(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "standard" / "guided" / "deepseek" / "long_reasoning"
    output_root = tmp_path / "standard" / "guided" / "deepseek" / "short_reasoning"
    traces_dir = source_root / "validation" / "traces" / "m"
    good_path = traces_dir / "good.json"
    bad_path = traces_dir / "bad.json"
    good_trace = _trace(seed=11)
    bad_trace = _trace(seed=12, hidden_reasoning="short")
    _write_trace(good_path, good_trace)
    _write_trace(bad_path, bad_trace)
    rewriter = _FakeShortReasoningRewriter(
        {2: "Use the current evidence to choose the next privilege-escalation command."}
    )

    stats = transform_reasoning_variant_split(
        source_trace_root=source_root,
        output_trace_root=output_root,
        teacher_model="m",
        source_dir="validation",
        quality_filter=TraceQualityFilter(
            QualityConfig(reject_tool_call_free_turns=True)
        ),
        variant="short_reasoning",
        short_reasoning_rewriter=rewriter,
    )

    assert stats.total == 2
    assert stats.passed_quality == 1
    assert stats.transformed == 1
    assert stats.llm_calls == 1
    assert stats.assistant_messages_rewritten == 1
    assert rewriter.calls == [good_path]
    output_good = output_root / "validation" / "traces" / "m" / "good.json"
    output_bad = output_root / "validation" / "traces" / "m" / "bad.json"
    assert output_good.exists()
    assert not output_bad.exists()
    transformed = json.loads(output_good.read_text(encoding="utf-8"))
    assert (
        transformed["history"][2]["content"]
        == "Use the current evidence to choose the next privilege-escalation command."
    )
    assert (
        transformed["history"][2]["tool_calls"]
        == good_trace["history"][2]["tool_calls"]
    )
    stats_path = output_root / "validation" / "stats" / "short_reasoning_transform.json"
    assert json.loads(stats_path.read_text(encoding="utf-8"))["llm_calls"] == 1


def test_short_reasoning_real_trace_smoke_preserves_tool_outputs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "long_reasoning"
    output_root = tmp_path / "short_reasoning"
    source_file = source_root / "training" / "traces" / "m" / "trace_1.json"
    trace = json.loads(
        Path("test/trajectories/trace_1.json").read_text(encoding="utf-8")
    )
    trace["history"] = trace.pop("messages")
    trace["metadata"] = {"generator_name": "real_fixture", "seed": 1}
    original = deepcopy(trace)
    assistant_indices = [
        idx
        for idx, message in enumerate(trace["history"])
        if message["role"] == "assistant"
    ]
    rewriter = _FakeShortReasoningRewriter(
        {
            idx: f"Concise reasoning for assistant turn {idx}."
            for idx in assistant_indices
        }
    )

    result = derive_short_reasoning_trace(
        trace,
        context=_transform_context(
            source_file=source_file,
            source_root=source_root,
            output_root=output_root,
            variant="short_reasoning",
            quality_metrics={"passed": True},
        ),
        rewriter=rewriter,
    )
    transformed = result.trace

    assert trace == original
    assert result.assistant_messages_rewritten == len(assistant_indices)
    for idx, message in enumerate(transformed["history"]):
        if idx in assistant_indices:
            assert message["content"] == f"Concise reasoning for assistant turn {idx}."
            assert message["tool_calls"] == original["history"][idx]["tool_calls"]
        elif message["role"] == "tool":
            assert message == original["history"][idx]


def test_transform_split_rejects_missing_reasoning(tmp_path: Path) -> None:
    source_root = tmp_path / "standard" / "guided" / "deepseek" / "long_reasoning"
    output_root = tmp_path / "standard" / "guided" / "deepseek" / "no_reasoning"
    trace_path = source_root / "training" / "traces" / "m" / "bad.json"
    _write_trace(trace_path, _trace(seed=3, hidden_reasoning="short"))

    quality_filter = TraceQualityFilter(QualityConfig(reject_tool_call_free_turns=True))

    stats = transform_reasoning_variant_split(
        source_trace_root=source_root,
        output_trace_root=output_root,
        teacher_model="m",
        source_dir="training",
        quality_filter=quality_filter,
    )

    assert stats.transformed == 0
    assert not (output_root / "training" / "traces" / "m" / "bad.json").exists()


def test_transform_split_removes_stale_outputs_on_rerun(tmp_path: Path) -> None:
    source_root = tmp_path / "standard" / "unguided" / "deepseek" / "long_reasoning"
    output_root = tmp_path / "standard" / "unguided" / "deepseek" / "no_reasoning"
    trace_path = source_root / "training" / "traces" / "m" / "good.json"
    _write_trace(trace_path, _trace(seed=4))
    quality_filter = TraceQualityFilter(QualityConfig(reject_tool_call_free_turns=True))

    first = transform_reasoning_variant_split(
        source_trace_root=source_root,
        output_trace_root=output_root,
        teacher_model="m",
        source_dir="training",
        quality_filter=quality_filter,
    )
    output_path = output_root / "training" / "traces" / "m" / "good.json"
    assert first.transformed == 1
    assert output_path.exists()

    _write_trace(trace_path, _trace(seed=4, hidden_reasoning="short"))
    second = transform_reasoning_variant_split(
        source_trace_root=source_root,
        output_trace_root=output_root,
        teacher_model="m",
        source_dir="training",
        quality_filter=quality_filter,
    )

    assert second.transformed == 0
    assert not output_path.exists()
