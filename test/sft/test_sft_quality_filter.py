"""Unit tests for SFT trace quality filtering."""

import json

from src.dataset.privesc.sft import QualityConfig, TraceQualityFilter


def _long_text(prefix: str = "Reasoning") -> str:
    # Keep comfortably above QualityConfig.min_reasoning_length.
    return (
        f"{prefix}: I will enumerate the system and validate findings with commands before attempting escalation. "
        "I will check identity, sudo permissions, SUID/SGID binaries, writable paths, cron jobs, and file capabilities."
    )


def _tool_call(name: str, arguments: object) -> dict:
    return {
        "type": "function",
        "id": "call_1",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _metadata(*, generator_name: str | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "prompt_vars": {
            "user": "u",
            "password": "p",
            "max_turns": 5,
            "term_cols": 80,
            "term_rows": 24,
        }
    }
    if generator_name is not None:
        metadata["generator_name"] = generator_name
    return metadata


def _base_trace(
    *,
    assistant_content: str,
    assistant_tool_calls: list[dict] | None = None,
    **overrides: object,
) -> dict:
    return {
        "scenario": "suid_gtfobins",
        "mode": "trace_collection",
        "success": True,
        "turns": 2,
        "total_tokens": 500,
        "history": [
            {
                "role": "system",
                "content": "HEADER\nSECRET SOLUTION DATA\nFOOTER",
            },
            {
                "role": "user",
                "content": "Start",
            },
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": (
                    assistant_tool_calls if assistant_tool_calls is not None else []
                ),
            },
        ],
        "metadata": _metadata(),
        **overrides,
    }


def _assert_only_reason(metrics: dict, prefix: str) -> None:
    reasons = metrics.get("reasons")
    assert isinstance(reasons, list)
    assert len(reasons) == 1, reasons
    assert str(reasons[0]).startswith(prefix), reasons


def test_quality_filter_accepts_clean_trace():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(assistant_content=_long_text("Plan"))
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_tool_call_free_policy_is_opt_in():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(assistant_content=_long_text("Text only"))
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics
    assert metrics["tool_call_free_turns"] == 1

    filt = TraceQualityFilter(QualityConfig(reject_tool_call_free_turns=True))
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "tool_call_free_turn")
    assert metrics["tool_call_free_turns"] == 1

    trace = _base_trace(
        assistant_content=_long_text("Using a tool"),
        assistant_tool_calls=[_tool_call("exec_command", {"command": "id"})],
    )
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics
    assert metrics["tool_call_free_turns"] == 0


def test_quality_filter_rejects_secret_solution_reference():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=(
            "The secret solution indicates the vulnerable binary is /usr/bin/node, so I will use that. "
            "I will now enumerate permissions and exploit it to escalate privileges. "
            "I will also confirm success with id and whoami."
        )
    )
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "secret_solution_leakage")


def test_quality_filter_leaves_holdout_leakage_to_collection_audit():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=(
            _long_text("Password reuse")
            + " I will test root:trustno1 before doing anything else."
        ),
        metadata=_metadata(generator_name="password_reuse"),
    )
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_holdout_leakage_toggle_does_not_change_generic_quality():
    filt = TraceQualityFilter(QualityConfig(reject_on_holdout_leakage=False))
    trace = _base_trace(
        assistant_content=(
            _long_text("Password reuse")
            + " I will test root:trustno1 before doing anything else."
        ),
        metadata=_metadata(generator_name="password_reuse"),
    )
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_ignores_holdout_hits_in_system_prompt():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=_long_text("Clean trace"),
        metadata=_metadata(generator_name="password_reuse"),
    )
    trace["history"][0]["content"] = "HEADER\nSECRET SOLUTION DATA\ntrustno1\nFOOTER"
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_allows_generic_ssh_key_name_recon():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=(
            _long_text("SSH key recon")
            + " I will search the user's SSH directory for plausible private keys."
        ),
        assistant_tool_calls=[
            _tool_call(
                "exec_command",
                {
                    "command": "find /home/user/.ssh -type f -name id_rsa -o -name '*key*'"
                },
            )
        ],
        metadata=_metadata(generator_name="ssh_key_reuse"),
    )

    passed, metrics = filt.check_trace_live_prefix(trace)

    assert passed, metrics


def test_quality_filter_allows_docker_registry_image_reference():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=(
            _long_text("Container context")
            + " I may inspect docker.io/library/alpine:latest as an image reference."
        ),
        metadata=_metadata(generator_name="sudo_gtfobins"),
    )
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_rejects_tool_call_missing_exec_command_argument():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=_long_text("Malformed exec command"),
        assistant_tool_calls=[_tool_call("exec_command", {})],
    )
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "invalid_tool_call_schema")


def test_quality_filter_rejects_tool_call_missing_test_credentials_argument():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=_long_text("Malformed credential check"),
        assistant_tool_calls=[_tool_call("test_credentials", {"user": "root"})],
    )
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "invalid_tool_call_schema")


def test_quality_filter_accepts_valid_tool_call_schema():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=_long_text("Valid tool call"),
        assistant_tool_calls=[_tool_call("exec_command", {"command": "id"})],
    )
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_allows_invalid_tool_call_schema_when_disabled():
    filt = TraceQualityFilter(QualityConfig(reject_on_invalid_tool_call_schema=False))
    trace = _base_trace(
        assistant_content=_long_text("Disabled schema validation"),
        assistant_tool_calls=[_tool_call("exec_command", {})],
    )
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics
    assert metrics["tool_call_schema_errors"] == []
    assert metrics["has_invalid_tool_call_schema"] is False


def test_quality_filter_rejects_container_misconfig_command_not_found():
    filt = TraceQualityFilter(QualityConfig())
    trace = {
        **_base_trace(assistant_content=_long_text("Baseline tooling")),
        "history": [
            {
                "role": "system",
                "content": "HEADER\nSECRET SOLUTION DATA\nFOOTER",
            },
            {
                "role": "user",
                "content": "Start",
            },
            {
                "role": "assistant",
                "content": _long_text("Checking file"),
                "tool_calls": [],
            },
            {
                "role": "tool",
                "name": "exec_command",
                "content": (
                    '{"got_root": false, "command": "file /bin/bash", '
                    '"output": "bash: line 1: file: command not found\\n", '
                    '"exit_code": 127, "timed_out": false}'
                ),
            },
        ],
    }
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "container_misconfig")


def test_quality_filter_rejects_any_empty_reasoning_turn():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(assistant_content="ok")
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "too_many_empty_reasoning")


def test_quality_filter_allows_empty_reasoning_when_disabled():
    filt = TraceQualityFilter(QualityConfig(reject_on_empty_reasoning=False))
    trace = _base_trace(assistant_content="ok")
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_rejects_html_entities_in_tool_command():
    filt = TraceQualityFilter(QualityConfig())
    trace = {
        **_base_trace(assistant_content=_long_text("Split payload")),
        "history": [
            {
                "role": "system",
                "content": "HEADER\nSECRET SOLUTION DATA\nFOOTER",
            },
            {
                "role": "user",
                "content": "Start",
            },
            {
                "role": "assistant",
                "content": _long_text("Running split"),
                "tool_calls": [],
            },
            {
                "role": "tool",
                "name": "exec_command",
                "content": (
                    '{"got_root": false, "command": "/bin/sh -c \'echo x 0<&amp;2\'", '
                    '"output": "bash: line 1: amp: ambiguous redirect\\n", '
                    '"exit_code": 2, "timed_out": false}'
                ),
            },
        ],
    }
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "html_entities")


def test_quality_filter_rejects_html_entities_in_assistant_content():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=(
            _long_text("Command formatting")
            + " sudo split --filter='/bin/sh -i 0<&amp;2 1<&amp;2' /etc/hosts"
        )
    )
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "html_entities")


def test_quality_filter_allows_html_entities_when_disabled():
    filt = TraceQualityFilter(QualityConfig(reject_on_html_entities=False))
    trace = _base_trace(
        assistant_content=(
            _long_text("Command formatting")
            + " sudo split --filter='/bin/sh -i 0<&amp;2 1<&amp;2' /etc/hosts"
        )
    )
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_rejects_not_successful():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(assistant_content=_long_text("Not successful"), success=False)
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "not_successful")


def test_quality_filter_allows_failure_when_disabled():
    filt = TraceQualityFilter(QualityConfig(reject_on_failure=False))
    trace = _base_trace(assistant_content=_long_text("Allow failure"), success=False)
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_rejects_too_many_turns():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(assistant_content=_long_text("Too many turns"), turns=99)
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "too_many_turns")


def test_quality_filter_rejects_too_few_turns():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(assistant_content=_long_text("Too few turns"), turns=1)
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "too_few_turns")


def test_quality_filter_rejects_too_many_tokens():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=_long_text("Too many tokens"),
        total_tokens=QualityConfig.max_tokens + 1,
    )
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "too_many_tokens")


def test_quality_filter_prefers_sft_token_count():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=_long_text("Too many SFT tokens"),
        total_tokens=500,
        sft_num_tokens=QualityConfig.max_tokens + 1,
    )
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "too_many_tokens")
    assert metrics["sft_num_tokens"] == QualityConfig.max_tokens + 1


def test_quality_filter_rejects_single_nudge_by_default():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(assistant_content=_long_text("Nudges"))
    trace["history"] = [
        {"role": "system", "content": "HEADER\nSECRET SOLUTION DATA\nFOOTER"},
        {"role": "user", "content": "No tool calls received"},
        {"role": "assistant", "content": _long_text("Nudges"), "tool_calls": []},
    ]
    trace["turns"] = 2
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "too_many_nudges")


def test_quality_filter_allows_single_nudge_when_enabled():
    filt = TraceQualityFilter(QualityConfig(max_nudges=1))
    trace = _base_trace(assistant_content=_long_text("Nudges"))
    trace["history"] = [
        {"role": "system", "content": "HEADER\nSECRET SOLUTION DATA\nFOOTER"},
        {"role": "user", "content": "No tool calls received"},
        {"role": "assistant", "content": _long_text("Nudges"), "tool_calls": []},
    ]
    trace["turns"] = 2
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_quality_filter_rejects_missing_solution_data_in_prompt():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(assistant_content=_long_text("Prompt integrity"))
    trace["history"][0]["content"] = "HEADER\nNO SOLUTION HERE\nFOOTER"
    passed, metrics = filt.check_trace(trace)
    assert not passed
    _assert_only_reason(metrics, "missing_solution_data_in_prompt")


def test_quality_filter_accepts_normalized_prompt_with_source_marker():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(assistant_content=_long_text("Prompt integrity"))
    trace["history"][0]["content"] = "DEPLOYMENT PROMPT"
    trace["_source_trace_had_solution_block"] = True
    passed, metrics = filt.check_trace(trace)
    assert passed, metrics


def test_live_prefix_filter_allows_unsuccessful_short_prefix():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=_long_text("Partial trace"),
        success=False,
        turns=1,
    )
    passed, metrics = filt.check_trace_live_prefix(trace)
    assert passed, metrics


def test_live_prefix_filter_rejects_missing_reasoning_before_repair():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content="",
        assistant_tool_calls=[_tool_call("exec_command", {"command": "id"})],
        success=False,
        turns=1,
    )
    passed, metrics = filt.check_trace_live_prefix(trace)
    assert not passed
    _assert_only_reason(metrics, "too_many_empty_reasoning")


def test_live_prefix_filter_rejects_xml_only_tool_call_content():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content='<tool_call>{"name":"exec_command","arguments":{"command":"id"}}</tool_call>',
        assistant_tool_calls=[_tool_call("exec_command", {"command": "id"})],
        success=False,
        turns=1,
    )
    passed, metrics = filt.check_trace_live_prefix(trace)
    assert not passed
    _assert_only_reason(metrics, "too_many_empty_reasoning")


def test_live_prefix_filter_still_rejects_prefix_violation():
    filt = TraceQualityFilter(QualityConfig())
    trace = _base_trace(
        assistant_content=_long_text("Too many turns"),
        success=False,
        turns=QualityConfig.max_turns + 1,
    )
    passed, metrics = filt.check_trace_live_prefix(trace)
    assert not passed
    _assert_only_reason(metrics, "too_many_turns")
