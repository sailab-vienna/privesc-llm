"""Tests for the paper reward ladder."""

import pytest

from src.config import PrivEscRewardConfig, resolve_reward_config
from src.vf_envs import reward as calc


def _tool_call(name: str, arguments: str) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


def _assistant(content: str, *tool_calls: dict) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": list(tool_calls),
    }


def _tool(content: str = '{"output":"ok"}') -> dict:
    return {"role": "tool", "content": content}


def _clean_success_trace() -> list[dict]:
    return [
        _assistant(
            "Check the current account.",
            _tool_call("exec_command", '{"command":"id"}'),
        ),
        _tool('{"output":"uid=1000(lowpriv)"}'),
        _assistant(
            "Try the verified root credential.",
            _tool_call(
                "test_credentials",
                '{"user":"root","password":"trustno1"}',
            ),
        ),
        _tool('{"success":true,"got_root":true}'),
    ]


def _clean_failed_trace() -> list[dict]:
    return [
        _assistant(
            "Check the current account.",
            _tool_call("exec_command", '{"command":"id"}'),
        ),
        _tool('{"output":"uid=1000(lowpriv)"}'),
        _assistant(
            "Check sudo privileges.",
            _tool_call("exec_command", '{"command":"sudo -l"}'),
        ),
        _tool('{"output":"not allowed"}'),
    ]


def _metrics(
    *,
    first_root_turn: int | None = 2,
    llm_ms: float = 40_000.0,
    tool_ms: float = 20_000.0,
    assistant_turns: int = 2,
    tool_calls_executed: int = 2,
) -> dict:
    result = {
        "assistant_turns": assistant_turns,
        "tool_calls_executed": tool_calls_executed,
        "total_llm_ms_clipped": llm_ms,
        "total_tool_ms_clipped": tool_ms,
    }
    if first_root_turn is not None:
        result["first_root_turn"] = first_root_turn
    return result


def _calculate(
    mode: str,
    *,
    trace: list[dict] | None = None,
    got_root: bool = True,
    first_root_turn: int | None = 2,
    llm_ms: float = 40_000.0,
    tool_ms: float = 20_000.0,
    reward_overrides: dict | None = None,
) -> calc.RewardOutput:
    config = {"mode": mode, **(reward_overrides or {})}
    return calc.calculate_privesc_reward(
        trace or _clean_success_trace(),
        got_root=got_root,
        max_turns=20,
        reward_config=config,
        episode_metrics=_metrics(
            first_root_turn=first_root_turn,
            llm_ms=llm_ms,
            tool_ms=tool_ms,
        ),
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("outcome", 1.0),
        ("outcome_cost", 1.0 + 0.1 * (1.0 - 60_000.0 / 540_000.0)),
        ("outcome_round", 1.9),
        ("outcome_round_cost", 1.9 + 0.1 * (1.0 - 60_000.0 / 540_000.0)),
    ],
)
def test_successful_clean_trace_under_ladder_mode(mode: str, expected: float) -> None:
    result = _calculate(mode)

    assert result.value == pytest.approx(expected)
    assert result.metadata["R_out"] == pytest.approx(1.0)
    assert result.metadata["R_round"] == pytest.approx(0.9)
    assert result.metadata["R_iface"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "mode",
    [
        "outcome",
        "outcome_cost",
        "outcome_round",
        "outcome_round_cost",
    ],
)
def test_failed_clean_trace_under_ladder_mode(mode: str) -> None:
    result = _calculate(
        mode,
        trace=_clean_failed_trace(),
        got_root=False,
        first_root_turn=None,
    )

    assert result.value == pytest.approx(-1.0)
    assert result.metadata["R_out"] == pytest.approx(-1.0)
    assert result.metadata["R_round"] == pytest.approx(0.0)
    assert result.metadata["R_cost"] == pytest.approx(0.0)
    assert result.metadata["R_iface"] == pytest.approx(0.0)


def test_r_out_success_and_failure() -> None:
    success = _calculate("outcome")
    failure = _calculate("outcome", got_root=False, first_root_turn=None)

    assert success.metadata["R_out"] == pytest.approx(1.0)
    assert failure.metadata["R_out"] == pytest.approx(-1.0)


def test_r_round_uses_first_root_turn_and_clips() -> None:
    trace = _clean_success_trace() + [
        _assistant(
            "This later turn should not define H_root.",
            _tool_call("exec_command", '{"command":"whoami"}'),
        )
    ]

    early = _calculate("outcome_round", trace=trace, first_root_turn=2)
    clipped = _calculate("outcome_round", first_root_turn=40)
    failed = _calculate("outcome_round", got_root=False, first_root_turn=None)

    assert early.metadata["H_root"] == 2
    assert early.metadata["R_round"] == pytest.approx(0.9)
    assert clipped.metadata["R_round"] == pytest.approx(0.0)
    assert failed.metadata["R_round"] == pytest.approx(0.0)


def test_c_ms_adds_clipped_llm_and_tool_costs_for_fanout() -> None:
    trace = [
        _assistant(
            "Fan out two checks.",
            _tool_call("exec_command", '{"command":"id"}'),
            _tool_call("exec_command", '{"command":"whoami"}'),
        ),
        _tool(),
        _tool(),
    ]

    result = calc.calculate_privesc_reward(
        trace,
        got_root=True,
        max_turns=20,
        reward_config={"mode": "outcome_round_cost"},
        episode_metrics=_metrics(
            first_root_turn=1,
            llm_ms=20_000.0,
            tool_ms=130_000.0,
            assistant_turns=1,
            tool_calls_executed=2,
        ),
    )

    assert result.metadata["tool_calls_executed"] == 2
    assert result.metadata["total_llm_ms_clipped"] == pytest.approx(20_000.0)
    assert result.metadata["total_tool_ms_clipped"] == pytest.approx(130_000.0)
    assert result.metadata["C_ms"] == pytest.approx(150_000.0)


def test_r_cost_success_gating_and_reference_clipping() -> None:
    half_ref = _calculate("outcome_round_cost", llm_ms=200_000.0, tool_ms=70_000.0)
    over_ref = _calculate("outcome_round_cost", llm_ms=540_000.0, tool_ms=1.0)
    failed = _calculate(
        "outcome_round_cost",
        got_root=False,
        first_root_turn=None,
        llm_ms=0.0,
        tool_ms=0.0,
    )

    assert half_ref.metadata["R_cost"] == pytest.approx(0.5)
    assert over_ref.metadata["R_cost"] == pytest.approx(0.0)
    assert failed.metadata["R_cost"] == pytest.approx(0.0)


def test_weighted_r_cost_uses_lambda_cost() -> None:
    result = _calculate(
        "outcome_round_cost",
        llm_ms=100_000.0,
        tool_ms=0.0,
        reward_overrides={"lambda_cost": 0.25},
    )

    assert result.metadata["R_cost"] == pytest.approx(1.0 - 100_000.0 / 540_000.0)
    assert result.metadata["weighted_R_cost"] == pytest.approx(
        0.25 * result.metadata["R_cost"]
    )
    assert result.value == pytest.approx(
        result.metadata["R_out"]
        + result.metadata["R_round"]
        + result.metadata["weighted_R_cost"]
        + result.metadata["R_iface"]
    )


def test_outcome_cost_uses_outcome_iface_and_weighted_cost_only() -> None:
    result = _calculate("outcome_cost", first_root_turn=2)

    assert result.metadata["R_out"] == pytest.approx(1.0)
    assert result.metadata["R_round"] == pytest.approx(0.9)
    assert result.metadata["weighted_R_cost"] == pytest.approx(
        0.1 * (1.0 - 60_000.0 / 540_000.0)
    )
    assert result.value == pytest.approx(
        result.metadata["R_out"]
        + result.metadata["weighted_R_cost"]
        + result.metadata["R_iface"]
    )


def test_observability_terms_are_populated_for_all_modes() -> None:
    trace = [
        _assistant("Check identity.", _tool_call("exec_command", '{"command":"id"}')),
        _tool('{"output":"uid=1000(lowpriv)"}'),
        _assistant("Repeat identity.", _tool_call("exec_command", '{"command":" id "}')),
        _tool('{"error":"transient failure"}'),
        _assistant(
            "Try root credential.",
            _tool_call("test_credentials", '{"user":"root","password":"trustno1"}'),
        ),
        _tool('{"success":true,"got_root":true}'),
    ]

    result = _calculate("outcome", trace=trace, first_root_turn=3)

    assert result.value == pytest.approx(1.0)
    assert result.metadata["total_repetitions"] == 1
    assert result.metadata["R_repeat"] == pytest.approx(0.15)
    assert result.metadata["total_short_content"] == 3
    assert result.metadata["total_tool_errors"] == 1
    assert result.metadata["total_recon_commands"] == 1
    assert result.metadata["speed_bonus"] == pytest.approx(0.85)
    assert result.metadata["recon_bonus"] == pytest.approx(0.05)
    assert result.metadata["structured_final_reward"] != pytest.approx(result.value)


def test_recon_observability_ignores_non_string_commands() -> None:
    result = _calculate(
        "outcome",
        trace=[
            _assistant(
                "Malformed command.",
                _tool_call("exec_command", '{"command":null}'),
            ),
            _tool(),
        ],
        first_root_turn=1,
    )

    assert result.metadata["total_recon_commands"] == 0


@pytest.mark.parametrize(
    ("trace", "expected_counts"),
    [
        (
            [{"role": "assistant", "content": "I forgot to call a tool."}],
            {"no_tool_call_count": 1},
        ),
        (
            [{"role": "assistant", "content": '<tool_call>{"name":</tool_call>'}],
            {"malformed_tool_call_count": 1},
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": '<tool_call>{"arguments":{"command":"id"}}</tool_call>',
                }
            ],
            {"invalid_tool_name_count": 1},
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"exec_command","arguments":{}}</tool_call>',
                }
            ],
            {"invalid_tool_args_count": 1},
        ),
        (
            [{"role": "assistant", "content": "<tool_call>[]</tool_call>"}],
            {"malformed_tool_call_count": 1},
        ),
        (
            [
                _assistant(
                    "Call an unavailable tool.",
                    _tool_call("shell_command", '{"command":"id"}'),
                )
            ],
            {"invalid_tool_name_count": 1},
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "Missing raw tool name.",
                    "tool_calls": [{"function": {"arguments": '{"command":"id"}'}}],
                }
            ],
            {"invalid_tool_name_count": 1},
        ),
    ],
)
def test_interface_penalty_signals(
    trace: list[dict], expected_counts: dict[str, int]
) -> None:
    result = _calculate("outcome", trace=trace, first_root_turn=1)

    assert result.metadata["interface_violation"] is True
    assert result.metadata["R_iface"] == pytest.approx(-0.05)
    for key, expected in expected_counts.items():
        assert result.metadata[key] == expected


def test_multiple_malformed_turns_have_one_capped_interface_penalty() -> None:
    result = _calculate(
        "outcome",
        trace=[
            {"role": "assistant", "content": '<tool_call>{"name":</tool_call>'},
            {"role": "assistant", "content": "<tool_call>[]</tool_call>"},
        ],
        first_root_turn=1,
    )

    assert result.metadata["malformed_tool_call_count"] == 2
    assert result.metadata["R_iface"] == pytest.approx(-0.05)
    assert result.value == pytest.approx(0.95)


def test_interface_penalty_preserves_success_failure_dominance() -> None:
    malformed_success = _calculate(
        "outcome_round_cost",
        trace=[{"role": "assistant", "content": "<tool_call>[]</tool_call>"}],
        first_root_turn=20,
        llm_ms=540_000.0,
        tool_ms=0.0,
    )
    clean_failure = _calculate(
        "outcome_round_cost",
        trace=_clean_failed_trace(),
        got_root=False,
        first_root_turn=None,
        llm_ms=0.0,
        tool_ms=0.0,
    )

    assert malformed_success.value == pytest.approx(0.95)
    assert clean_failure.value == pytest.approx(-1.0)
    assert malformed_success.value > clean_failure.value


def test_same_round_lower_cost_trace_ranks_higher() -> None:
    low_cost = _calculate("outcome_round_cost", first_root_turn=4, llm_ms=50_000.0)
    high_cost = _calculate("outcome_round_cost", first_root_turn=4, llm_ms=500_000.0)

    assert low_cost.metadata["H_root"] == high_cost.metadata["H_root"]
    assert low_cost.value > high_cost.value


def test_one_round_faster_trace_ranks_higher_unless_cost_is_extreme() -> None:
    faster = _calculate("outcome_round_cost", first_root_turn=3, llm_ms=100_000.0)
    slower = _calculate("outcome_round_cost", first_root_turn=4, llm_ms=100_000.0)
    offset_tie = _calculate(
        "outcome_round_cost", first_root_turn=3, llm_ms=270_000.0, tool_ms=0.0
    )
    offset_over = _calculate(
        "outcome_round_cost", first_root_turn=3, llm_ms=271_000.0, tool_ms=0.0
    )
    free_slower = _calculate(
        "outcome_round_cost", first_root_turn=4, llm_ms=0.0, tool_ms=0.0
    )

    assert faster.value > slower.value
    assert offset_tie.value == pytest.approx(free_slower.value)
    assert free_slower.value > offset_over.value


def test_formatted_failures_cannot_outrank_successful_trajectories() -> None:
    failure = _calculate(
        "outcome_round_cost",
        trace=_clean_failed_trace(),
        got_root=False,
        first_root_turn=None,
        llm_ms=0.0,
        tool_ms=0.0,
    )
    worst_success = _calculate(
        "outcome_round_cost",
        first_root_turn=20,
        llm_ms=540_000.0,
        tool_ms=0.0,
    )

    assert failure.value == pytest.approx(-1.0)
    assert worst_success.value == pytest.approx(1.0)
    assert worst_success.value > failure.value


def test_missing_reasoning_is_logged_but_does_not_change_reward() -> None:
    with_reasoning = _calculate("outcome_round", trace=_clean_success_trace())
    no_reasoning_trace = [
        _assistant("", _tool_call("exec_command", '{"command":"id"}')),
        _tool(),
        _assistant(
            "",
            _tool_call("test_credentials", '{"user":"root","password":"trustno1"}'),
        ),
        _tool('{"got_root":true}'),
    ]
    without_reasoning = _calculate("outcome_round", trace=no_reasoning_trace)

    assert without_reasoning.value == pytest.approx(with_reasoning.value)
    assert with_reasoning.metadata["missing_reasoning_count"] == 0
    assert without_reasoning.metadata["missing_reasoning_count"] == 2


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("outcome_only", "outcome"),
        ("outcome_speed", "outcome_round"),
        ("outcome_speed_cost", "outcome_round_cost"),
    ],
)
def test_deprecated_aliases_resolve_to_canonical_modes(
    alias: str, canonical: str
) -> None:
    cfg = resolve_reward_config({"mode": alias})

    assert cfg.mode == canonical
    assert cfg.to_dict()["mode"] == canonical


@pytest.mark.parametrize(
    "reward_config",
    [
        {"mode": "invalid_mode"},
        {"mode": "outcome_round_cost", "h_max": 0},
        {"mode": "outcome_round_cost", "lambda_cost": -0.1},
        {"mode": "outcome_round_cost", "c_ref_ms": 0},
        {"mode": "outcome_round_cost", "llm_ms_clip_ms": 0},
        {"mode": "outcome_round_cost", "tool_ms_clip_ms": 0},
        {"mode": "outcome_round_cost", "iface_penalty": -0.1},
    ],
)
def test_invalid_reward_config_raises(reward_config: dict) -> None:
    with pytest.raises(ValueError):
        calc.build_reward_builder(reward_config)


def test_round_modes_require_first_root_turn_for_success() -> None:
    with pytest.raises(ValueError, match="first_root_turn"):
        _calculate("outcome_round", first_root_turn=None)


def test_cost_mode_requires_clipped_cost_metrics_for_success() -> None:
    for mode in ("outcome_cost", "outcome_round_cost"):
        with pytest.raises(ValueError, match="total_llm_ms_clipped"):
            calc.calculate_privesc_reward(
                _clean_success_trace(),
                got_root=True,
                max_turns=20,
                reward_config={"mode": mode},
                episode_metrics={"first_root_turn": 2},
            )


def test_failed_cost_mode_allows_missing_cost_metrics() -> None:
    result = calc.calculate_privesc_reward(
        _clean_failed_trace(),
        got_root=False,
        max_turns=20,
        reward_config={"mode": "outcome_round_cost"},
    )

    assert result.value == pytest.approx(-1.0)
    assert result.metadata["C_ms"] == pytest.approx(0.0)
    assert result.metadata["R_cost"] == pytest.approx(0.0)


def test_paper_default_config_values() -> None:
    cfg = PrivEscRewardConfig()

    assert cfg.mode == "outcome_round_cost"
    assert cfg.h_max == 20
    assert cfg.lambda_cost == pytest.approx(0.1)
    assert cfg.c_ref_ms == pytest.approx(540_000.0)
    assert cfg.llm_ms_clip_ms == pytest.approx(20_000.0)
    assert cfg.tool_ms_clip_ms == pytest.approx(65_000.0)
    assert cfg.iface_penalty == pytest.approx(0.05)


def test_reward_metadata_includes_required_fields() -> None:
    result = _calculate("outcome_round_cost")
    required = {
        "reward_mode",
        "reward_config",
        "got_root",
        "R_out",
        "R_round",
        "R_cost",
        "weighted_R_cost",
        "R_repeat",
        "R_iface",
        "final_reward",
        "H_root",
        "C_ms",
        "total_llm_ms_clipped",
        "total_tool_ms_clipped",
        "interface_violation",
        "no_tool_call_count",
        "malformed_tool_call_count",
        "invalid_tool_name_count",
        "invalid_tool_args_count",
        "missing_reasoning_count",
        "total_repetitions",
        "total_short_content",
        "total_tool_errors",
        "total_recon_commands",
        "speed_bonus",
        "recon_bonus",
        "total_penalty",
        "repetition_penalty",
        "tool_error_penalty",
        "no_tool_calls_penalty",
        "short_content_penalty",
        "structured_no_round_final_reward",
        "structured_final_reward",
    }
    removed_aliases = {
        "success",
        "root",
        "first_root_turn",
        "cost_ms",
        "trajectory_length",
        "total_assistant_turns",
        "tau",
        "tau_max",
        "total_no_tool_calls",
        "total_wasted_turns_no_valid_interaction",
        "R_waste",
        "R_dup_pen",
        "R_eff_turn",
        "R_eff_tool",
        "R_eff",
        "lambda_cost",
        "h_max",
        "c_ref_ms",
        "llm_ms_clip_ms",
        "tool_ms_clip_ms",
        "iface_penalty",
    }

    assert required <= set(result.metadata)
    assert removed_aliases.isdisjoint(result.metadata)
