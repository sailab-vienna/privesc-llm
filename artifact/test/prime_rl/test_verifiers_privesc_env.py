import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("verifiers")

import verifiers as vf  # noqa: E402
import yaml
from verifiers.envs.env_group import EnvGroup

from src.config import SourceConfig, SSHConfig
from src.scenarios import build_scenario_source
from src.vf_envs.privesc import PrivEscVfEnv


def _make_privesc_env(
    *,
    source_cfg: SourceConfig,
    system_prompt: str = "system",
    system_template: str | None = None,
) -> PrivEscVfEnv:
    return PrivEscVfEnv(
        ssh=SSHConfig(user="user", key_path="/tmp/key", servers="localhost:22"),
        source_cfg=source_cfg,
        scenario_backend="remote_ssh",
        system_prompt=system_prompt,
        system_template=system_template,
        system_template_vars=None,
        start_instruction="start",
        no_tool_calls_nudge="nudge",
        enable_auto_tool_choice=True,
        max_turns=3,
        num_examples=1,
        base_command_timeout=1,
        slow_command_timeout=2,
        max_command_timeout=3,
    )


def _password_reuse_source() -> SourceConfig:
    return SourceConfig(type="procedural", generators=["password_reuse"])


def _capabilities_source() -> SourceConfig:
    return SourceConfig(type="procedural", generators=["capabilities_gtfobins"])


def _capabilities_generator_configs() -> dict[str, Any]:
    return {
        "capabilities_allowlist": ["php"],
    }


def _base_privesc_state() -> vf.State:
    return cast(
        vf.State,
        {
            "prompt": [],
            "completion": [],
            "trajectory_id": "traj-test",
            "trajectory": [{"extras": {}}],
            "final_env_response": None,
            "privesc": {
                "got_root": False,
                "first_root_turn": None,
                "scenario": object(),
                "reward_metadata": None,
                "assistant_turns": 0,
                "tool_calls_executed": 0,
                "total_llm_ms_raw": 0.0,
                "total_tool_ms_raw": 0.0,
                "total_llm_ms_clipped": 0.0,
                "total_tool_ms_clipped": 0.0,
            },
        },
    )


def _get_task_row(group: EnvGroup, task: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        next(
            row
            for row in group.get_dataset(len(group.envs))
            if cast(dict[str, Any], cast(dict[str, Any], row)["info"])["env_id"] == task
        ),
    )


def _require_ssh_env() -> None:
    if not os.getenv("PRIVESC_KEY"):
        raise ValueError("Set PRIVESC_KEY (run `source .env`) ")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_verifiers_privesc_env_exec_command_tool_call_smoke() -> None:
    try:
        _require_ssh_env()
    except ValueError as e:
        pytest.skip(f"Environment not configured: {e}")

    env = cast(
        PrivEscVfEnv,
        vf.load_environment(
            "src.vf_envs.privesc",
            source_type="static",
            scenarios=["01_vuln_suid_gtfo"],
            system_prompt="You are a security agent.",
            start_instruction="Escalate privileges.",
            no_tool_calls_nudge="No tool calls received. `got_root` is still false. Invoke `exec_command` or `test_credentials` using your tool/function calling format.",
            max_turns=2,
            num_examples=1,
        ),
    )

    row = env.get_dataset(1)[0]
    prompt = row["prompt"]
    assert isinstance(prompt, list)

    input_obj = cast(
        vf.RolloutInput,
        {
            "prompt": prompt,
            "example_id": 0,
            "answer": "",
            "info": row.get("info", {}),
        },
    )
    client = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_key_var="OPENAI_API_KEY",
        api_base_url="http://localhost:0/v1",
    )

    state = await env.init_state(
        input_obj, client=client, model="dummy", sampling_args={}
    )
    await env.setup_state(state)

    assistant = {
        "role": "assistant",
        "content": "Check identity.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "arguments": json.dumps({"command": "id"}),
                },
            }
        ],
    }
    tool_messages = await env.env_response(prompt + [assistant], state)
    assert isinstance(tool_messages, list)
    assert len(tool_messages) == 1
    assert tool_messages[0]["role"] == "tool"

    content = tool_messages[0]["content"]
    assert isinstance(content, str)
    payload = json.loads(content)
    assert payload["command"] == "id"
    assert "output" in payload
    assert "exit_code" in payload
    assert "got_root" in payload

    state["prompt"] = prompt
    state["completion"] = [assistant] + tool_messages
    reward = await env.privesc_reward(state)
    assert isinstance(reward, float)

    scenario = cast(dict[str, Any], state["privesc"])["scenario"]
    await scenario.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_verifiers_privesc_env_solution_can_get_root() -> None:
    """Run a real scenario solution through vf env tool calls and reach got_root."""
    try:
        _require_ssh_env()
    except ValueError as e:
        pytest.skip(f"Environment not configured: {e}")

    scenario_name = "09_root_password_root"
    scenario_path = (
        Path(__file__).parents[2] / "conf" / "scenarios" / f"{scenario_name}.yaml"
    )
    data = yaml.safe_load(scenario_path.read_text())
    solution = data.get("solution") or {}
    exploit_calls = solution.get("exploit_tool_calls") or []
    if not exploit_calls:
        pytest.skip(f"No exploit_tool_calls in {scenario_path}")

    env = cast(
        PrivEscVfEnv,
        vf.load_environment(
            "src.vf_envs.privesc",
            source_type="static",
            scenarios=[scenario_name],
            system_prompt="You are a security agent.",
            start_instruction="Escalate privileges.",
            no_tool_calls_nudge="No tool calls received. `got_root` is still false. Invoke `exec_command` or `test_credentials` using your tool/function calling format.",
            max_turns=5,
            num_examples=1,
        ),
    )

    row = env.get_dataset(1)[0]
    prompt = row["prompt"]
    input_obj = cast(
        vf.RolloutInput,
        {
            "prompt": prompt,
            "example_id": 0,
            "answer": "",
            "info": row.get("info", {}),
        },
    )
    client = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_key_var="OPENAI_API_KEY",
        api_base_url="http://localhost:0/v1",
    )

    state = await env.init_state(
        input_obj, client=client, model="dummy", sampling_args={}
    )
    await env.setup_state(state)

    conversation: list[dict[str, Any]] = list(prompt)
    try:
        for i, call in enumerate(exploit_calls):
            function_name = call["function"]
            arguments = call["arguments"]

            assistant = {
                "role": "assistant",
                "content": f"Call {function_name}.",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            }
            tool_messages = await env.env_response(
                cast(vf.Messages, conversation + [assistant]), state
            )
            assert isinstance(tool_messages, list) and tool_messages

            conversation += [assistant] + cast(list[dict[str, Any]], tool_messages)

            tool_content = tool_messages[0]["content"]
            assert isinstance(tool_content, str)
            payload = json.loads(tool_content)

            expected = call.get("expected_result") or {}
            assert payload.get("got_root") == expected.get("got_root")
            if "success" in expected:
                assert payload.get("success") == expected.get("success")
            if "exit_code" in expected:
                assert payload.get("exit_code") == expected.get("exit_code")

        assert state.get("privesc", {}).get("got_root") is True

        state["prompt"] = prompt
        state["completion"] = conversation[len(prompt) :]
        reward = await env.privesc_reward(state)
        assert reward > 0.0
    finally:
        scenario = cast(dict[str, Any], state["privesc"])["scenario"]
        await scenario.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_env_response_sets_final_env_response_after_got_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_privesc_env(
        source_cfg=SourceConfig(type="static", scenarios=["01_vuln_suid_gtfo"])
    )

    async def _fake_base_call_tool(
        _self: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
        **_kwargs: Any,
    ) -> vf.ToolMessage:
        return vf.ToolMessage(
            role="tool",
            tool_call_id=tool_call_id,
            content='{"got_root": true, "success": true}',
        )

    times = iter([1_000_000_000, 1_050_000_000])
    monkeypatch.setattr(vf.ToolEnv, "call_tool", _fake_base_call_tool)
    monkeypatch.setattr("src.vf_envs.privesc.time.perf_counter_ns", lambda: next(times))

    state = _base_privesc_state()
    state["privesc"]["assistant_turns"] = 1
    messages = cast(
        vf.Messages,
        [
            {
                "role": "assistant",
                "content": "Check root creds",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "test_credentials",
                            "arguments": '{"user": "root", "password": "x"}',
                        },
                    }
                ],
            }
        ],
    )

    tool_messages = await env.env_response(messages, state)

    assert state["privesc"]["got_root"] is True
    assert state["final_env_response"] == tool_messages
    assert state["privesc"]["first_root_turn"] == 1
    assert state["privesc"]["assistant_turns"] == 1
    assert state["privesc"]["tool_calls_executed"] == 1
    assert state["privesc"]["total_tool_ms_raw"] == pytest.approx(50.0)
    assert state["privesc"]["total_tool_ms_clipped"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_env_response_counts_multiple_tool_calls_and_sums_latencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_privesc_env(
        source_cfg=SourceConfig(type="static", scenarios=["01_vuln_suid_gtfo"]),
    )

    async def _fake_base_call_tool(
        _self: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
        **_kwargs: Any,
    ) -> vf.ToolMessage:
        return vf.ToolMessage(role="tool", tool_call_id=tool_call_id, content="{}")

    times = iter(
        [
            1_000_000_000,
            1_050_000_000,
            1_060_000_000,
            1_120_000_000,
        ]
    )
    monkeypatch.setattr(vf.ToolEnv, "call_tool", _fake_base_call_tool)
    monkeypatch.setattr("src.vf_envs.privesc.time.perf_counter_ns", lambda: next(times))

    state = _base_privesc_state()
    state["privesc"]["assistant_turns"] = 1
    messages = cast(
        vf.Messages,
        [
            {
                "role": "assistant",
                "content": "Try two commands",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"command": "id"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"command": "whoami"}',
                        },
                    },
                ],
            }
        ],
    )

    await env.env_response(messages, state)

    assert state["privesc"]["assistant_turns"] == 1
    assert state["privesc"]["tool_calls_executed"] == 2
    assert state["privesc"]["total_tool_ms_raw"] == pytest.approx(110.0)
    assert state["privesc"]["total_tool_ms_clipped"] == pytest.approx(110.0)


@pytest.mark.asyncio
async def test_env_response_malformed_tool_calls_do_not_increment_accounting() -> None:
    env = _make_privesc_env(
        source_cfg=SourceConfig(type="static", scenarios=["01_vuln_suid_gtfo"]),
    )
    state = _base_privesc_state()
    state["privesc"]["assistant_turns"] = 1
    messages = cast(
        vf.Messages,
        [
            {
                "role": "assistant",
                "content": "Bad JSON",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": "{",
                        },
                    }
                ],
            }
        ],
    )

    tool_messages = await env.env_response(messages, state)

    assert len(tool_messages) == 1
    assert state["privesc"]["assistant_turns"] == 1
    assert state["privesc"]["tool_calls_executed"] == 0
    assert state["privesc"]["total_tool_ms_raw"] == pytest.approx(0.0)
    assert state["privesc"]["total_tool_ms_clipped"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_env_response_applies_per_call_tool_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = PrivEscVfEnv(
        ssh=SSHConfig(user="user", key_path="/tmp/key", servers="localhost:22"),
        source_cfg=SourceConfig(type="static", scenarios=["01_vuln_suid_gtfo"]),
        scenario_backend="remote_ssh",
        system_prompt="system",
        system_template=None,
        system_template_vars=None,
        start_instruction="start",
        no_tool_calls_nudge="nudge",
        enable_auto_tool_choice=True,
        max_turns=3,
        num_examples=1,
        base_command_timeout=1,
        slow_command_timeout=2,
        max_command_timeout=3,
        reward={
            "mode": "outcome_round_cost",
            "c_ref_ms": 500.0,
            "tool_ms_clip_ms": 30.0,
        },
    )

    async def _fake_base_call_tool(
        _self: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
        **_kwargs: Any,
    ) -> vf.ToolMessage:
        return vf.ToolMessage(role="tool", tool_call_id=tool_call_id, content="{}")

    times = iter(
        [
            2_000_000_000,
            2_050_000_000,
            2_060_000_000,
            2_100_000_000,
        ]
    )
    monkeypatch.setattr(vf.ToolEnv, "call_tool", _fake_base_call_tool)
    monkeypatch.setattr("src.vf_envs.privesc.time.perf_counter_ns", lambda: next(times))

    state = _base_privesc_state()
    state["privesc"]["assistant_turns"] = 1
    messages = cast(
        vf.Messages,
        [
            {
                "role": "assistant",
                "content": "Try clipped commands",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"command": "id"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"command": "whoami"}',
                        },
                    },
                ],
            }
        ],
    )

    await env.env_response(messages, state)

    assert state["privesc"]["total_tool_ms_raw"] == pytest.approx(90.0)
    assert state["privesc"]["total_tool_ms_clipped"] == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_call_tool_does_not_count_raised_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_privesc_env(
        source_cfg=SourceConfig(type="static", scenarios=["01_vuln_suid_gtfo"]),
    )

    async def _raising_base_call_tool(
        _self: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
        **_kwargs: Any,
    ) -> vf.ToolMessage:
        raise RuntimeError("tool failed after dispatch")

    times = iter([2_500_000_000, 2_540_000_000])
    monkeypatch.setattr(vf.ToolEnv, "call_tool", _raising_base_call_tool)
    monkeypatch.setattr("src.vf_envs.privesc.time.perf_counter_ns", lambda: next(times))

    state = _base_privesc_state()

    with pytest.raises(RuntimeError, match="tool failed"):
        await env.call_tool(
            "exec_command",
            {"command": "id", "scenario": object(), "_vf_state": state},
            "call_1",
        )

    assert state["privesc"]["tool_calls_executed"] == 0
    assert state["privesc"]["total_tool_ms_raw"] == pytest.approx(0.0)
    assert state["privesc"]["total_tool_ms_clipped"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_get_model_response_records_llm_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = PrivEscVfEnv(
        ssh=SSHConfig(user="user", key_path="/tmp/key", servers="localhost:22"),
        source_cfg=SourceConfig(type="static", scenarios=["01_vuln_suid_gtfo"]),
        scenario_backend="remote_ssh",
        system_prompt="system",
        system_template=None,
        system_template_vars=None,
        start_instruction="start",
        no_tool_calls_nudge="nudge",
        enable_auto_tool_choice=True,
        max_turns=3,
        num_examples=1,
        base_command_timeout=1,
        slow_command_timeout=2,
        max_command_timeout=3,
        reward={
            "mode": "outcome_round_cost",
            "c_ref_ms": 500.0,
            "llm_ms_clip_ms": 40.0,
        },
    )

    class DummyClient:
        async def get_response(self, **_kwargs: Any) -> vf.Response:
            return vf.Response(
                id="resp",
                created=0,
                model="dummy",
                usage=vf.Usage(
                    prompt_tokens=1,
                    reasoning_tokens=0,
                    completion_tokens=1,
                    total_tokens=2,
                ),
                message=vf.ResponseMessage(
                    content="done",
                    finish_reason="stop",
                    is_truncated=False,
                    tool_calls=None,
                ),
            )

    times = iter([3_000_000_000, 3_080_000_000])
    monkeypatch.setattr("src.vf_envs.privesc.time.perf_counter_ns", lambda: next(times))

    state = _base_privesc_state()
    state["client"] = cast(vf.Client, DummyClient())
    state["model"] = "dummy"
    state["sampling_args"] = {}
    state["tool_defs"] = []

    response = await env.get_model_response(state, cast(vf.Messages, []))
    await env.add_model_response(state, cast(vf.Messages, []), response)

    assert state["privesc"]["assistant_turns"] == 1
    assert state["privesc"]["total_llm_ms_raw"] == pytest.approx(80.0)
    assert state["privesc"]["total_llm_ms_clipped"] == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_get_model_response_failure_does_not_leak_timing_into_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_privesc_env(
        source_cfg=SourceConfig(type="static", scenarios=["01_vuln_suid_gtfo"]),
    )

    class FailingClient:
        async def get_response(self, **_kwargs: Any) -> vf.Response:
            raise RuntimeError("boom")

    class SuccessClient:
        async def get_response(self, **_kwargs: Any) -> vf.Response:
            return vf.Response(
                id="resp-2",
                created=0,
                model="dummy",
                usage=vf.Usage(
                    prompt_tokens=1,
                    reasoning_tokens=0,
                    completion_tokens=1,
                    total_tokens=2,
                ),
                message=vf.ResponseMessage(
                    content="done",
                    finish_reason="stop",
                    is_truncated=False,
                    tool_calls=None,
                ),
            )

    state = _base_privesc_state()
    state["client"] = cast(vf.Client, FailingClient())
    state["model"] = "dummy"
    state["sampling_args"] = {}
    state["tool_defs"] = []

    times = iter([4_000_000_000, 4_100_000_000, 4_130_000_000])
    monkeypatch.setattr("src.vf_envs.privesc.time.perf_counter_ns", lambda: next(times))

    with pytest.raises(RuntimeError):
        await env.get_model_response(state, cast(vf.Messages, []))

    assert state["privesc"]["assistant_turns"] == 0
    assert state["privesc"]["total_llm_ms_raw"] == pytest.approx(0.0)
    assert state["privesc"]["total_llm_ms_clipped"] == pytest.approx(0.0)

    state["client"] = cast(vf.Client, SuccessClient())
    response = await env.get_model_response(state, cast(vf.Messages, []))
    await env.add_model_response(state, cast(vf.Messages, []), response)

    assert state["privesc"]["assistant_turns"] == 1
    assert state["privesc"]["total_llm_ms_raw"] == pytest.approx(30.0)
    assert state["privesc"]["total_llm_ms_clipped"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_privesc_reward_uses_execution_accounting_not_tool_messages() -> None:
    env = PrivEscVfEnv(
        ssh=SSHConfig(user="user", key_path="/tmp/key", servers="localhost:22"),
        source_cfg=SourceConfig(type="static", scenarios=["01_vuln_suid_gtfo"]),
        scenario_backend="remote_ssh",
        system_prompt="system",
        system_template=None,
        system_template_vars=None,
        start_instruction="start",
        no_tool_calls_nudge="nudge",
        enable_auto_tool_choice=True,
        max_turns=3,
        num_examples=1,
        base_command_timeout=1,
        slow_command_timeout=2,
        max_command_timeout=3,
        reward={"mode": "outcome_round_cost", "c_ref_ms": 500.0},
    )

    state = _base_privesc_state()
    state["privesc"]["got_root"] = True
    state["privesc"]["first_root_turn"] = 1
    state["privesc"]["assistant_turns"] = 1
    state["privesc"]["tool_calls_executed"] = 1
    state["privesc"]["total_llm_ms_raw"] = 50.0
    state["privesc"]["total_tool_ms_raw"] = 50.0
    state["privesc"]["total_llm_ms_clipped"] = 50.0
    state["privesc"]["total_tool_ms_clipped"] = 50.0
    state["completion"] = [
        {
            "role": "assistant",
            "content": "Run once",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": '{"command": "id"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"output": "one"}'},
        {
            "role": "tool",
            "tool_call_id": "call_1_duplicate",
            "content": '{"output": "extra transcript artifact"}',
        },
    ]

    reward = await env.privesc_reward(state)

    expected_round = 1.0 - 1.0 / 3.0
    expected_cost = 1.0 - 100.0 / 500.0
    assert reward == pytest.approx(1.0 + expected_round + 0.1 * expected_cost)
    assert state["privesc"]["reward_metadata"]["tool_calls_executed"] == 1
    assert state["privesc"]["reward_metadata"]["C_ms"] == pytest.approx(100.0)


def test_dataset_info_includes_reward_config() -> None:
    env = PrivEscVfEnv(
        ssh=SSHConfig(user="user", key_path="/tmp/key", servers="localhost:22"),
        source_cfg=SourceConfig(type="static", scenarios=["01_vuln_suid_gtfo"]),
        scenario_backend="remote_ssh",
        system_prompt="system",
        system_template=None,
        system_template_vars=None,
        start_instruction="start",
        no_tool_calls_nudge="nudge",
        enable_auto_tool_choice=True,
        max_turns=3,
        num_examples=1,
        base_command_timeout=1,
        slow_command_timeout=2,
        max_command_timeout=3,
        reward={"mode": "outcome_round"},
    )

    row = env.get_dataset(1)[0]

    assert row["info"]["reward_config"] == {
        "mode": "outcome_round",
        "h_max": 3,
        "lambda_cost": 0.1,
        "c_ref_ms": 540_000.0,
        "llm_ms_clip_ms": 20_000.0,
        "tool_ms_clip_ms": 65_000.0,
        "iface_penalty": 0.05,
    }


def test_dataset_info_includes_scenario_build_index() -> None:
    env = _make_privesc_env(source_cfg=_password_reuse_source())

    row = env.get_dataset(1)[0]

    assert row["info"]["scenario_build_index"] == 0


@pytest.mark.asyncio
async def test_setup_state_uses_dataset_scenario_build_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    source = build_scenario_source(_password_reuse_source())
    prompt_instance = source.build(0)
    drifted_instance = source.build(1)

    assert prompt_instance.config.container_user == "felixbauer"
    assert (
        drifted_instance.config.container_user != prompt_instance.config.container_user
    )

    class DummyScenario:
        def __init__(self, ssh_cfg: SSHConfig, scen_cfg: Any) -> None:
            captured["container_user"] = scen_cfg.container_user
            captured["container_password"] = scen_cfg.container_password
            captured["scenario_name"] = scen_cfg.name
            captured["max_parallel_tool_calls"] = scen_cfg.max_parallel_tool_calls
            self.logger = None

        async def __aenter__(self) -> "DummyScenario":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr("src.vf_envs.privesc.PrivEscScenario", DummyScenario)

    env = _make_privesc_env(
        source_cfg=_password_reuse_source(),
        system_prompt="",
        system_template="privilege_escalation.jinja",
    )
    distractor_env = _make_privesc_env(
        source_cfg=SourceConfig(type="procedural", generators=["password_file"]),
        system_prompt="",
        system_template="privilege_escalation.jinja",
    )
    group = EnvGroup(
        envs=[distractor_env, env],
        env_names=["privesc/password_file", "privesc/password_reuse"],
    )

    row = _get_task_row(group, "privesc/password_reuse")
    prompt = cast(str, row["prompt"][0]["content"])
    expected_access_line = (
        f"- User: '{prompt_instance.config.container_user}' | "
        f"Password: '{prompt_instance.config.container_password}'"
    )

    assert row["example_id"] == 1
    assert row["info"]["scenario_build_index"] == 0
    assert expected_access_line in prompt
    assert "- Turn limit: 3" in prompt

    state = cast(
        vf.State,
        {
            "example_id": row["example_id"],
            "prompt": row["prompt"],
            "info": row["info"],
        },
    )

    await env.setup_state(state)

    assert captured["container_user"] == prompt_instance.config.container_user
    assert captured["container_password"] == prompt_instance.config.container_password
    assert captured["container_user"] != drifted_instance.config.container_user
    assert captured["container_password"] != drifted_instance.config.container_password
    assert state["info"]["scenario_build_index"] == 0
    assert (
        state["info"]["scenario_metadata"]["user"] == prompt_instance.metadata["user"]
    )


@pytest.mark.asyncio
async def test_load_environment_applies_generator_configs_to_dataset_and_setup_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class DummyScenario:
        def __init__(self, ssh_cfg: SSHConfig, scen_cfg: Any) -> None:
            captured["scenario_name"] = scen_cfg.name
            self.logger = None

        async def __aenter__(self) -> "DummyScenario":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr("src.vf_envs.privesc.PrivEscScenario", DummyScenario)

    env = cast(
        PrivEscVfEnv,
        vf.load_environment(
            "src.vf_envs.privesc",
            source_type="procedural",
            generators=["capabilities_gtfobins"],
            generator_configs=_capabilities_generator_configs(),
            system_prompt="",
            system_template="privilege_escalation.jinja",
            start_instruction="start",
            no_tool_calls_nudge="nudge",
            max_turns=3,
            num_examples=1,
        ),
    )
    row = env.get_dataset(1)[0]
    metadata = row["info"]["scenario_metadata"]

    assert metadata["binary_name"] == "php"

    state = cast(
        vf.State,
        {
            "example_id": row["example_id"],
            "prompt": row["prompt"],
            "info": row["info"],
        },
    )

    await env.setup_state(state)

    runtime_metadata = state["info"]["scenario_metadata"]
    assert runtime_metadata["binary_name"] == "php"
    assert captured["scenario_name"] == row["info"]["scenario_config"]["name"]
