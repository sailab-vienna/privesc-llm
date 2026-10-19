"""Verifiers environment for privilege escalation (Prime-RL integration).

Prime-RL loads this environment via `verifiers.load_environment("src.vf_envs.privesc", ...)`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict, replace
from hashlib import sha1
from typing import Any, Callable, cast

import verifiers as vf
from datasets import Dataset
from verifiers.types import (
    AssistantMessage,
    Response,
    SamplingArgs,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from verifiers.utils.message_utils import maybe_normalize_messages

from src.config import (
    PrivEscRewardConfig,
    RewardConfigLike,
    ScenarioConfig,
    SourceConfig,
    SSHConfig,
    resolve_default_host_path,
    resolve_reward_config,
)
from src.gym.prompts import render_system_prompt_from_template
from src.gym.scenario import PrivEscScenario
from src.parsers.qwen3 import tool_call_parse_error_payload
from src.parsers.tool_calls import (
    ToolCallParser,
    parse_tool_calls_detailed,
    validate_tool_call_parser,
)
from src.vf_envs.reward import calculate_privesc_reward
from src.scenarios import build_scenario_source
from src.tui.agent_panels import AgentLogger

log = logging.getLogger("privesc_vf")


def _redact_scenario_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    password = cfg.pop("container_password", None)
    if isinstance(password, str):
        cfg["container_password_len"] = len(password)
        cfg["container_password_fp"] = sha1(password.encode("utf-8")).hexdigest()[:10]
    return cfg


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (
        position - lower
    )


def _verifier_tool_calls(
    calls: list[Any], make_id: Callable[[int], str]
) -> list[ToolCall]:
    return [
        ToolCall(id=make_id(index), name=str(call.name), arguments=str(call.arguments))
        for index, call in enumerate(calls)
    ]


class _AgentLoggerAdapter:
    def __init__(self, logger: Any):
        self._log = logger

    def session_start(
        self, model: str, max_turns: int, metadata: dict[str, Any] | None = None
    ) -> None:
        self._log.info(
            f"session_start model={model} max_turns={max_turns} meta={metadata or {}}"
        )

    def system_prompt(self, content: str) -> None:
        self._log.info(f"system_prompt len={len(content)}")

    def assistant(
        self, content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        tool_names: list[str] = []
        for call in tool_calls or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name:
                tool_names.append(name)
        self._log.info(f"assistant tool_calls={tool_names} text_len={len(content)}")

    def nudge(self, message: str) -> None:
        self._log.info(f"nudge {message}")

    def info(self, message: str) -> None:
        self._log.info(message)

    def tool_result(
        self, name: str, result: dict[str, Any], duration_ms: int | None = None
    ) -> None:
        extra = f" duration_ms={duration_ms}" if duration_ms is not None else ""
        got_root = result.get("got_root")
        self._log.info(f"tool_result name={name} got_root={got_root}{extra}")

    def reset_attempt(self) -> None:
        self._log.info("reset_attempt")

    def session_end(
        self,
        success: bool,
        turns: int,
        tokens: int = 0,
        cost: float = 0.0,
        *,
        rejected: bool = False,
        message: str | None = None,
    ) -> None:
        self._log.info(
            f"session_end success={success} turns={turns} tokens={tokens} cost={cost}"
        )

    def set_token_source(self, source: Any) -> None:
        self._log.debug(f"set_token_source {type(source)}")


def _ssh_from_env() -> SSHConfig:
    home = os.getenv("HOME", "")
    return SSHConfig(
        user=os.getenv("PRIVESC_USER", "root"),
        key_path=os.getenv("PRIVESC_KEY", os.path.join(home, ".ssh", "id_rsa")),
        servers=os.getenv("PRIVESC_SSH_SERVERS", ""),
        host_path=os.getenv(
            "PRIVESC_HOST_PATH",
            resolve_default_host_path(home),
        ),
    )


def _build_dataset(
    *,
    source_cfg: SourceConfig,
    source: Any,
    scenario_backend: str,
    system_prompt: str,
    system_template: str | None,
    system_template_vars: dict[str, Any] | None,
    start_instruction: str,
    num_examples: int,
    max_turns: int,
    base_command_timeout: int,
    slow_command_timeout: int,
    max_command_timeout: int,
    reward_config: PrivEscRewardConfig,
) -> Dataset:
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")

    if source_cfg.type == "static":
        items = source_cfg.scenarios
    elif source_cfg.type == "procedural":
        items = source_cfg.generators
    else:
        raise ValueError(f"Unknown source type: {source_cfg.type}")

    if not items:
        raise ValueError(f"source config missing items for type={source_cfg.type}")

    rows: list[dict[str, Any]] = []
    for i in range(num_examples):
        instance = source.build(i)
        instance.config.backend = scenario_backend
        instance.config.base_command_timeout = base_command_timeout
        instance.config.slow_command_timeout = slow_command_timeout
        instance.config.max_command_timeout = max_command_timeout

        rendered_system_prompt = system_prompt
        if system_template is not None:
            scen_cfg = instance.config
            vars_dict: dict[str, Any] = dict(system_template_vars or {})
            vars_dict.update(
                {
                    "user": scen_cfg.container_user,
                    "password": scen_cfg.container_password,
                    "max_turns": max_turns,
                    "term_cols": scen_cfg.term_cols,
                    "term_rows": scen_cfg.term_rows,
                }
            )
            rendered_system_prompt = render_system_prompt_from_template(
                system_template, vars_dict
            )

        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": rendered_system_prompt},
                    {"role": "user", "content": start_instruction},
                ],
                "answer": "",
                "info": {
                    "scenario_build_index": i,
                    "scenario_config": _redact_scenario_config(asdict(instance.config)),
                    "scenario_metadata": dict(instance.metadata),
                    "scenario_id": instance.id,
                    "reward_config": reward_config.to_dict(),
                },
            }
        )
    return Dataset.from_list(rows)


class PrivEscVfEnv(vf.StatefulToolEnv):
    _META_METRIC_KEYS: list[str] = [
        "R_out",
        "R_round",
        "R_cost",
        "weighted_R_cost",
        "R_repeat",
        "R_iface",
        "final_reward",
        "C_ms",
        "assistant_turns",
        "tool_calls_executed",
        "total_llm_ms_raw",
        "total_tool_ms_raw",
        "total_llm_ms_clipped",
        "total_tool_ms_clipped",
        "H_root",
        "interface_violation",
        "no_tool_call_count",
        "malformed_tool_call_count",
        "invalid_tool_name_count",
        "invalid_tool_args_count",
        "missing_reasoning_count",
        "turn_number",
        "total_messages",
        "total_valid_tool_calls",
        "total_repetitions",
        "total_short_content",
        "total_tool_errors",
        "total_recon_commands",
        "total_wasted_turns",
        "speed_bonus",
        "recon_bonus",
        "total_penalty",
        "repetition_penalty",
        "tool_error_penalty",
        "no_tool_calls_penalty",
        "short_content_penalty",
        "structured_no_round_final_reward",
        "structured_final_reward",
    ]

    def __init__(
        self,
        *,
        ssh: SSHConfig,
        source_cfg: SourceConfig,
        generator_configs: dict[str, Any] | None = None,
        scenario_backend: str,
        system_prompt: str,
        system_template: str | None,
        system_template_vars: dict[str, Any] | None,
        start_instruction: str,
        no_tool_calls_nudge: str,
        enable_auto_tool_choice: bool,
        tool_call_parser: ToolCallParser = "hermes",
        max_turns: int,
        num_examples: int,
        base_command_timeout: int,
        slow_command_timeout: int,
        max_command_timeout: int,
        max_parallel_tool_calls: int | None = None,
        reward: RewardConfigLike = None,
        **kwargs: Any,
    ) -> None:
        self._ssh = ssh
        self._source = build_scenario_source(
            source_cfg,
            generators_cfg=generator_configs,
        )
        self._scenario_backend = scenario_backend
        self._max_turns = max_turns
        self._no_tool_calls_nudge = no_tool_calls_nudge
        self._enable_auto_tool_choice = bool(enable_auto_tool_choice)
        self._tool_call_parser = validate_tool_call_parser(tool_call_parser)
        self._base_command_timeout = base_command_timeout
        self._slow_command_timeout = slow_command_timeout
        self._max_command_timeout = max_command_timeout
        self._max_parallel_tool_calls = (
            ScenarioConfig().max_parallel_tool_calls
            if max_parallel_tool_calls is None
            else max_parallel_tool_calls
        )
        self._reward_config = replace(resolve_reward_config(reward), h_max=max_turns)
        self._sanity_logged = False

        def dataset() -> Dataset:
            return _build_dataset(
                source_cfg=source_cfg,
                source=self._source,
                scenario_backend=scenario_backend,
                system_prompt=system_prompt,
                system_template=system_template,
                system_template_vars=system_template_vars,
                start_instruction=start_instruction,
                num_examples=num_examples,
                max_turns=max_turns,
                base_command_timeout=base_command_timeout,
                slow_command_timeout=slow_command_timeout,
                max_command_timeout=max_command_timeout,
                reward_config=self._reward_config,
            )

        super().__init__(
            dataset=dataset,
            system_prompt=None,
            max_turns=max_turns,
            **kwargs,
        )

        self.add_tool(self.exec_command, args_to_skip=["scenario"])
        self.add_tool(self.test_credentials, args_to_skip=["scenario"])

        self.rubric.add_reward_func(self.privesc_reward)
        for func in [
            self.success_metric,
            self.got_root_metric,
            self.turns_to_root_metric,
            self.successful_cost_ms,
            self.successful_llm_call_ms_raw_p99,
            self.successful_tool_call_ms_raw_p99,
            self.invalid_tool_calls_total_metric,
            self.invalid_tool_calls_invalid_json_metric,
            self.invalid_tool_calls_missing_name_metric,
            self.invalid_tool_calls_invalid_arguments_metric,
            self.invalid_tool_calls_invalid_shape_metric,
        ]:
            self.rubric.add_metric(func)

        for _key in self._META_METRIC_KEYS:

            async def _metric(state: vf.State, k: str = _key) -> float:
                return self._reward_meta(state, k)

            _metric.__name__ = f"reward_{_key}"
            self.rubric.add_metric(_metric)

    async def add_trajectory_step(
        self, state: vf.State, trajectory_step: vf.TrajectoryStep
    ) -> None:
        completion = trajectory_step.get("completion")
        if isinstance(completion, list) and completion:
            traj_id = state.get("trajectory_id")
            if not isinstance(traj_id, str):
                traj_id = "traj"
            turn_idx = len(state.get("trajectory") or [])

            normalized: list[Any] = []
            for message in completion:
                if not isinstance(message, AssistantMessage):
                    normalized.append(message)
                    continue
                if message.tool_calls:
                    normalized.append(message)
                    continue
                content = message.content
                if not isinstance(content, str) or "<tool_call>" not in content:
                    normalized.append(message)
                    continue

                content_wo_calls, calls, _ = parse_tool_calls_detailed(
                    content, parser=self._tool_call_parser
                )
                if not calls:
                    normalized.append(message)
                    continue

                tool_calls = _verifier_tool_calls(
                    calls, lambda index: f"call_{traj_id}_{turn_idx}_{index}"
                )
                normalized.append(
                    message.model_copy(
                        update={"content": content_wo_calls, "tool_calls": tool_calls}
                    )
                )

            trajectory_step = cast(
                vf.TrajectoryStep,
                {**trajectory_step, "completion": cast(vf.Messages, normalized)},
            )

        await super().add_trajectory_step(state, trajectory_step)

    async def setup_state(self, state: vf.State, **kwargs: Any) -> None:
        await super().setup_state(state, **kwargs)
        sampling_args = state.get("sampling_args")
        if isinstance(sampling_args, dict) and "tool_choice" not in sampling_args:
            sampling_args["tool_choice"] = (
                "auto" if self._enable_auto_tool_choice else "none"
            )

        state["privesc"] = {
            "got_root": False,
            "first_root_turn": None,
            "scenario": None,
            "reward_metadata": None,
            "assistant_turns": 0,
            "tool_calls_executed": 0,
            "total_llm_ms_raw": 0.0,
            "total_tool_ms_raw": 0.0,
            "total_llm_ms_clipped": 0.0,
            "total_tool_ms_clipped": 0.0,
            "llm_call_ms_raw": [],
            "tool_call_ms_raw": [],
        }

        example_id = state.get("example_id")
        if not isinstance(example_id, int):
            raise ValueError("State missing example_id")
        state_info = state.get("info")
        build_index = example_id
        if isinstance(state_info, dict):
            idx = state_info.get("scenario_build_index")
            if isinstance(idx, int):
                build_index = idx
        instance = self._source.build(build_index)
        scen_cfg = instance.config
        scen_cfg.backend = self._scenario_backend
        scen_cfg.base_command_timeout = self._base_command_timeout
        scen_cfg.slow_command_timeout = self._slow_command_timeout
        scen_cfg.max_command_timeout = self._max_command_timeout
        scen_cfg.max_parallel_tool_calls = self._max_parallel_tool_calls

        info = state_info if isinstance(state_info, dict) else {}
        info.update(
            {
                "scenario_build_index": build_index,
                "scenario_id": instance.id,
                "scenario_metadata": dict(instance.metadata),
                "scenario_config": _redact_scenario_config(asdict(instance.config)),
                "reward_config": self._reward_config.to_dict(),
            }
        )
        state["info"] = info

        generator_name = instance.metadata.get("generator_name")
        if isinstance(generator_name, str) and generator_name:
            state["task"] = f"privesc/{generator_name}"

        self._run_prompt_sanity_check(state, scen_cfg)

        scenario = PrivEscScenario(self._ssh, scen_cfg)
        scenario.logger = cast(AgentLogger, _AgentLoggerAdapter(self.logger))
        await scenario.__aenter__()
        state["privesc"]["scenario"] = scenario

    @vf.cleanup
    async def cleanup_scenario(self, state: vf.State) -> None:
        privesc = state.get("privesc") or {}
        scenario = privesc.get("scenario")
        if scenario is None:
            return
        await scenario.__aexit__(None, None, None)
        privesc["scenario"] = None

    @vf.stop(priority=50)
    async def got_root(self, state: vf.State) -> bool:
        return bool(self._privesc_state(state).get("got_root"))

    async def no_tools_called(self, state: vf.State) -> bool:
        """Disable ToolEnv's built-in stop-on-no-tool-calls behavior.

        When the model returns no tool calls we issue a nudge message via
        env_response instead of terminating the episode early.
        """
        return False

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        # Keep this tiny wrapper because ToolEnv stops on missing tool calls, while
        # PrivEsc nudges the model and needs the full terminating tool response saved.
        messages = maybe_normalize_messages(messages, field_name="env_response")
        last = messages[-1]
        if not isinstance(last, AssistantMessage):
            raise TypeError(
                f"Expected AssistantMessage in env_response, got {type(last).__name__}"
            )

        invalid_tool_messages: list[ToolMessage] = []
        if not last.tool_calls:
            if isinstance(last.content, str) and "<tool_call>" in last.content:
                content_wo_calls, calls, errors = parse_tool_calls_detailed(
                    last.content, parser=self._tool_call_parser
                )
                if errors:
                    invalid_tool_messages = self._build_invalid_tool_call_messages(
                        errors
                    )
                if calls:
                    parsed_tool_calls = _verifier_tool_calls(
                        calls, lambda _index: f"call_{uuid.uuid4().hex[:8]}"
                    )
                    last = last.model_copy(
                        update={
                            "content": content_wo_calls,
                            "tool_calls": parsed_tool_calls,
                        }
                    )
                    messages = [*messages[:-1], last]

            if not last.tool_calls:
                if invalid_tool_messages:
                    return cast(vf.Messages, invalid_tool_messages)
                return cast(
                    vf.Messages, [UserMessage(content=self._no_tool_calls_nudge)]
                )

        tool_messages = await super().env_response(messages, state, **kwargs)
        if invalid_tool_messages:
            tool_messages = [*tool_messages, *invalid_tool_messages]
        if (
            self._privesc_state(state).get("got_root")
            and state.get("final_env_response") is None
        ):
            state["final_env_response"] = tool_messages
        return tool_messages

    def _build_invalid_tool_call_messages(self, errors: list[Any]) -> list[ToolMessage]:
        return [
            ToolMessage(
                tool_call_id=f"invalid_{uuid.uuid4().hex[:8]}",
                content=json.dumps(
                    tool_call_parse_error_payload(error),
                    ensure_ascii=False,
                ),
            )
            for error in errors
        ]

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        privesc = state.get("privesc") or {}
        scenario = privesc.get("scenario")
        if scenario is None:
            raise RuntimeError("Scenario is not initialized")
        tool_args["scenario"] = scenario
        # smuggled through tool_args because base call_tool has no state param
        tool_args["_vf_state"] = state
        return tool_args

    async def exec_command(self, command: str, scenario: PrivEscScenario) -> str:
        result = await scenario.exec_command(command)
        log.debug("exec_command exit=%s got_root=%s", result.exit_code, result.got_root)
        return json.dumps(asdict(result), ensure_ascii=False)

    async def get_model_response(
        self,
        state: vf.State,
        prompt: vf.Messages,
        client: vf.Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> Response:
        start_ns = time.perf_counter_ns()
        response = await super().get_model_response(
            state,
            prompt,
            client=client,
            model=model,
            tool_defs=tool_defs,
            sampling_args=sampling_args,
        )
        self._record_llm_latency(state, time.perf_counter_ns() - start_ns)
        privesc = self._privesc_state(state)
        privesc["assistant_turns"] = int(privesc.get("assistant_turns", 0)) + 1
        return response

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs: Any
    ) -> ToolMessage:
        state = cast(vf.State, tool_args.pop("_vf_state"))
        if tool_name not in self.tool_map:
            return await super().call_tool(tool_name, tool_args, tool_call_id, **kwargs)
        # Match gym tool accounting: only count/timestamp tool executions that return
        # successfully from the awaited super().call_tool(...) boundary.
        start_ns = time.perf_counter_ns()
        tool_message = await super().call_tool(
            tool_name, tool_args, tool_call_id, **kwargs
        )
        self._record_tool_execution(state, time.perf_counter_ns() - start_ns)
        self._maybe_mark_root_from_tool_message(state, tool_message)
        return tool_message

    async def test_credentials(
        self, user: str, password: str, scenario: PrivEscScenario
    ) -> str:
        result = await scenario.test_credentials(user, password)
        log.debug(
            "test_credentials success=%s got_root=%s", result.success, result.got_root
        )
        return json.dumps(asdict(result), ensure_ascii=False)

    async def privesc_reward(self, state: vf.State) -> float:
        privesc = self._privesc_state(state)
        got_root = bool(privesc.get("got_root"))
        prompt = state.get("prompt")
        completion = state.get("completion")
        if not isinstance(prompt, list):
            prompt = []
        if not isinstance(completion, list):
            completion = []
        trace = list(prompt) + list(completion)

        result = calculate_privesc_reward(
            trace_history=trace,
            got_root=got_root,
            max_turns=self._max_turns,
            reward_config=self._reward_config,
            episode_metrics={
                "total_llm_ms_clipped": privesc.get("total_llm_ms_clipped"),
                "total_tool_ms_clipped": privesc.get("total_tool_ms_clipped"),
                "assistant_turns": privesc.get("assistant_turns"),
                "first_root_turn": privesc.get("first_root_turn"),
                "tool_calls_executed": privesc.get("tool_calls_executed"),
                "total_llm_ms_raw": privesc.get("total_llm_ms_raw"),
                "total_tool_ms_raw": privesc.get("total_tool_ms_raw"),
            },
            tool_call_parser=self._tool_call_parser,
        )
        privesc["reward_metadata"] = dict(result.metadata)
        return float(result.value)

    async def got_root_metric(self, state: vf.State) -> float:
        return 1.0 if self._privesc_state(state).get("got_root") else 0.0

    async def success_metric(self, state: vf.State) -> float:
        return await self.got_root_metric(state)

    async def turns_to_root_metric(self, state: vf.State) -> float:
        turns = self._privesc_state(state).get("first_root_turn")
        return float(turns) if isinstance(turns, int) else 0.0

    async def successful_cost_ms(self, state: vf.State) -> float:
        privesc = self._privesc_state(state)
        if not privesc.get("got_root"):
            return 0.0
        return float(privesc.get("total_llm_ms_clipped", 0.0)) + float(
            privesc.get("total_tool_ms_clipped", 0.0)
        )

    async def successful_llm_call_ms_raw_p99(self, state: vf.State) -> float:
        privesc = self._privesc_state(state)
        if not privesc.get("got_root"):
            return 0.0
        return _quantile(
            [float(value) for value in privesc.get("llm_call_ms_raw", [])], 0.99
        )

    async def successful_tool_call_ms_raw_p99(self, state: vf.State) -> float:
        privesc = self._privesc_state(state)
        if not privesc.get("got_root"):
            return 0.0
        return _quantile(
            [float(value) for value in privesc.get("tool_call_ms_raw", [])], 0.99
        )

    def _privesc_state(self, state: vf.State) -> dict[str, Any]:
        pr = state.get("privesc")
        return pr if isinstance(pr, dict) else {}

    def _find_system_msg(self, state: vf.State) -> dict[str, Any] | None:
        for key in ("prompt", "messages"):
            msgs = state.get(key)
            if isinstance(msgs, list) and msgs:
                first = msgs[0]
                if isinstance(first, dict) and first.get("role") == "system":
                    return cast(dict[str, Any], first)
        return None

    def _run_prompt_sanity_check(self, state: vf.State, scen_cfg: Any) -> None:
        # Ensure the system prompt matches per-scenario vars.
        # Use ToolEnv's logger so Prime-RL captures it.
        prompt_msg = self._find_system_msg(state)
        if not self._sanity_logged:
            self.logger.info(
                "state_prompt_fields prompt=%s messages=%s info=%s",
                type(state.get("prompt")).__name__,
                type(state.get("messages")).__name__,
                type(state.get("info")).__name__,
            )
        if prompt_msg is None:
            return
        content = prompt_msg.get("content")
        if not isinstance(content, str):
            return
        needle = f"- User: '{scen_cfg.container_user}' | Password: '{scen_cfg.container_password}'"
        ok = needle in content
        if not self._sanity_logged:
            pw_fp = sha1(scen_cfg.container_password.encode()).hexdigest()[:10]
            self.logger.info(
                "prompt_vars_ok=%s user=%s pass_len=%d pass_fp=%s scenario=%s",
                ok,
                scen_cfg.container_user,
                len(scen_cfg.container_password),
                pw_fp,
                scen_cfg.name,
            )
        self._sanity_logged = True
        if not ok:
            raise ValueError(
                "System prompt credentials do not match scenario config; "
                "dynamic scenario vars are not injected correctly"
            )

    def _clip_latency_ms(self, latency_ms: float, clip_ms: float | None) -> float:
        return latency_ms if clip_ms is None else min(latency_ms, clip_ms)

    def _record_llm_latency(self, state: vf.State, latency_ns: int) -> None:
        privesc = self._privesc_state(state)
        raw_ms = latency_ns / 1_000_000.0
        clipped_ms = self._clip_latency_ms(raw_ms, self._reward_config.llm_ms_clip_ms)
        privesc["total_llm_ms_raw"] = (
            float(privesc.get("total_llm_ms_raw", 0.0)) + raw_ms
        )
        cast(list[float], privesc.setdefault("llm_call_ms_raw", [])).append(raw_ms)
        privesc["total_llm_ms_clipped"] = (
            float(privesc.get("total_llm_ms_clipped", 0.0)) + clipped_ms
        )

    def _record_tool_execution(self, state: vf.State, latency_ns: int) -> None:
        privesc = self._privesc_state(state)
        raw_ms = latency_ns / 1_000_000.0
        clipped_ms = self._clip_latency_ms(raw_ms, self._reward_config.tool_ms_clip_ms)
        privesc["tool_calls_executed"] = int(privesc.get("tool_calls_executed", 0)) + 1
        privesc["total_tool_ms_raw"] = (
            float(privesc.get("total_tool_ms_raw", 0.0)) + raw_ms
        )
        cast(list[float], privesc.setdefault("tool_call_ms_raw", [])).append(raw_ms)
        privesc["total_tool_ms_clipped"] = (
            float(privesc.get("total_tool_ms_clipped", 0.0)) + clipped_ms
        )

    def _maybe_mark_root_from_tool_message(
        self, state: vf.State, tool_message: ToolMessage
    ) -> None:
        content = tool_message.get("content")
        if not isinstance(content, str):
            return
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or payload.get("got_root") is not True:
            return
        privesc = self._privesc_state(state)
        privesc["got_root"] = True
        if isinstance(privesc.get("first_root_turn"), int):
            return
        assistant_turns = privesc.get("assistant_turns")
        if isinstance(assistant_turns, int) and not isinstance(assistant_turns, bool):
            privesc["first_root_turn"] = max(1, assistant_turns)
            return
        trajectory = state.get("trajectory")
        privesc["first_root_turn"] = (
            len(trajectory) + 1 if isinstance(trajectory, list) else 1
        )

    def _reward_meta(self, state: vf.State, key: str, default: float = 0.0) -> float:
        meta = self._privesc_state(state).get("reward_metadata")
        if not isinstance(meta, dict):
            return default
        value = meta.get(key, default)
        if isinstance(value, (bool, int, float)):
            return float(value)
        return default

    def _invalid_tool_call_count(
        self, state: vf.State, kind: str | None = None
    ) -> float:
        meta = self._privesc_state(state).get("reward_metadata")
        if not isinstance(meta, dict):
            return 0.0
        key = (
            "total_invalid_tool_calls"
            if kind is None
            else f"total_invalid_tool_calls_{kind}"
        )
        value = meta.get(key, 0)
        return float(value) if isinstance(value, (int, float)) else 0.0

    async def invalid_tool_calls_total_metric(self, state: vf.State) -> float:
        return self._invalid_tool_call_count(state)

    async def invalid_tool_calls_invalid_json_metric(self, state: vf.State) -> float:
        return self._invalid_tool_call_count(state, "invalid_json")

    async def invalid_tool_calls_missing_name_metric(self, state: vf.State) -> float:
        return self._invalid_tool_call_count(state, "missing_name")

    async def invalid_tool_calls_invalid_arguments_metric(
        self, state: vf.State
    ) -> float:
        return self._invalid_tool_call_count(state, "invalid_arguments")

    async def invalid_tool_calls_invalid_shape_metric(self, state: vf.State) -> float:
        return self._invalid_tool_call_count(state, "invalid_shape")


def load_environment(
    *,
    source_type: str = "procedural",
    generators: list[str] | None = None,
    generator_configs: dict[str, Any] | None = None,
    scenarios: list[str] | None = None,
    seed: int = 42,
    random_seed: bool = False,
    scenario_backend: str = "remote_ssh",
    max_turns: int = 10,
    system_prompt: str = "",
    system_template: str | None = None,
    system_template_vars: dict[str, Any] | None = None,
    start_instruction: str = "",
    no_tool_calls_nudge: str | None = None,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: ToolCallParser = "hermes",
    tools: Any | None = None,  # accepted for API compatibility; not used
    num_examples: int = 64,
    base_command_timeout: int = 1,
    slow_command_timeout: int = 10,
    max_command_timeout: int = 90,
    max_parallel_tool_calls: int | None = None,
    reward: RewardConfigLike = None,
    **kwargs: Any,
) -> vf.Environment:
    if source_type not in {"procedural", "static"}:
        raise ValueError("source_type must be procedural or static")

    if source_type == "procedural":
        if not generators:
            raise ValueError("procedural source requires generators")
        source_cfg = SourceConfig(
            type="procedural",
            generators=list(generators),
            seed=seed,
            random_seed=bool(random_seed),
        )
    else:
        if not scenarios:
            raise ValueError("static source requires scenarios")
        source_cfg = SourceConfig(type="static", scenarios=list(scenarios))

    ssh = _ssh_from_env()

    if no_tool_calls_nudge is None:
        raise ValueError("no_tool_calls_nudge is required")
    return PrivEscVfEnv(
        ssh=ssh,
        source_cfg=source_cfg,
        generator_configs=generator_configs,
        scenario_backend=scenario_backend,
        system_prompt=system_prompt,
        system_template=system_template,
        system_template_vars=system_template_vars,
        start_instruction=start_instruction,
        no_tool_calls_nudge=no_tool_calls_nudge,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tool_call_parser=validate_tool_call_parser(tool_call_parser),
        max_turns=max_turns,
        num_examples=num_examples,
        base_command_timeout=base_command_timeout,
        slow_command_timeout=slow_command_timeout,
        max_command_timeout=max_command_timeout,
        max_parallel_tool_calls=max_parallel_tool_calls,
        reward=reward,
        **kwargs,
    )
