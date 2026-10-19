import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast

import pytest

pytest.importorskip("verifiers")

import verifiers as vf  # noqa: E402
import yaml

from src.vf_envs.privesc import PrivEscVfEnv


PROJECT_ROOT = Path(__file__).parents[2]
_PROMPT_DEFAULTS = yaml.safe_load(
    (PROJECT_ROOT / "conf" / "prompts" / "default.yaml").read_text()
)
SFT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SFT_LORA_NAME = "privesc-sft-checkpoint-375"


def _require_e2e_opt_in() -> None:
    if os.getenv("RUN_PRIVESC_E2E") != "1":
        pytest.skip("Set RUN_PRIVESC_E2E=1 to run PrivEsc end-to-end integration tests")


def _require_ssh_env() -> None:
    if not os.getenv("PRIVESC_KEY"):
        raise ValueError("Set PRIVESC_KEY (run `source .env`) ")


def _require_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the local vLLM integration test")


def _require_local_sft_adapter() -> Path:
    adapter_path = os.getenv("PRIVESC_SFT_ADAPTER_DIR")
    if not adapter_path:
        pytest.skip("Set PRIVESC_SFT_ADAPTER_DIR to run the local SFT adapter test")

    adapter_dir = Path(adapter_path).expanduser()
    if not adapter_dir.is_absolute():
        adapter_dir = PROJECT_ROOT / adapter_dir
    if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").is_file():
        return adapter_dir
    pytest.skip(f"Local SFT adapter checkpoint not found: {adapter_dir}")


def _get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_log_tail(log_path: Path) -> str:
    if not log_path.exists():
        return "<no log file>"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text[-8000:]


def _wait_for_http_ready(url: str, proc: subprocess.Popen[str], log_path: Path) -> None:
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(
                f"vLLM exited before becoming ready.\n{_read_log_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(2)
    pytest.fail(f"Timed out waiting for vLLM health.\n{_read_log_tail(log_path)}")


def _post_json(url: str, payload: dict[str, Any]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} from {url}")


def _load_lora_adapter(endpoint: str, adapter_dir: Path, log_path: Path) -> None:
    deadline = time.monotonic() + 120.0
    payload = {"lora_name": SFT_LORA_NAME, "lora_path": str(adapter_dir.resolve())}
    url = f"{endpoint}/load_lora_adapter"
    while time.monotonic() < deadline:
        try:
            _post_json(url, payload)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in {404, 500}:
                raise
        except urllib.error.URLError:
            pass
        time.sleep(2)
    pytest.fail(f"Timed out loading LoRA adapter.\n{_read_log_tail(log_path)}")


@contextmanager
def _local_vllm_server(tmp_path: Path, adapter_dir: Path) -> Iterator[tuple[str, str]]:
    log_path = tmp_path / "vllm-privesc-integration.log"
    port = _get_free_port()
    endpoint = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    pythonpath_parts = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "external" / "prime-rl" / "src"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    command = [
        sys.executable,
        "-m",
        "prime_rl.inference.server",
        "--model.name",
        SFT_BASE_MODEL,
        "--server.port",
        str(port),
        "--model.max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.85",
        "--model.tool-call-parser",
        "hermes",
        "--max-lora-rank",
        "64",
        "--api-server-count",
        "1",
        "--enable-lora",
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_http_ready(f"{endpoint}/health", proc, log_path)
            _load_lora_adapter(endpoint, adapter_dir, log_path)
            yield f"{endpoint}/v1", SFT_LORA_NAME
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_verifiers_privesc_env_outcome_round_cost_with_local_sft_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _require_e2e_opt_in()
    try:
        _require_ssh_env()
    except ValueError as e:
        pytest.skip(f"Environment not configured: {e}")

    _require_cuda()
    adapter_dir = _require_local_sft_adapter()

    max_turns = 3
    env = cast(
        PrivEscVfEnv,
        vf.load_environment(
            "src.vf_envs.privesc",
            source_type="static",
            scenarios=["08_root_password_reuse"],
            system_prompt="",
            system_template="privilege_escalation.jinja",
            start_instruction=_PROMPT_DEFAULTS["start_instruction"],
            no_tool_calls_nudge=_PROMPT_DEFAULTS["no_tool_calls_nudge"],
            max_turns=max_turns,
            num_examples=1,
            reward={
                "mode": "outcome_round_cost",
                "c_ref_ms": 20_000.0,
                "llm_ms_clip_ms": 5_000.0,
                "tool_ms_clip_ms": 5_000.0,
            },
        ),
    )

    row = env.get_dataset(1)[0]
    input_obj = cast(
        vf.RolloutInput,
        {
            "prompt": row["prompt"],
            "example_id": 0,
            "answer": "",
            "info": row.get("info", {}),
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "default")

    with _local_vllm_server(tmp_path, adapter_dir) as (api_base_url, model_name):
        client = vf.ClientConfig(
            client_type="openai_chat_completions",
            api_key_var="OPENAI_API_KEY",
            api_base_url=api_base_url,
        )
        output = await env.run_rollout(
            input_obj,
            client=client,
            model=model_name,
            sampling_args={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
        )

    metrics = cast(dict[str, float], output["metrics"])
    success = metrics["success_metric"]
    cost_ms = metrics["reward_C_ms"]
    llm_ms = metrics["reward_total_llm_ms_clipped"]
    tool_ms = metrics["reward_total_tool_ms_clipped"]
    expected_r_out = 2.0 * success - 1.0
    expected_round = success * max(
        0.0, 1.0 - metrics["turns_to_root_metric"] / float(max_turns)
    )
    expected_cost = success * (1.0 - min(1.0, cost_ms / 20_000.0))

    assert success in {0.0, 1.0}
    assert metrics["got_root_metric"] == pytest.approx(success)
    assert metrics["privesc_reward"] == pytest.approx(float(output["reward"]))
    assert metrics["reward_R_out"] == pytest.approx(expected_r_out)
    assert llm_ms > 0.0
    assert tool_ms >= 0.0
    assert cost_ms == pytest.approx(llm_ms + tool_ms)
    assert metrics["reward_R_round"] == pytest.approx(expected_round)
    assert metrics["reward_R_cost"] == pytest.approx(expected_cost)
    assert float(output["reward"]) == pytest.approx(
        expected_r_out
        + expected_round
        + 0.1 * expected_cost
        + metrics["reward_R_iface"]
    )
