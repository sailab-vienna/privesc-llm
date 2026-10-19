"""Tests for Prime-RL training config wiring."""

import json

import pytest

pytest.importorskip("prime_rl")
pytest.importorskip("pydantic_config")

from src.config import (
    AgentConfig,
    AppConfig,
    PrimeRLRLConfig,
    PrivEscRewardConfig,
    PromptsConfig,
    RLConfig,
    SSHConfig,
    SourceConfig,
)
from src.rl.prime_rl.train import build_prime_rl_config


@pytest.fixture
def _patch_prompts(monkeypatch):
    monkeypatch.setattr("src.rl.prime_rl.train.render_system_prompt", lambda cfg: "sys")


_DEFAULT_SSH = SSHConfig(user="user", key_path="/tmp/key", servers="localhost:22")
_DEFAULT_AGENT = AgentConfig(
    api_key="k", api_base="https://example.com", model="m", max_turns=10
)
_DEFAULT_PROMPTS = PromptsConfig(
    system_template="{task}",
    start_instruction="Start",
    no_tool_calls_nudge="No tool calls received. `got_root` is still false. Invoke `exec_command` or `test_credentials` using your tool/function calling format.",
    template_vars={"task": "x"},
)


def _make_cfg(
    prime: PrimeRLRLConfig | None = None,
    reward: PrivEscRewardConfig | None = None,
) -> AppConfig:
    if prime is None:
        prime = PrimeRLRLConfig(
            source=SourceConfig(
                type="procedural",
                generators=["password_file", "password_reuse"],
                seed=1337,
            ),
            inference_gpu_ids=[0],
            trainer_gpu_ids=[1],
            use_lora=False,
        )
    elif prime.init_adapter_path is None:
        prime.use_lora = False
        prime.ckpt_save_adapter_separately = False

    if (
        prime is not None
        and prime.inference_gpu_ids == [0]
        and prime.trainer_gpu_ids == [0]
    ):
        prime.trainer_gpu_ids = [1]

    return AppConfig(
        ssh=_DEFAULT_SSH,
        agent=_DEFAULT_AGENT,
        prompts=_DEFAULT_PROMPTS,
        reward=reward or PrivEscRewardConfig(),
        rl=RLConfig(
            backend="prime_rl",
            prime_rl=prime,
        ),
    )


def _expected_reward(mode: str = "outcome_round_cost", *, h_max: int = 20) -> dict:
    data = PrivEscRewardConfig(mode=mode).to_dict()
    data["h_max"] = h_max
    return data


def _write_adapter_config(adapter_dir):
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
            }
        )
    )


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_uses_procedural_source():
    cfg = _make_cfg(
        PrimeRLRLConfig(
            base_model="Qwen/Qwen3-4B-Instruct-2507",
            source=SourceConfig(
                type="procedural",
                generators=["password_file", "password_reuse"],
                seed=1337,
            ),
            max_turns=12,
            batch_size=24,
            rollouts_per_example=4,
            learning_rate=4e-5,
            max_tokens=1024,
            seq_len=4096,
            max_async_level=1,
            max_off_policy_steps=8,
            output_dir="outputs/prime_rl/test",
        )
    )

    config = build_prime_rl_config(cfg)
    env = config.orchestrator.train.env[0]

    assert env.id == "src.vf_envs.privesc"
    assert env.args["source_type"] == "procedural"
    assert env.args["generators"] == ["password_file"]
    assert env.args["seed"] == 1337
    assert env.args["max_turns"] == 12
    assert env.args["no_tool_calls_nudge"]
    assert env.args["reward"] == _expected_reward(h_max=12)
    assert "tools" not in env.args
    assert config.model is not None
    assert config.model.name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.orchestrator.batch_size == 24
    assert config.inference is not None
    assert config.inference.model.tool_call_parser == "auto"
    zero_advantage_filters = [
        f for f in config.orchestrator.filters if f.type == "zero_advantage"
    ]
    assert len(zero_advantage_filters) == 1
    assert zero_advantage_filters[0].enforce is False
    assert config.deployment.type == "single_node"
    assert getattr(config.deployment, "num_infer_gpus") == 1
    assert getattr(config.deployment, "num_train_gpus") == 1


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_maps_gpu_ids_to_deployment_counts():
    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            inference_gpu_ids=[0, 1, 2],
            trainer_gpu_ids=[3, 4],
            output_dir="x",
        )
    )

    config = build_prime_rl_config(cfg)
    assert config.deployment.type == "single_node"
    assert getattr(config.deployment, "gpus_per_node") == 5
    assert getattr(config.deployment, "num_infer_gpus") == 3
    assert getattr(config.deployment, "num_train_gpus") == 2


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_rejects_overlapping_gpu_ids():
    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            inference_gpu_ids=[0, 1],
            trainer_gpu_ids=[1, 2],
            output_dir="x",
        )
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        build_prime_rl_config(cfg)


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_requires_generators():
    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=[]), output_dir="x"
        )
    )
    with pytest.raises(ValueError, match="rl.prime_rl.source.generators"):
        build_prime_rl_config(cfg)


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_maps_init_adapter_path(tmp_path):
    adapter_dir = tmp_path / "adapter"
    _write_adapter_config(adapter_dir)

    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            init_adapter_path=str(adapter_dir),
            output_dir="x",
        )
    )

    config = build_prime_rl_config(cfg)

    assert config.trainer.model.lora is not None
    assert config.trainer.model.lora.rank == 8
    assert config.trainer.model.lora.alpha == 16
    assert config.trainer.model.lora.target_modules == ["q_proj", "v_proj"]
    if hasattr(config.trainer.model.lora, "init_adapter_path"):
        assert config.trainer.model.lora.init_adapter_path == adapter_dir.resolve()


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_saves_lora_adapter_separately_when_enabled(tmp_path):
    adapter_dir = tmp_path / "adapter"
    _write_adapter_config(adapter_dir)

    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            init_adapter_path=str(adapter_dir),
            output_dir="x",
            ckpt_interval=100,
            ckpt_save_adapter_separately=True,
        )
    )

    config = build_prime_rl_config(cfg)

    assert config.ckpt is not None
    assert config.trainer.ckpt is not None
    assert config.trainer.ckpt.weights is not None
    assert config.trainer.ckpt.weights.save_adapter_separately is True


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_creates_checkpoint_config_for_adapter_only_saving(tmp_path):
    adapter_dir = tmp_path / "adapter"
    _write_adapter_config(adapter_dir)

    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            init_adapter_path=str(adapter_dir),
            output_dir="x",
            ckpt_save_adapter_separately=True,
        )
    )

    config = build_prime_rl_config(cfg)

    assert config.ckpt is not None
    assert config.trainer.ckpt is not None
    assert config.trainer.ckpt.weights is not None
    assert config.trainer.ckpt.weights.save_adapter_separately is True


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_rejects_init_adapter_without_lora():
    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            use_lora=False,
            init_adapter_path="/tmp/adapter",
            output_dir="x",
        )
    )

    with pytest.raises(ValueError, match="init_adapter_path"):
        build_prime_rl_config(cfg)


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_rejects_missing_init_adapter_path():
    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            init_adapter_path="/tmp/does-not-exist-privesc-llm",
            output_dir="x",
        )
    )

    with pytest.raises(ValueError, match="does not exist"):
        build_prime_rl_config(cfg)


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_requires_eval_every_for_benchmark():
    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            eval_on_benchmark=True,
            eval_every=0,
            output_dir="x",
        )
    )
    with pytest.raises(ValueError, match="eval_every"):
        build_prime_rl_config(cfg)


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_enables_benchmark_eval(monkeypatch):
    monkeypatch.setattr(
        "src.rl.prime_rl.train.load_benchmark_scenarios",
        lambda: ["01_vuln_suid_gtfo", "02_vuln_password_in_shell_history"],
    )

    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            eval_on_benchmark=True,
            eval_every=7,
            output_dir="x",
        )
    )
    config = build_prime_rl_config(cfg)

    assert config.orchestrator.eval is not None
    assert config.orchestrator.eval.interval == 7
    assert config.orchestrator.eval.num_examples == 2
    eval_env = config.orchestrator.eval.env[0]
    assert eval_env.args["source_type"] == "static"
    assert eval_env.args["no_tool_calls_nudge"]
    assert eval_env.args["reward"] == _expected_reward()
    assert eval_env.args["scenarios"] == [
        "01_vuln_suid_gtfo",
        "02_vuln_password_in_shell_history",
    ]


@pytest.mark.usefixtures("_patch_prompts")
def test_build_prime_rl_config_passes_reward_override_into_env():
    cfg = _make_cfg(
        PrimeRLRLConfig(
            source=SourceConfig(type="procedural", generators=["password_file"]),
            output_dir="x",
        ),
        reward=PrivEscRewardConfig(mode="outcome_round"),
    )

    config = build_prime_rl_config(cfg)
    env = config.orchestrator.train.env[0]

    assert env.args["reward"] == _expected_reward("outcome_round")
