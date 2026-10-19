from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.config import _DEFAULT_RL_GENERATORS, register_with_hydra


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
PAPER_REWARD_DEFAULTS = {
    "h_max": 20,
    "lambda_cost": 0.1,
    "c_ref_ms": 540_000,
    "llm_ms_clip_ms": 20_000,
    "tool_ms_clip_ms": 65_000,
    "iface_penalty": 0.05,
}


def _compose_experiment(experiment_id: str, overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    register_with_hydra()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(
            config_name="config",
            overrides=[f"+experiment={experiment_id}", *(overrides or [])],
        )


@pytest.mark.parametrize(
    ("experiment_id", "reward_mode", "output_dir_suffix"),
    [
        (
            "train/paper_prime_rl_reward_outcome",
            "outcome",
            "/paper_prime_rl_reward_outcome",
        ),
        (
            "train/paper_prime_rl_reward_outcome_cost",
            "outcome_cost",
            "/paper_prime_rl_reward_outcome_cost",
        ),
        (
            "train/paper_prime_rl_reward_outcome_round",
            "outcome_round",
            "/paper_prime_rl_reward_outcome_round",
        ),
        (
            "train/paper_prime_rl_reward_outcome_round_cost",
            "outcome_round_cost",
            "/paper_prime_rl_reward_outcome_round_cost",
        ),
    ],
)
def test_reward_ablation_experiment_resolves(
    experiment_id: str, reward_mode: str, output_dir_suffix: str
):
    cfg = _compose_experiment(experiment_id)

    assert cfg.reward.mode == reward_mode
    assert str(cfg.rl.prime_rl.output_dir).endswith(output_dir_suffix)
    for key, expected in PAPER_REWARD_DEFAULTS.items():
        assert cfg.reward[key] == pytest.approx(expected)
    assert cfg.reward.h_max == cfg.rl.prime_rl.max_turns


def test_outcome_round_cost_experiment_resolves_secondary_cost_defaults():
    cfg = _compose_experiment("train/paper_prime_rl_reward_outcome_round_cost")

    assert cfg.reward.mode == "outcome_round_cost"
    for key, expected in PAPER_REWARD_DEFAULTS.items():
        assert cfg.reward[key] == pytest.approx(expected)
    assert cfg.reward.h_max == cfg.rl.prime_rl.max_turns


def test_paper_qwen3_4b_sft_experiment_uses_canonical_defaults():
    cfg = _compose_experiment("train/paper_qwen3_4b_sft")

    assert (
        str(cfg.datasets.sft.output_dir)
        == "outputs/data/privesc_sft/standard/unguided/deepseek/long_reasoning/training"
    )
    assert (
        str(cfg.datasets.sft.validation_output_dir)
        == "outputs/data/privesc_sft/standard/unguided/deepseek/long_reasoning/validation"
    )
    assert cfg.sft.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert str(cfg.sft.output_dir).endswith("/paper_qwen3_4b_sft")
    assert cfg.sft.max_seq_length == 32768
    assert cfg.sft.learning_rate == pytest.approx(1.5e-4)
    assert cfg.sft.lora_rank == 8
    assert cfg.sft.lora_alpha == pytest.approx(32)
    assert cfg.sft.batch_size == 8
    assert cfg.sft.num_train_epochs == 10
    assert cfg.sft.seed == 1337
    assert cfg.sft.run_dir is None
    assert cfg.sft.unsloth.load_in_4bit is True
    assert cfg.sft.unsloth.load_in_8bit is False
    assert cfg.sft.unsloth.full_finetuning is False
    assert cfg.sft.unsloth.use_lora is True
    assert cfg.sft.unsloth.per_device_train_batch_size == 2
    assert cfg.sft.unsloth.gradient_accumulation_steps == 4
    assert list(cfg.sft.unsloth.target_modules) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]
    assert cfg.sft.unsloth.use_gradient_checkpointing == "unsloth"
    assert cfg.sft.unsloth.adapter_only_lm_head is True
    assert cfg.sft.unsloth.warmup_steps == 0
    assert cfg.sft.unsloth.eval_strategy == "epoch"
    assert cfg.sft.unsloth.logging_steps == 1
    assert cfg.sft.unsloth.optim == "adamw_8bit"
    assert cfg.sft.unsloth.adam_beta2 == pytest.approx(0.95)
    assert cfg.sft.unsloth.weight_decay == pytest.approx(0.0)
    assert cfg.sft.unsloth.lr_scheduler_type == "linear"
    assert cfg.sft.unsloth.report_to == "wandb"
    assert cfg.sft.unsloth.wandb_project == "privesc-llm-unsloth-sft"
    assert cfg.sft.unsloth.save_strategy == "epoch"
    assert cfg.sft.unsloth.save_steps == 1


def test_paper_qwen3_4b_sft_paths_follow_reasoning_variant():
    cfg = _compose_experiment(
        "train/paper_qwen3_4b_sft",
        overrides=["datasets.sft.reasoning_variant=no_reasoning"],
    )

    assert cfg.datasets.sft.reasoning_variant == "no_reasoning"
    assert (
        str(cfg.datasets.sft.output_dir)
        == "outputs/data/privesc_sft/standard/unguided/deepseek/no_reasoning/training"
    )
    assert (
        str(cfg.datasets.sft.validation_output_dir)
        == "outputs/data/privesc_sft/standard/unguided/deepseek/no_reasoning/validation"
    )


def test_paper_prime_rl_experiment_uses_canonical_defaults():
    cfg = _compose_experiment("train/paper_prime_rl")

    assert cfg.reward.mode == "outcome_round_cost"
    for key, expected in PAPER_REWARD_DEFAULTS.items():
        assert cfg.reward[key] == pytest.approx(expected)
    assert cfg.reward.h_max == cfg.rl.prime_rl.max_turns
    assert str(cfg.rl.prime_rl.output_dir).endswith("/paper_prime_rl")
    assert cfg.rl.prime_rl.base_model == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.rl.prime_rl.init_adapter_path == ""
    assert cfg.rl.prime_rl.max_off_policy_steps == 8
    assert cfg.rl.prime_rl.trainer_loss.kl_tau == pytest.approx(0.0)
    assert cfg.rl.prime_rl.inference_tool_call_parser is None
    assert cfg.rl.prime_rl.inference_enable_auto_tool_choice is False
    assert list(cfg.rl.prime_rl.source.generators) == _DEFAULT_RL_GENERATORS
    assert cfg.rl.prime_rl.ckpt_save_adapter_separately is True
