"""Prime-RL training entry point for PrivEsc RL."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import hydra
from hydra.utils import to_absolute_path
from prime_rl.configs.inference import (
    InferenceConfig,
    ModelConfig as InferenceModelConfig,
)
from prime_rl.configs.orchestrator import (
    BufferConfig,
    EvalConfig,
    EvalEnvConfig,
    EvalSamplingConfig,
    OptimizerConfig,
    OrchestratorConfig,
    TrainConfig,
    TrainEnvConfig,
    TrainSamplingConfig,
)
from prime_rl.configs.rl import (
    RLConfig,
    SharedCheckpointConfig,
    SharedModelConfig,
    SharedWandbConfig,
    SingleNodeDeploymentConfig,
)
from prime_rl.configs.shared import LogConfig, TrainerLogConfig
from prime_rl.configs.trainer import (
    ActivationCheckpointConfig,
    AdamWConfig,
    DefaultLossConfig,
    LoRAConfig,
    ModelConfig as TrainerModelConfig,
    SFTLossConfig,
    CheckpointConfig as TrainerCheckpointConfig,
    TokenizerConfig,
    TrainerConfig,
    WeightCheckpointConfig,
)
from prime_rl.entrypoints.rl import rl as run_prime_rl
from src.config import AppConfig, SSHConfig, register_with_hydra, resolve_ssh_endpoints
from src.gym.backends.host_ssh_pool import HostSSHConnectionPool
from src.gym.prompts import render_system_prompt
from src.paths import model_dir_name
from src.rl.benchmark import load_benchmark_scenarios
from src.rl.prime_rl.env_args import build_eval_env_args, build_train_envs
from src.utils.docker_cleanup import run_periodic_cleanup


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
)
log = logging.getLogger("privesc_prime_rl")


def _resolve_optional_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    candidate = str(raw_path).strip()
    if not candidate:
        return None
    return Path(to_absolute_path(candidate))


def _resolve_optional_string(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    candidate = str(raw_value).strip()
    return candidate or None


def _normalize_log_level(raw_log_level: str) -> str:
    log_level = raw_log_level.lower()
    if log_level == "warning":
        return "warn"
    if log_level in {"debug", "info", "warn", "error"}:
        return log_level
    return "info"


def _normalize_target_modules(raw_modules: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for module in raw_modules:
        if module not in seen:
            seen.add(module)
            normalized.append(module)
    return normalized


def _load_adapter_lora_config(init_adapter_path: Path | None) -> dict[str, Any] | None:
    if init_adapter_path is None:
        return None
    adapter_config_path = init_adapter_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise ValueError(
            "rl.prime_rl.init_adapter_path is missing adapter_config.json: "
            f"{init_adapter_path}"
        )

    adapter_config = json.loads(adapter_config_path.read_text())
    if adapter_config.get("peft_type") not in {None, "LORA"}:
        raise ValueError(
            "rl.prime_rl.init_adapter_path must point to a LoRA adapter; "
            f"got peft_type={adapter_config.get('peft_type')}"
        )
    adapter_rank = adapter_config.get("r")
    if not isinstance(adapter_rank, int) or adapter_rank <= 0:
        raise ValueError(
            "rl.prime_rl.init_adapter_path adapter_config.json must contain positive integer r"
        )
    adapter_alpha = adapter_config.get("lora_alpha")
    if not isinstance(adapter_alpha, int | float):
        raise ValueError(
            "rl.prime_rl.init_adapter_path adapter_config.json must contain numeric lora_alpha"
        )
    adapter_target_modules = adapter_config.get("target_modules")
    if not isinstance(adapter_target_modules, list) or not adapter_target_modules:
        raise ValueError(
            "rl.prime_rl.init_adapter_path adapter_config.json must contain target_modules"
        )
    return {
        "rank": adapter_rank,
        "alpha": float(adapter_alpha),
        "target_modules": _normalize_target_modules(
            [str(module) for module in adapter_target_modules]
        ),
    }


def _trainer_fused_lm_head_field_name() -> str | None:
    if "fused_lm_head_chunk_size" in TrainerModelConfig.model_fields:
        return "fused_lm_head_chunk_size"
    if "fused_lm_head_token_chunk_size" in TrainerModelConfig.model_fields:
        return "fused_lm_head_token_chunk_size"
    return None


def _trainer_lora_supports_init_adapter_path() -> bool:
    return "init_adapter_path" in LoRAConfig.model_fields


def _inference_supports_reasoning_parser() -> bool:
    return "reasoning_parser" in InferenceModelConfig.model_fields


def _build_orchestrator_log_config(log_level: str) -> LogConfig:
    log_config_kwargs: dict[str, Any] = {
        "level": log_level,
        "vf_level": log_level,
        "log_data": True,
    }
    if "env_worker_logs" in LogConfig.model_fields:
        log_config_kwargs["env_worker_logs"] = True
    return LogConfig.model_validate(log_config_kwargs)


def _build_trainer_loss(cfg: AppConfig) -> DefaultLossConfig | SFTLossConfig:
    loss_cfg = getattr(cfg.rl.prime_rl, "trainer_loss", None)
    if loss_cfg is None:
        return DefaultLossConfig()
    loss_type = loss_cfg.type.strip().lower()
    if loss_type == "sft":
        return SFTLossConfig()
    if loss_type != "default":
        raise ValueError("rl.prime_rl.trainer_loss.type must be one of: default, sft")

    loss_kwargs = {
        field_name: getattr(loss_cfg, field_name)
        for field_name in DefaultLossConfig.model_fields
        if field_name != "type" and hasattr(loss_cfg, field_name)
    }
    return DefaultLossConfig.model_validate({"type": "default", **loss_kwargs})


def _apply_zero_advantage_filter_config(
    orchestrator: OrchestratorConfig, cfg: AppConfig
) -> None:
    for filter_cfg in getattr(orchestrator, "filters", []):
        if getattr(filter_cfg, "type", None) == "zero_advantage" and hasattr(
            filter_cfg, "enforce"
        ):
            filter_cfg.enforce = bool(cfg.rl.prime_rl.zero_advantage_filter_enforce)


def _build_trainer_model(
    cfg: AppConfig,
    init_adapter_path: Path | None,
    adapter_lora_config: dict[str, Any] | None,
) -> TrainerModelConfig:
    prime_cfg = cfg.rl.prime_rl
    trainer_model_kwargs: dict[str, Any] = {
        "name": prime_cfg.base_model,
        "impl": cast(Literal["hf", "custom", "auto"], prime_cfg.trainer_impl),
        "attn": cast(
            Literal["sdpa", "flash_attention_2", "flash_attention_3", "fa4"],
            prime_cfg.trainer_attn,
        ),
        "ac": ActivationCheckpointConfig(freq=prime_cfg.activation_checkpoint_freq),
    }
    fused_lm_head_field_name = _trainer_fused_lm_head_field_name()
    if fused_lm_head_field_name is not None:
        trainer_model_kwargs[fused_lm_head_field_name] = cast(
            Literal["auto", "disabled"],
            prime_cfg.fused_lm_head_chunk_size,
        )

    trainer_model = TrainerModelConfig.model_validate(trainer_model_kwargs)
    if prime_cfg.use_lora:
        lora_target_modules = (
            adapter_lora_config["target_modules"]
            if adapter_lora_config is not None
            else _normalize_target_modules(list(prime_cfg.lora_target_modules))
        )
        lora_kwargs: dict[str, Any] = {
            "rank": adapter_lora_config["rank"]
            if adapter_lora_config is not None
            else prime_cfg.lora_rank,
            "alpha": adapter_lora_config["alpha"]
            if adapter_lora_config is not None
            else prime_cfg.lora_alpha,
            "target_modules": lora_target_modules,
        }
        if init_adapter_path is not None and _trainer_lora_supports_init_adapter_path():
            lora_kwargs["init_adapter_path"] = init_adapter_path
        trainer_model.lora = LoRAConfig.model_validate(lora_kwargs)
    return trainer_model


def _build_orchestrator_config(cfg: AppConfig, log_level: str, env_id: str) -> OrchestratorConfig:
    prime_cfg = cfg.rl.prime_rl
    buffer_kwargs: dict[str, Any] = {}
    if "env_sampling_strategy" in BufferConfig.model_fields:
        buffer_kwargs["env_sampling_strategy"] = prime_cfg.env_sampling_strategy
    train_envs = [TrainEnvConfig.model_validate(env) for env in build_train_envs(cfg, env_id)]
    orchestrator = OrchestratorConfig(
        train=TrainConfig(
            env=train_envs,
            sampling=TrainSamplingConfig(max_completion_tokens=prime_cfg.max_tokens),
        ),
        optim=OptimizerConfig(lr=prime_cfg.learning_rate),
        batch_size=prime_cfg.batch_size,
        rollouts_per_example=prime_cfg.rollouts_per_example,
        seq_len=prime_cfg.seq_len,
        max_steps=prime_cfg.max_steps,
        max_async_level=prime_cfg.max_async_level,
        max_off_policy_steps=prime_cfg.max_off_policy_steps,
        buffer=BufferConfig.model_validate(buffer_kwargs),
        log=_build_orchestrator_log_config(log_level),
    )
    _apply_zero_advantage_filter_config(orchestrator, cfg)

    if prime_cfg.eval_on_benchmark:
        benchmark_scenarios = load_benchmark_scenarios()
        eval_num_examples = int(prime_cfg.eval_num_examples)
        if eval_num_examples == -1:
            eval_num_examples = len(benchmark_scenarios)
        if eval_num_examples < 1:
            raise ValueError("rl.prime_rl.eval_num_examples must be -1 or >= 1")

        eval_env = EvalEnvConfig.model_validate(
            {
                "id": env_id,
                "args": build_eval_env_args(
                    cfg,
                    benchmark_scenarios=benchmark_scenarios,
                    num_examples=eval_num_examples,
                ),
            }
        )

        orchestrator.eval = EvalConfig(
            interval=prime_cfg.eval_every,
            num_examples=eval_num_examples,
            rollouts_per_example=prime_cfg.eval_rollouts_per_example,
            sampling=EvalSamplingConfig(
                max_completion_tokens=prime_cfg.eval_max_tokens
                if prime_cfg.eval_max_tokens is not None
                else prime_cfg.max_tokens
            ),
            env=[eval_env],
        )

    return orchestrator


def _validate_prime_rl_settings(cfg: AppConfig) -> None:
    prime_cfg = cfg.rl.prime_rl
    init_adapter_path = _resolve_optional_path(getattr(prime_cfg, "init_adapter_path", None))
    if prime_cfg.source.type != "procedural":
        raise ValueError(
            "Prime-RL training requires rl.prime_rl.source.type=procedural"
        )
    if not prime_cfg.source.generators:
        raise ValueError(
            "Procedural RL training requires rl.prime_rl.source.generators"
        )
    if prime_cfg.use_lora and init_adapter_path is None:
        raise ValueError(
            "Prime-RL LoRA training requires rl.prime_rl.init_adapter_path so LoRA "
            "rank, alpha, and target_modules can be derived from the SFT adapter."
        )
    if init_adapter_path and not prime_cfg.use_lora:
        raise ValueError(
            "rl.prime_rl.init_adapter_path requires rl.prime_rl.use_lora=true"
        )
    if init_adapter_path and not init_adapter_path.exists():
        raise ValueError(
            f"rl.prime_rl.init_adapter_path does not exist: {init_adapter_path}"
        )
    _load_adapter_lora_config(init_adapter_path)

    if prime_cfg.eval_on_benchmark and prime_cfg.eval_every <= 0:
        raise ValueError(
            "rl.prime_rl.eval_on_benchmark requires rl.prime_rl.eval_every > 0"
        )
    if prime_cfg.trainer_impl not in {"hf", "custom", "auto"}:
        raise ValueError("rl.prime_rl.trainer_impl must be one of: hf, custom, auto")
    if prime_cfg.trainer_attn not in {
        "sdpa",
        "flash_attention_2",
        "flash_attention_3",
        "fa4",
    }:
        raise ValueError(
            "rl.prime_rl.trainer_attn must be one of: sdpa, flash_attention_2, flash_attention_3, fa4"
        )
    supports_fused_lm_head_chunk_size = _trainer_fused_lm_head_field_name() is not None
    if supports_fused_lm_head_chunk_size and prime_cfg.fused_lm_head_chunk_size not in {
        "auto",
        "disabled",
    }:
        raise ValueError(
            "rl.prime_rl.fused_lm_head_chunk_size must be one of: auto, disabled"
        )
    if prime_cfg.env_sampling_strategy not in {"random", "round_robin"}:
        raise ValueError(
            "rl.prime_rl.env_sampling_strategy must be one of: random, round_robin"
        )
    if (
        prime_cfg.env_sampling_strategy != "random"
        and "env_sampling_strategy" not in BufferConfig.model_fields
    ):
        raise ValueError(
            "The current prime-rl BufferConfig does not expose env_sampling_strategy; "
            "update the submodule or local checkout before relying on round_robin sampling."
        )
    inference_tool_call_parser = _resolve_optional_string(prime_cfg.inference_tool_call_parser)
    if inference_tool_call_parser is not None:
        raise ValueError(
            "rl.prime_rl.inference_tool_call_parser must be null so Prime-RL "
            "receives raw completions for reward parsing"
        )
    if prime_cfg.inference_enable_auto_tool_choice:
        raise ValueError(
            "rl.prime_rl.inference_enable_auto_tool_choice must be false so "
            "Prime-RL receives raw completions for reward parsing"
        )
    if (
        _resolve_optional_string(getattr(prime_cfg, "inference_reasoning_parser", None)) is not None
        and not _inference_supports_reasoning_parser()
    ):
        raise ValueError(
            "The current prime-rl InferenceModelConfig does not expose reasoning_parser."
        )
    if not prime_cfg.inference_gpu_ids:
        raise ValueError("rl.prime_rl.inference_gpu_ids must not be empty")
    if len(set(prime_cfg.inference_gpu_ids)) != len(prime_cfg.inference_gpu_ids):
        raise ValueError("rl.prime_rl.inference_gpu_ids must be unique")
    if min(prime_cfg.inference_gpu_ids) < 0:
        raise ValueError("rl.prime_rl.inference_gpu_ids must be >= 0")
    if not prime_cfg.trainer_gpu_ids:
        raise ValueError("rl.prime_rl.trainer_gpu_ids must not be empty")
    if len(set(prime_cfg.trainer_gpu_ids)) != len(prime_cfg.trainer_gpu_ids):
        raise ValueError("rl.prime_rl.trainer_gpu_ids must be unique")
    if min(prime_cfg.trainer_gpu_ids) < 0:
        raise ValueError("rl.prime_rl.trainer_gpu_ids must be >= 0")
    if set(prime_cfg.inference_gpu_ids) & set(prime_cfg.trainer_gpu_ids):
        raise ValueError(
            "rl.prime_rl.inference_gpu_ids and rl.prime_rl.trainer_gpu_ids must be disjoint"
        )
    if prime_cfg.ckpt_resume_step is not None and "${now:" in str(prime_cfg.output_dir):
        raise ValueError(
            "Checkpoint resume requires a stable rl.prime_rl.output_dir (no ${now:...} placeholders)."
        )
    if prime_cfg.ckpt_resume_step is not None and prime_cfg.clean:
        raise ValueError("Cannot resume from checkpoint with rl.prime_rl.clean=true")


def _resolve_output_dir(raw_output_dir: str, model_name: str) -> Path:
    resolved = raw_output_dir
    if "${now:" in resolved:
        date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        resolved = f"outputs/prime_rl/{model_dir_name(model_name)}-{date_and_time}"
    return Path(to_absolute_path(resolved))


async def _run_docker_cleanup(ssh_cfg: SSHConfig) -> None:
    endpoints = resolve_ssh_endpoints(ssh_cfg)
    cleanup_pool = HostSSHConnectionPool()
    await asyncio.gather(
        *(
            run_periodic_cleanup(
                endpoint.host,
                endpoint.port,
                ssh_cfg.user,
                ssh_cfg.key_path,
                cleanup_pool,
            )
            for endpoint in endpoints
        )
    )


def _docker_cleanup_main(ssh_cfg: SSHConfig) -> None:
    asyncio.run(_run_docker_cleanup(ssh_cfg))


def _configured_gpu_id_order(cfg: AppConfig) -> list[int]:
    prime_cfg = cfg.rl.prime_rl
    return list(prime_cfg.inference_gpu_ids) + list(prime_cfg.trainer_gpu_ids)


def _resolve_cuda_visible_device_order(cfg: AppConfig) -> list[str]:
    configured_gpu_ids = _configured_gpu_id_order(cfg)
    current = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not current:
        return [str(gpu_id) for gpu_id in configured_gpu_ids]

    visible_devices = [
        device.strip() for device in current.split(",") if device.strip()
    ]
    if not visible_devices:
        return [str(gpu_id) for gpu_id in configured_gpu_ids]

    max_gpu_id = max(configured_gpu_ids, default=-1)
    if max_gpu_id >= len(visible_devices):
        raise ValueError(
            "Configured GPU IDs exceed the currently visible CUDA devices: "
            f"requested max index {max_gpu_id}, visible devices={visible_devices}"
        )
    return [visible_devices[gpu_id] for gpu_id in configured_gpu_ids]


def build_prime_rl_config(cfg: AppConfig):
    """Build prime_rl.rl.RLConfig from Hydra config."""
    prime_cfg = cfg.rl.prime_rl
    _validate_prime_rl_settings(cfg)

    # Validate the prompt template early (fail-fast on missing templates).
    render_system_prompt(cfg)

    output_dir = _resolve_output_dir(prime_cfg.output_dir, prime_cfg.base_model)
    init_adapter_path = _resolve_optional_path(getattr(prime_cfg, "init_adapter_path", None))
    adapter_lora_config = _load_adapter_lora_config(init_adapter_path)

    log_level = _normalize_log_level(str(cfg.log_level))
    inference_reasoning_parser = _resolve_optional_string(getattr(prime_cfg, "inference_reasoning_parser", None))

    model_slug = model_dir_name(prime_cfg.base_model)
    default_wandb_name = (
        f"prime-rl-procedural-{model_slug}-{len(prime_cfg.source.generators)}g"
    )
    wandb_name = prime_cfg.wandb_run_name or default_wandb_name

    ckpt_cfg = None
    if (
        prime_cfg.ckpt_interval is not None
        or prime_cfg.ckpt_resume_step is not None
        or prime_cfg.ckpt_keep_last is not None
        or prime_cfg.ckpt_keep_interval is not None
        or prime_cfg.ckpt_save_adapter_separately
    ):
        ckpt_cfg = SharedCheckpointConfig(
            interval=prime_cfg.ckpt_interval,
            resume_step=prime_cfg.ckpt_resume_step,
            keep_last=prime_cfg.ckpt_keep_last,
            keep_interval=prime_cfg.ckpt_keep_interval,
        )

    env_id = "src.vf_envs.privesc"
    orchestrator = _build_orchestrator_config(cfg, log_level, env_id)
    trainer_model = _build_trainer_model(cfg, init_adapter_path, adapter_lora_config)

    trainer_cfg = TrainerConfig(
        model=trainer_model,
        tokenizer=TokenizerConfig(name=prime_cfg.base_model),
        loss=_build_trainer_loss(cfg),
        optim=AdamWConfig(lr=prime_cfg.learning_rate),
        log=TrainerLogConfig(level=log_level, vf_level=log_level),
    )
    if ckpt_cfg is not None:
        trainer_cfg.ckpt = TrainerCheckpointConfig(
            interval=prime_cfg.ckpt_interval,
            resume_step=prime_cfg.ckpt_resume_step,
            keep_last=prime_cfg.ckpt_keep_last,
            keep_interval=prime_cfg.ckpt_keep_interval,
            weights=WeightCheckpointConfig(
                save_adapter_separately=bool(prime_cfg.ckpt_save_adapter_separately)
            ),
        )
    inference_model_kwargs: dict[str, Any] = {
        "max_model_len": prime_cfg.inference_max_model_len or prime_cfg.seq_len,
        "enforce_eager": bool(prime_cfg.inference_enforce_eager),
        "tool_call_parser": None,
    }
    if inference_reasoning_parser is not None and _inference_supports_reasoning_parser():
        inference_model_kwargs["reasoning_parser"] = inference_reasoning_parser
    inference = InferenceConfig(
        model=InferenceModelConfig(
            **inference_model_kwargs,
        ),
        gpu_memory_utilization=float(prime_cfg.inference_gpu_memory_utilization),
    )

    config = RLConfig(
        trainer=trainer_cfg,
        orchestrator=orchestrator,
        inference=inference,
        output_dir=output_dir,
        ckpt=ckpt_cfg,
        model=SharedModelConfig(name=prime_cfg.base_model),
        deployment=SingleNodeDeploymentConfig(
            gpus_per_node=len(_configured_gpu_id_order(cfg)),
            num_infer_gpus=len(prime_cfg.inference_gpu_ids),
            num_train_gpus=len(prime_cfg.trainer_gpu_ids),
        ),
        seq_len=prime_cfg.seq_len,
        max_steps=prime_cfg.max_steps,
        max_async_level=prime_cfg.max_async_level,
        wandb=SharedWandbConfig(project=prime_cfg.wandb_project, name=wandb_name),
        clean_output_dir=prime_cfg.clean,
    )
    if config.inference is not None and len(prime_cfg.inference_gpu_ids) > 1:
        config.inference.data_parallel_size_local = config.inference.parallel.dp

    if prime_cfg.dump_subconfigs_only:
        config.dry_run = True

    return config


def run(cfg: AppConfig) -> None:
    """Run Prime-RL training from a resolved application config."""
    project_root = str(Path(__file__).resolve().parents[3])
    existing = os.environ.get("PYTHONPATH")
    if not existing:
        os.environ["PYTHONPATH"] = project_root
    elif project_root not in existing.split(":"):
        os.environ["PYTHONPATH"] = f"{project_root}:{existing}"

    prime_cfg = cfg.rl.prime_rl
    configured_gpu_ids = _configured_gpu_id_order(cfg)
    visible_device_order = _resolve_cuda_visible_device_order(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_device_order)
    log.info("Prime-RL RL training")
    log.info("Model: %s", prime_cfg.base_model)
    log.info("Output dir: %s", prime_cfg.output_dir)
    log.info("Procedural generators: %d", len(prime_cfg.source.generators))
    log.info("Configured GPU IDs: %s", configured_gpu_ids)
    log.info("CUDA_VISIBLE_DEVICES: %s", visible_device_order)
    config = build_prime_rl_config(cfg)

    if prime_cfg.scenario_backend.strip().lower() == "remote_ssh":
        cleanup_thread = threading.Thread(
            target=_docker_cleanup_main,
            args=(cfg.ssh,),
            daemon=True,
            name="docker-cleanup",
        )
        cleanup_thread.start()
    else:
        log.info(
            "Skipping Docker cleanup thread for scenario backend: %s",
            prime_cfg.scenario_backend,
        )

    run_prime_rl(config)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: AppConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    register_with_hydra()
    main()
