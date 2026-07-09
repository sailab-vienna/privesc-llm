from unsloth import (
    FastLanguageModel,
)  # Must be imported before torch, transformers, etc.
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
import subprocess

import hydra
import numpy as np
import torch
import torch.distributed as dist
from accelerate import DistributedDataParallelKwargs
from datasets import Dataset, load_dataset
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from peft import PeftModel
from torch.distributed.distributed_c10d import destroy_process_group, is_initialized
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from src.config import AppConfig, register_with_hydra
from src.gym.tools import get_tools_for_chat_template
from src.paths import base_model_slug, path_from_outputs
from src.sft.chat_template import apply_chat_templates
from src.sft.traces import scenario_trace_paths


class StaticGraphDDPSFTTrainer(SFTTrainer):
    def _build_accelerator_args(self, **kwargs) -> dict[str, Any]:
        args = super()._build_accelerator_args(**kwargs)
        if distributed_world_size() <= 1:
            return args

        args["kwargs_handlers"] = [
            DistributedDataParallelKwargs(
                **(handler.to_kwargs() | {"static_graph": True})
            )
            if isinstance(handler, DistributedDataParallelKwargs)
            else handler
            for handler in args.get("kwargs_handlers", [])
        ]
        return args


class AdapterOnlySaveSFTTrainer(StaticGraphDDPSFTTrainer):
    def save_model(
        self, output_dir: str | None = None, _internal_call: bool = False
    ) -> None:  # noqa: ARG002
        out = str(output_dir or self.args.output_dir)

        if isinstance(self.model, PeftModel):
            if not self.is_world_process_zero():
                return

            Path(out).mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(
                out,
                save_embedding_layers=False,
                state_dict=self.model.state_dict(),
            )
            if self.processing_class is not None:
                self.processing_class.save_pretrained(out)
            return

        Path(out).mkdir(parents=True, exist_ok=True)
        super().save_model(output_dir=out, _internal_call=_internal_call)


def apply_lora(model: Any, sft_cfg, unsloth_cfg, adapter_only_lm_head: bool) -> Any:
    target_modules = list(unsloth_cfg.target_modules)
    if adapter_only_lm_head:
        from peft import LoraConfig, get_peft_model

        peft_cfg = LoraConfig(
            r=int(sft_cfg.lora_rank),
            lora_alpha=int(sft_cfg.lora_alpha),
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_cfg.ensure_weight_tying = bool(unsloth_cfg.ensure_weight_tying)

        model = get_peft_model(model, peft_cfg)
        return FastLanguageModel.patch_peft_model(
            model,
            use_gradient_checkpointing=str(
                unsloth_cfg.use_gradient_checkpointing or "unsloth"
            ),
        )

    return FastLanguageModel.get_peft_model(
        model,
        r=int(sft_cfg.lora_rank),
        target_modules=target_modules,
        lora_alpha=int(sft_cfg.lora_alpha),
        use_gradient_checkpointing=(
            str(unsloth_cfg.use_gradient_checkpointing)
            if unsloth_cfg.use_gradient_checkpointing is not None
            else "False"
        ),
        random_state=int(sft_cfg.seed),
    )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def configure_ddp_device() -> None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))


def cleanup_distributed_process_group() -> None:
    if dist.is_available() and is_initialized():
        destroy_process_group()


def training_gradient_accumulation_steps(sft_cfg, unsloth_cfg) -> int:
    batch_size = int(sft_cfg.batch_size)
    per_device_batch_size = int(unsloth_cfg.per_device_train_batch_size)
    world_size = distributed_world_size()

    if world_size == 1:
        gradient_accumulation_steps = int(unsloth_cfg.gradient_accumulation_steps)
        expected_batch_size = per_device_batch_size * gradient_accumulation_steps
        if batch_size != expected_batch_size:
            raise ValueError(
                "Effective batch size mismatch: "
                f"sft.batch_size={batch_size} must equal "
                "sft.unsloth.per_device_train_batch_size * "
                "sft.unsloth.gradient_accumulation_steps "
                f"({expected_batch_size})."
            )
        return gradient_accumulation_steps

    denominator = per_device_batch_size * world_size
    if batch_size % denominator != 0:
        raise ValueError(
            "Effective batch size mismatch: "
            f"sft.batch_size={batch_size} must be divisible by "
            "sft.unsloth.per_device_train_batch_size * world_size "
            f"({denominator})."
        )
    return batch_size // denominator


def git_commit_hash() -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def load_traces(dataset_root: Path, leave_out_scenario: str | None) -> Dataset | None:
    trace_paths = scenario_trace_paths(dataset_root, leave_out_scenario)
    if not trace_paths:
        return None
    return cast(
        Dataset,
        load_dataset("json", data_files=[str(p) for p in trace_paths], split="train"),
    )


def best_eval_entry(log_history: list[dict[str, Any]]) -> dict[str, Any] | None:
    eval_entries = [entry for entry in log_history if "eval_loss" in entry]
    if not eval_entries:
        return None
    return min(eval_entries, key=lambda entry: float(entry["eval_loss"]))


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: AppConfig) -> None:
    if is_main_process():
        print(OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=False))

    sft_cfg = cfg.sft
    unsloth_cfg = cfg.sft.unsloth
    configure_ddp_device()
    set_all_seeds(int(sft_cfg.seed))

    if unsloth_cfg.max_steps is not None and int(unsloth_cfg.max_steps) < 1:
        raise ValueError(
            f"sft.unsloth.max_steps must be >= 1 when set (got {unsloth_cfg.max_steps})."
        )

    gradient_accumulation_steps = training_gradient_accumulation_steps(
        sft_cfg, unsloth_cfg
    )

    dataset_cfg = cfg.datasets.sft
    dataset_root = Path(to_absolute_path(dataset_cfg.output_dir))

    if sft_cfg.leave_out_scenario:
        print(f"Leaving out scenario: {sft_cfg.leave_out_scenario}")

    tools = get_tools_for_chat_template()

    reasoning_dataset = load_traces(dataset_root, sft_cfg.leave_out_scenario)
    if reasoning_dataset is None:
        raise ValueError("No training data found!")

    if unsloth_cfg.full_finetuning:
        if unsloth_cfg.load_in_4bit or unsloth_cfg.load_in_8bit or unsloth_cfg.use_lora:
            raise ValueError(
                "full_finetuning=True requires load_in_4bit=False, load_in_8bit=False, use_lora=False"
            )
    else:
        if unsloth_cfg.load_in_4bit and unsloth_cfg.load_in_8bit:
            raise ValueError("Only one of load_in_4bit or load_in_8bit can be True.")
        if not unsloth_cfg.use_lora and unsloth_cfg.load_in_4bit:
            raise ValueError("QLoRA requires use_lora=True when load_in_4bit=True.")

    from_pretrained_kwargs = {
        "model_name": sft_cfg.model_name,
        "max_seq_length": int(sft_cfg.max_seq_length),
        "load_in_4bit": bool(unsloth_cfg.load_in_4bit),
    }
    if unsloth_cfg.load_in_8bit:
        from_pretrained_kwargs["load_in_8bit"] = True
    if unsloth_cfg.full_finetuning:
        from_pretrained_kwargs["full_finetuning"] = True
    if unsloth_cfg.use_lora and not unsloth_cfg.full_finetuning:
        from_pretrained_kwargs["max_lora_rank"] = int(sft_cfg.lora_rank)

    model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)

    chat_template_kwargs = dict(sft_cfg.chat_template_kwargs or {})
    reasoning_dataset = apply_chat_templates(
        reasoning_dataset,
        tokenizer,
        tools,
        num_proc=int(unsloth_cfg.dataset_num_proc),
        chat_template_kwargs=chat_template_kwargs,
    )
    reasoning_dataset = reasoning_dataset.shuffle(seed=sft_cfg.seed)
    print(reasoning_dataset)

    eval_dataset = None
    val_root: Path | None = None
    if dataset_cfg.validation_output_dir:
        val_root = Path(to_absolute_path(dataset_cfg.validation_output_dir))
        if val_root.exists():
            val_ds = load_traces(val_root, sft_cfg.leave_out_scenario)
            if val_ds is not None:
                eval_dataset = apply_chat_templates(
                    val_ds,
                    tokenizer,
                    tools,
                    num_proc=int(unsloth_cfg.dataset_num_proc),
                    chat_template_kwargs=chat_template_kwargs,
                )
                print(f"Validation dataset: {eval_dataset}")
            else:
                print(f"Warning: no validation traces found in {val_root}")
        else:
            print(
                f"Warning: datasets.sft.validation_output_dir {val_root} does not exist, skipping eval"
            )

    adapter_only_lm_head = False
    if unsloth_cfg.use_lora and not unsloth_cfg.full_finetuning:
        adapter_only_lm_head = bool(
            unsloth_cfg.adapter_only_lm_head
            and "lm_head" in list(unsloth_cfg.target_modules)
        )
        model = apply_lora(model, sft_cfg, unsloth_cfg, adapter_only_lm_head)

    model_slug = base_model_slug(sft_cfg.model_name)
    leaveout_slug = (
        f"leaveout-{sft_cfg.leave_out_scenario}"
        if sft_cfg.leave_out_scenario
        else "all"
    )
    hp_slug = (
        f"lr{sft_cfg.learning_rate:.1e}_"
        f"r{sft_cfg.lora_rank}_"
        f"ep{int(sft_cfg.num_train_epochs)}_"
        f"sd{sft_cfg.seed}"
    )

    run_id = datetime.now(UTC).strftime("%H-%M-%S")
    run_name = f"sft_{model_slug}_{leaveout_slug}_{hp_slug}_{run_id}"
    wandb_run_name = unsloth_cfg.wandb_run_name or run_name
    if sft_cfg.run_dir:
        run_root = Path(to_absolute_path(sft_cfg.run_dir))
    else:
        models_base = Path(to_absolute_path(sft_cfg.output_dir))
        run_root = models_base / run_name

    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        (run_root / "config.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=False), encoding="utf-8"
        )

    if unsloth_cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = unsloth_cfg.wandb_project

    eval_kwargs = {}
    if eval_dataset is not None:
        eval_kwargs["eval_strategy"] = unsloth_cfg.eval_strategy
        if unsloth_cfg.eval_steps is not None:
            eval_kwargs["eval_steps"] = int(unsloth_cfg.eval_steps)
        eval_kwargs["per_device_eval_batch_size"] = int(
            unsloth_cfg.per_device_train_batch_size
        )
        eval_kwargs["eval_on_start"] = bool(unsloth_cfg.eval_on_start)
        eval_kwargs["prediction_loss_only"] = bool(unsloth_cfg.prediction_loss_only)
        eval_kwargs["load_best_model_at_end"] = True
        eval_kwargs["metric_for_best_model"] = "eval_loss"
        eval_kwargs["greater_is_better"] = False

    trainer_cls = AdapterOnlySaveSFTTrainer if adapter_only_lm_head else StaticGraphDDPSFTTrainer

    max_steps_kwargs = {}
    if unsloth_cfg.max_steps is not None:
        max_steps_kwargs["max_steps"] = int(unsloth_cfg.max_steps)

    ddp_kwargs = {}
    if distributed_world_size() > 1:
        ddp_kwargs["ddp_find_unused_parameters"] = True

    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        train_dataset=reasoning_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            run_name=wandb_run_name,
            per_device_train_batch_size=int(unsloth_cfg.per_device_train_batch_size),
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=int(unsloth_cfg.warmup_steps),
            num_train_epochs=float(sft_cfg.num_train_epochs),
            learning_rate=float(sft_cfg.learning_rate),
            adam_beta2=float(unsloth_cfg.adam_beta2),
            logging_steps=int(unsloth_cfg.logging_steps),
            optim=unsloth_cfg.optim,
            weight_decay=float(unsloth_cfg.weight_decay),
            lr_scheduler_type=unsloth_cfg.lr_scheduler_type,
            seed=int(sft_cfg.seed),
            report_to=unsloth_cfg.report_to,
            output_dir=str(ckpt_dir),
            save_strategy=unsloth_cfg.save_strategy,
            save_steps=int(unsloth_cfg.save_steps),
            packing=unsloth_cfg.packing,
            dataset_num_proc=int(unsloth_cfg.dataset_num_proc),
            **max_steps_kwargs,
            **eval_kwargs,
            **ddp_kwargs,
        ),
    )

    trainer_stats = trainer.train()
    if not is_main_process():
        cleanup_distributed_process_group()
        return

    (run_root / "train_stats.json").write_text(
        json.dumps(trainer_stats.metrics, indent=2), encoding="utf-8"
    )

    selection_summary: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "has_eval_dataset": eval_dataset is not None,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
    }
    best_eval = best_eval_entry(list(trainer.state.log_history or []))
    if best_eval is not None:
        selection_summary["best_eval"] = {
            "eval_loss": float(best_eval["eval_loss"]),
            "epoch": best_eval.get("epoch"),
            "step": best_eval.get("step"),
        }
    (run_root / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2), encoding="utf-8"
    )

    trainer.save_model(str(ckpt_dir / "final"))

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit_hash(),
        "base_model": sft_cfg.model_name,
        "run_name": run_name,
        "run_root": str(run_root.resolve()),
        "run_root_path_from_outputs": path_from_outputs(run_root),
        "config_path": str((run_root / "config.yaml").resolve()),
        "config_path_from_outputs": path_from_outputs(run_root / "config.yaml"),
        "checkpoint_dir": str(ckpt_dir.resolve()),
        "checkpoint_dir_path_from_outputs": path_from_outputs(ckpt_dir),
        "final_adapter_dir": str((ckpt_dir / "final").resolve()),
        "final_adapter_dir_path_from_outputs": path_from_outputs(ckpt_dir / "final"),
        "train_stats_path": str((run_root / "train_stats.json").resolve()),
        "train_stats_path_from_outputs": path_from_outputs(
            run_root / "train_stats.json"
        ),
        "selection_summary_path": str((run_root / "selection_summary.json").resolve()),
        "selection_summary_path_from_outputs": path_from_outputs(
            run_root / "selection_summary.json"
        ),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_root_path_from_outputs": path_from_outputs(dataset_root),
        "validation_dataset_root": (
            str(val_root.resolve())
            if val_root is not None and val_root.exists()
            else None
        ),
        "validation_dataset_root_path_from_outputs": (
            path_from_outputs(val_root)
            if val_root is not None and val_root.exists()
            else None
        ),
    }
    (run_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    cleanup_distributed_process_group()


if __name__ == "__main__":
    register_with_hydra()
    main()
