import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import torch
from datasets import Dataset, load_dataset
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.config import AppConfig, register_with_hydra
from src.gym.tools import get_tools_for_chat_template
from src.sft.chat_template import apply_chat_templates
from src.sft.traces import scenario_trace_paths


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_traces(dataset_root: Path, leave_out_scenario: str | None) -> Dataset | None:
    trace_paths = scenario_trace_paths(dataset_root, leave_out_scenario)
    if not trace_paths:
        return None
    ds = load_dataset("json", data_files=[str(p) for p in trace_paths], split="train")
    return cast(Dataset, ds)  # split="train" returns a Dataset


class PeftSaveSFTTrainer(SFTTrainer):
    """SFTTrainer with explicit PEFT adapter saving behavior.

    When LoRA targets the output embedding layer (`lm_head`), PEFT may auto-enable
    saving embedding layers, which bloats checkpoints and can cause tied-weight
    serialization issues. Force `save_embedding_layers=False` so checkpoints
    stay adapter-only.
    """

    def save_model(
        self, output_dir: str | None = None, _internal_call: bool = False
    ) -> None:  # noqa: ARG002
        out = str(output_dir or self.args.output_dir)
        Path(out).mkdir(parents=True, exist_ok=True)

        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(out, save_embedding_layers=False)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(out)
            return

        super().save_model(output_dir=out, _internal_call=_internal_call)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: AppConfig) -> None:
    print(OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=True))

    sft_cfg = cfg.sft
    trl_cfg = cfg.sft.trl
    set_all_seeds(int(sft_cfg.seed))

    expected_batch_size = int(trl_cfg.per_device_train_batch_size) * int(
        trl_cfg.gradient_accumulation_steps
    )
    if int(sft_cfg.batch_size) != expected_batch_size:
        raise ValueError(
            "Effective batch size mismatch: "
            f"sft.batch_size={int(sft_cfg.batch_size)} must equal "
            "sft.trl.per_device_train_batch_size * "
            "sft.trl.gradient_accumulation_steps "
            f"({expected_batch_size})."
        )

    dataset_cfg = cfg.datasets.sft
    dataset_root = Path(to_absolute_path(dataset_cfg.output_dir))

    if sft_cfg.leave_out_scenario:
        print(f"Leaving out scenario: {sft_cfg.leave_out_scenario}")

    tools = get_tools_for_chat_template()

    # Load and prepare training data
    reasoning_dataset = load_traces(dataset_root, sft_cfg.leave_out_scenario)
    if reasoning_dataset is None:
        raise ValueError("No training data found!")

    # Load the model in bf16 to match the local training stack.
    from_pretrained_kwargs: dict[str, object] = {"dtype": torch.bfloat16}
    if os.environ.get("SFT_ATTN_IMPL"):
        from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(sft_cfg.model_name)
    chat_template_kwargs = dict(sft_cfg.chat_template_kwargs or {})

    reasoning_dataset = reasoning_dataset.shuffle(seed=sft_cfg.seed)
    if trl_cfg.max_train_examples is not None:
        reasoning_dataset = reasoning_dataset.select(
            range(int(trl_cfg.max_train_examples))
        )
    reasoning_dataset = apply_chat_templates(
        reasoning_dataset,
        tokenizer,
        tools,
        num_proc=int(trl_cfg.dataset_num_proc),
        chat_template_kwargs=chat_template_kwargs,
    )
    print(reasoning_dataset)

    # Build validation dataset
    eval_dataset = None
    if dataset_cfg.validation_output_dir:
        val_root = Path(to_absolute_path(dataset_cfg.validation_output_dir))
        if val_root.exists():
            val_ds = load_traces(val_root, sft_cfg.leave_out_scenario)
            if val_ds is not None:
                eval_dataset = val_ds.shuffle(seed=sft_cfg.seed)
                if trl_cfg.max_eval_examples is not None:
                    eval_dataset = eval_dataset.select(
                        range(int(trl_cfg.max_eval_examples))
                    )
                eval_dataset = apply_chat_templates(
                    eval_dataset,
                    tokenizer,
                    tools,
                    num_proc=int(trl_cfg.dataset_num_proc),
                    chat_template_kwargs=chat_template_kwargs,
                )
                print(f"Validation dataset: {eval_dataset}")
            else:
                print(f"Warning: no validation traces found in {val_root}")
        else:
            print(
                f"Warning: datasets.sft.validation_output_dir {val_root} does not exist, skipping eval"
            )

    model = AutoModelForCausalLM.from_pretrained(
        sft_cfg.model_name,
        **from_pretrained_kwargs,
    )
    if trl_cfg.gradient_checkpointing:
        model.config.use_cache = False

    # PEFT LoRA config. For Qwen3, this explicit list covers attention + MLP
    # projections and includes the output layer (`lm_head`).
    peft_config = LoraConfig(
        r=int(sft_cfg.lora_rank),
        lora_alpha=int(sft_cfg.lora_alpha),
        target_modules=list(trl_cfg.target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Build run name
    model_slug = sft_cfg.model_name.split("/")[-1].replace(":", "-")
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
    models_base = Path(to_absolute_path(sft_cfg.output_dir))
    run_group = os.environ.get("SFT_RUN_GROUP")
    if run_group:
        models_base = models_base / run_group
    run_id = datetime.now(UTC).strftime("%H-%M-%S")
    run_name = f"sft_{model_slug}_{leaveout_slug}_{hp_slug}_{run_id}"
    run_root = models_base / run_name

    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Persist resolved config snapshot
    (run_root / "config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=True), encoding="utf-8"
    )

    if trl_cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = trl_cfg.wandb_project

    eval_kwargs = {}
    if eval_dataset is not None:
        eval_kwargs["eval_strategy"] = trl_cfg.eval_strategy
        if trl_cfg.eval_steps is not None:
            eval_kwargs["eval_steps"] = int(trl_cfg.eval_steps)
        eval_kwargs["per_device_eval_batch_size"] = int(
            trl_cfg.per_device_train_batch_size
        )

    trainer = PeftSaveSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=reasoning_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=int(sft_cfg.max_seq_length),
            run_name=run_name,
            per_device_train_batch_size=int(trl_cfg.per_device_train_batch_size),
            gradient_accumulation_steps=int(trl_cfg.gradient_accumulation_steps),
            warmup_steps=int(trl_cfg.warmup_steps),
            num_train_epochs=float(sft_cfg.num_train_epochs),
            learning_rate=float(sft_cfg.learning_rate),
            adam_beta2=float(trl_cfg.adam_beta2),
            logging_steps=int(trl_cfg.logging_steps),
            optim=trl_cfg.optim,
            weight_decay=float(trl_cfg.weight_decay),
            lr_scheduler_type=trl_cfg.lr_scheduler_type,
            seed=int(sft_cfg.seed),
            report_to=trl_cfg.report_to,
            output_dir=str(ckpt_dir),
            save_strategy=trl_cfg.save_strategy,
            save_steps=int(trl_cfg.save_steps),
            bf16=bool(trl_cfg.bf16),
            gradient_checkpointing=bool(trl_cfg.gradient_checkpointing),
            **eval_kwargs,
        ),
    )

    trainer_stats = trainer.train()
    (run_root / "train_stats.json").write_text(
        json.dumps(trainer_stats.metrics, indent=2), encoding="utf-8"
    )

    # Let TRL/HF handle checkpoint saving; avoid an extra manual save here.


if __name__ == "__main__":
    register_with_hydra()
    main()
