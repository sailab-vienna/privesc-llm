# Evaluation Protocol (Paper)

This document records the preprint evaluation protocol and points to the
configs that implement it.

## Static Benchmark

- Split: the 12 static Linux PrivEsc benchmark scenarios from
  `conf/experiment/eval/benchmark.yaml` and `conf/scenarios/`.
- Runs: 10 valid runs per scenario, 120 runs per model.
- Horizon: 60 rounds.
- Primary endpoint: success within 20 rounds, `P(H_root <= 20)`.
- Secondary endpoints: success within smaller/larger round budgets, especially
  `1, 3, 5, 10, 20, 50, 60`.
- Containers: fresh container per run.
- Selection rule: no static benchmark result is used to select trace design,
  prompt, SFT hyperparameters, reward variant, or RL checkpoint.

For exact paper-sized reruns, pass `runner.runs_per_item=10` to `src.runner`
or set `EVAL_RUNNER_RUNS_PER_ITEM=10` for `scripts/run_model_eval_vllm.sh`.

## Procedural Holdout

The procedural holdout is the internal validation oracle used before static
evaluation.

- Training profile: `conf/generators/training.yaml`
- Holdout profile: `conf/generators/holdout.yaml`
- Benchmark exclusion manifest: `src/generators/holdout_manifest.py`
- Split/leakage audit preset: `conf/experiment/audit/split_leakage.yaml`

The trace-design ablation reported in the paper evaluates 500 procedural
holdout runs per condition, balanced over the 10 generator families. Reward
selection uses procedural holdout success at `r <= 20`, with `r <= 60` recovery
and final-checkpoint stability used only as tie-breakers.

## Systems

Paper-facing systems:

- Qwen3-4B base: `Qwen/Qwen3-4B-Instruct-2507`
- Qwen3-4B SFT: base plus the paper SFT LoRA adapter
- PrivEsc-LLM: base plus the final Prime-RL LoRA adapter
- Gemma 4 31B IT local baseline
- DeepSeek V3.2 API baseline
- Claude Opus 4.7 API baseline
- ChainReactor plan-finding baseline

Traditional-tool and human rows are cited from prior work unless a separate
output artifact records fresh runs.

## Reporting

Report success as per-run probability under a fixed round budget, not best-of-k
retry success. Static-benchmark tables and figures use Wilson 95% confidence
intervals where intervals are shown. Cost comparisons use expected cost per
successful root at `r <= 20`.
