# Paper Reproduction Runbook

This runbook maps the paper pipeline to the public source repository and released artifacts. It documents
the configs readers need to reproduce or audit the reported experiments,
without listing older pilots or non-paper variants.

The paper pipeline is:

1. build the static benchmark and procedural Docker images
2. collect and assemble procedural SFT traces
3. train the paper Qwen3-4B SFT warm start
4. train Prime-RL from the SFT adapter with the final reward chosen on procedural holdout
5. evaluate base, SFT, and RL models on the static benchmark
6. inspect the split/leakage audit and paper output bundle

## Scope

- Training data comes only from procedural scenarios.
- The static benchmark is the final held-out evaluation.
- Trace design, SFT hyperparameters, reward variant, prompt, and checkpoint
  choices are selected on procedural holdout only.
- API baseline reruns are supported but are not fully immutable because
  provider-hosted model aliases and routing can drift.

## 1. Bootstrap

Clone with submodules:

```bash
git clone --recursive https://github.com/sailab-vienna/privesc-llm.git
cd privesc-llm
git submodule update --init --recursive
```

Use Linux for training, local-model inference, and Docker-backed reproduction:

```bash
uv sync --frozen --group dev
source .venv/bin/activate
```

Training stacks are split because SFT and RL/vLLM need different dependency
ranges:

```bash
uv sync --frozen --group dev --group sft   # Unsloth SFT
uv sync --frozen --group dev --group rl    # Prime-RL and vLLM eval
```

Paper-facing submodules:

- `external/benchmark-privesc-linux`: static benchmark Docker recipes
- `external/GTFOBins`: source for checked-in GTFOBins generator catalog
- `external/prime-rl`: RL trainer
- `external/chainreactor`: symbolic-planning baseline
- `external/.powerlifted-src`: planner build for ChainReactor

Apply the local patches after initializing submodules:

```bash
bash scripts/apply_benchmark_patch.sh
bash scripts/apply_prime_rl_sft_patch.sh
bash scripts/apply_chainreactor_local_docker_patch.sh
```

## 2. Docker Images

Build the static benchmark images:

```bash
bash external/benchmark-privesc-linux/docker/build.sh
```

Verify those images through the same local Docker backend used by the paper
runner:

```bash
PRIVESC_SCENARIO_BACKEND=local_docker uv run pytest -q test/test_solutions.py
```

Build the procedural training image:

```bash
bash docker/procedural/build.sh
```

## 3. Split And Leakage Audit Pointers

These are the main files reviewers should inspect to assess the clean split:

- `conf/generators/training.yaml` and `conf/generators/training/`: procedural
  training profile
- `conf/generators/holdout.yaml` and `conf/generators/holdout/`: value-disjoint
  procedural validation/evaluation profile
- `src/generators/holdout_manifest.py`: static benchmark cases and
  benchmark-specific exclusion rules
- `test/procedural/test_holdout_leakage.py`: regression tests that keep the
  holdout manifest aligned with the live generator mix and benchmark scenarios
- `conf/experiment/audit/split_leakage.yaml`: paper split/leakage audit preset
- `src/dataset/privesc/audit.py`: audit bundle writer for split diffs,
  benchmark holdouts, trace leakage summaries, and selection provenance
- `conf/datasets/sft/quality/shared.yaml`: SFT quality filters, including
  solution-leakage and benchmark-holdout rejection hooks

Run the paper audit bundle with:

```bash
uv run python -m src.dataset.privesc.audit +experiment=audit/split_leakage
```

The released data artifact includes `leakage_audit/summary.json`,
`leakage_audit/split_holdouts.tsv`, and `leakage_audit/benchmark_holdouts.tsv`.

## 4. Trace Collection And SFT Dataset

The preprint uses DeepSeek V4 Flash as the teacher and evaluates a 2x3 trace
design: guided vs. unguided collection crossed with no, short, and long
reasoning. The paper SFT data is unguided long reasoning.

Paper-facing trace collection configs:

- `conf/experiment/trace/standard_guided_deepseek_training.yaml`
- `conf/experiment/trace/standard_guided_deepseek_validation.yaml`
- `conf/experiment/trace/standard_unguided_deepseek_training.yaml`
- `conf/experiment/trace/standard_unguided_deepseek_validation.yaml`

Paper-facing derivation configs for the no-reasoning and short-reasoning rows:

- `conf/experiment/derive/standard_guided_deepseek_no_reasoning_training.yaml`
- `conf/experiment/derive/standard_guided_deepseek_no_reasoning_validation.yaml`
- `conf/experiment/derive/standard_guided_deepseek_short_reasoning_training.yaml`
- `conf/experiment/derive/standard_guided_deepseek_short_reasoning_validation.yaml`
- `conf/experiment/derive/standard_unguided_deepseek_no_reasoning_training.yaml`
- `conf/experiment/derive/standard_unguided_deepseek_no_reasoning_validation.yaml`
- `conf/experiment/derive/standard_unguided_deepseek_short_reasoning_training.yaml`
- `conf/experiment/derive/standard_unguided_deepseek_short_reasoning_validation.yaml`

Collected traces are assembled with:

- `scripts/collect_and_assemble.sh`
- `src/dataset/privesc/sft.py`
- `conf/datasets/sft/collection/standard/training.yaml`
- `conf/datasets/sft/collection/standard/validation.yaml`
- `conf/datasets/sft/quality/guided.yaml`
- `conf/datasets/sft/quality/unguided.yaml`

Released paper dataset path:

```text
privesc-llm-data/paper_sft_dataset/
  training/     # 2000 traces, 200 per generator
  validation/   # 200 traces, 20 per generator
```

Representative paper-data collection commands:

```bash
source .env
bash scripts/collect_and_assemble.sh unguided deepseek training scenario.backend=local_docker
bash scripts/collect_and_assemble.sh unguided deepseek validation scenario.backend=local_docker
```

## 5. SFT Warm Start

Final paper SFT config:

- `conf/experiment/train/paper_qwen3_4b_sft.yaml`
- trainer: `src/sft/unsloth/train.py`
- base model: `Qwen/Qwen3-4B-Instruct-2507`
- LoRA rank/alpha: `8/32`
- sequence length: `32768`
- epochs: `10`

Run:

```bash
source .env
uv run --group sft python -m src.sft.unsloth.train \
  +experiment=train/paper_qwen3_4b_sft
```

The appendix reports the SFT learning-rate/rank selection protocol. The
paper-facing sweep configs are the Unsloth configs under
`conf/experiment/sweep/unsloth_*.yaml`; the TRL configs are not part of the
preprint pipeline.

Each SFT run writes a resolved config, training stats, final adapter, and
`artifact_manifest.json`. Reviewers should check that the manifest points to
the paper unguided DeepSeek long-reasoning training and validation splits.

## 6. Prime-RL

The final reported model uses the paper SFT adapter and the
`outcome_cost` reward variant. The paper also reports the 2x2 reward ablation
over outcome, round shaping, and cost shaping.

Paper-facing RL configs:

- `conf/experiment/train/paper_prime_rl_reward_outcome.yaml`
- `conf/experiment/train/paper_prime_rl_reward_outcome_round.yaml`
- `conf/experiment/train/paper_prime_rl_reward_outcome_cost.yaml`
- `conf/experiment/train/paper_prime_rl_reward_outcome_round_cost.yaml`

The shared RL base config is `conf/experiment/train/paper_prime_rl.yaml`; the
entry point is `src/rl/prime_rl/train.py`.

Final reward run chosen on procedural holdout:

```bash
SFT_RUN=/absolute/path/to/sft_run

source .env
uv run --group rl python -m src.rl.prime_rl.train \
  +experiment=train/paper_prime_rl_reward_outcome_cost \
  rl.prime_rl.base_model=Qwen/Qwen3-4B-Instruct-2507 \
  rl.prime_rl.init_adapter_path="$SFT_RUN/checkpoints/final" \
  rl.prime_rl.scenario_backend=local_docker
```

Paper RL settings inherited by the reward configs include 1000 training steps,
batch size 80, 8 rollouts per instance, 20 training rounds, round-robin
procedural generator sampling, and no benchmark evaluation during training.

For reward-ablation reproduction, run the four configs listed above from the
same SFT adapter and select checkpoints on procedural holdout only.

## 7. Static Benchmark Evaluation

The paper evaluates 10 runs per static benchmark scenario, 12 scenarios total,
with a 60-round cap. The primary metric is success within 20 rounds.

Paper-facing evaluation configs:

- `conf/experiment/eval/paper_static_qwen3_4b_base.yaml`
- `conf/experiment/eval/paper_static_gemma4_31b_it_thinking.yaml`
- `conf/experiment/eval/paper_static_deepseek32.yaml`
- `conf/experiment/eval/paper_static_opus47.yaml`
- `conf/experiment/eval/paper_static_base.yaml`
- `conf/experiment/eval/benchmark.yaml`

For exact paper-sized reruns, set `EVAL_RUNNER_RUNS_PER_ITEM=10` when using
`scripts/run_model_eval_vllm.sh`, or pass `runner.runs_per_item=10` directly to
`src.runner`.

Base Qwen3-4B:

```bash
source .env
EVAL_RUNNER_RUNS_PER_ITEM=10 \
bash scripts/run_model_eval_vllm.sh \
  Qwen/Qwen3-4B-Instruct-2507 \
  qwen3-4b
```

SFT adapter:

```bash
SFT_RUN=/absolute/path/to/sft_run

source .env
EVAL_RUNNER_RUNS_PER_ITEM=10 \
PAPER_EVAL_ENABLE_LORA=1 \
PAPER_EVAL_LORA_ADAPTER_DIR="$SFT_RUN/checkpoints/final" \
PAPER_EVAL_LORA_MODEL_NAME=sft_qwen3_4b \
bash scripts/run_model_eval_vllm.sh \
  Qwen/Qwen3-4B-Instruct-2507 \
  sft_qwen3_4b
```

RL adapter:

```bash
RL_RUN=/absolute/path/to/rl_run/prime_rl
STEP=300

source .env
EVAL_RUNNER_RUNS_PER_ITEM=10 \
PAPER_EVAL_ENABLE_LORA=1 \
PAPER_EVAL_LORA_ADAPTER_DIR="$RL_RUN/run_default/broadcasts/step_${STEP}" \
PAPER_EVAL_LORA_MODEL_NAME="prime_rl_${STEP}" \
PAPER_EVAL_VLLM_REASONING_PARSER="" \
bash scripts/run_model_eval_vllm.sh \
  Qwen/Qwen3-4B-Instruct-2507 \
  "prime_rl_${STEP}"
```

API baseline examples:

```bash
source .env
uv run python -m src.runner \
  +experiment=eval/paper_static_opus47 \
  scenario.backend=local_docker \
  runner.runs_per_item=10

uv run python -m src.runner \
  +experiment=eval/paper_static_deepseek32 \
  scenario.backend=local_docker \
  runner.runs_per_item=10
```

## 8. ChainReactor Baseline

The preprint reports ChainReactor as a plan-finding baseline on the same
static benchmark. Use:

- `docs/CHAINREACTOR_STATIC_BASELINE_RUNBOOK.md`
- `src/evaluation/chainreactor_baseline.py`
- `patches/chainreactor-local-docker-ssh-port.patch`
- `scripts/build_powerlifted.sh`

## 9. Released Artifact Bundle

The public HuggingFace artifacts are:

```text
sailab-vienna/privesc-llm-data
sailab-vienna/privesc-llm-4b
sailab-vienna/privesc-llm-evals
```

Important files include:

- SFT dataset `stats.json` files for the paper training/validation splits
- SFT and RL LoRA adapters
- static eval trace directories and `evaluation_summary_paper.json`
- split/leakage audit `summary.json`, `split_holdouts.tsv`, and
  `benchmark_holdouts.tsv`

## 10. Quick Checks

Config resolution:

```bash
CONFIG_CHECK_DIR="${TMPDIR:-/tmp}/privesc-llm-artifact-config-check"
export PRIVESC_SSH_SERVERS='localhost:22'
uv run python -m src.config +experiment=train/paper_qwen3_4b_sft hydra.run.dir="$CONFIG_CHECK_DIR" hydra.output_subdir=null
uv run python -m src.config +experiment=train/paper_prime_rl_reward_outcome_cost hydra.run.dir="$CONFIG_CHECK_DIR" hydra.output_subdir=null
uv run python -m src.config +experiment=eval/paper_static_qwen3_4b_base hydra.run.dir="$CONFIG_CHECK_DIR" hydra.output_subdir=null
```

Focused static/procedural checks:

```bash
bash -n scripts/collect_and_assemble.sh scripts/collect_and_assemble_base.sh scripts/run_model_eval_vllm.sh docker/procedural/build.sh
uv run pytest -q test/procedural/test_holdout_leakage.py
PRIVESC_SCENARIO_BACKEND=local_docker uv run pytest -q test/test_solutions.py
```

Full SFT, Prime-RL, and vLLM-backed evaluation require Linux with the
appropriate CUDA stack.
