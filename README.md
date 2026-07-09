# PrivEsc-LLM

[![arXiv](https://img.shields.io/badge/arXiv-2603.17673-b31b1b.svg)](https://arxiv.org/abs/2603.17673)

> Post-training local LLM agents for Linux privilege escalation using SFT and RL with verifiable rewards.

## Public Release

This repository contains the public source code for PrivEsc-LLM and tracks the artifacts for the current preprint, [arXiv:2603.17673](https://arxiv.org/abs/2603.17673).
Runtime outputs are released separately as organized HuggingFace artifacts:

- [`sailab-vienna/privesc-llm-data`](https://huggingface.co/datasets/sailab-vienna/privesc-llm-data): paper SFT dataset, leakage audit, and examples
- [`sailab-vienna/privesc-llm-4b`](https://huggingface.co/sailab-vienna/privesc-llm-4b): paper SFT and final RL LoRA adapters
- [`sailab-vienna/privesc-llm-evals`](https://huggingface.co/datasets/sailab-vienna/privesc-llm-evals): compact paper evaluation summaries, manifests, and complete headline static benchmark traces

Start with `docs/PAPER_REPRODUCTION_RUNBOOK.md` for the experiment workflow.

## Motivation

Vulnerability assessments involve highly sensitive data: system configurations, credentials, internal network details. Organizations often cannot send this data to external cloud APIs. This repository studies post-training compact local models for verifiable Linux privilege-escalation tasks while keeping deployment-time inference local.

## Paper Pipeline

The paper-facing pipeline trains only on procedural scenarios, reserves the static benchmark for final evaluation, and compares a Qwen3-4B local model ladder against reported API, larger-local, and security-specific baselines under a fixed budgeted protocol. Reproduction details live in `docs/PAPER_REPRODUCTION_RUNBOOK.md`.

## Overview

- **Benchmark**: [Linux PrivEsc benchmark](https://github.com/ipa-lab/benchmark-privesc-linux) scenarios in isolated containers
- **Procedural Training**: 10 generators producing unlimited training scenarios with holdout design
- **Agent**: ReAct loop with two tools (`exec_command` and `test_credentials`)
- **Training**: SFT on procedural traces, then RL with Prime-RL

## Repository Layout

```
src/
├── gym/           # Agent loop, scenario runner, tools
├── generators/    # Procedural scenario generators
├── scenarios/     # Static/procedural scenario sources
├── rl/            # RL training and reward code
└── sft/           # SFT training pipelines

conf/
├── scenarios/     # Static benchmark scenario configs
├── generators/    # Procedural generator configs
├── experiment/    # Experiment presets
└── runner/        # Runner configurations

docs/              # Project documentation
```

## Quickstart

### Setup

General repo setup (works on macOS and Linux for docs, configs, and non-training development):

```bash
git submodule update --init --recursive
uv sync --frozen --group dev
source .venv/bin/activate
```

Training and local-model inference dependencies are Linux-oriented. Install them only on a supported Linux machine or cluster node:

```bash
uv sync --frozen --group sft --group dev
```

### Environment

```bash
cp .env.example .env
# Edit with your API keys and SSH settings
source .env
```

Required variables:
- `OPENAI_API_KEY`, `OPENAI_API_BASE` for API model evaluations
- `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` for experiment tracking
- `PRIVESC_SSH_SERVERS`, `PRIVESC_USER`, `PRIVESC_KEY` for SSH connection to remote Docker hosts

For fully local Docker runs, prefer `scenario.backend=local_docker` and `rl.prime_rl.scenario_backend=local_docker`; in that mode the `PRIVESC_*` variables are optional.

### Full Paper Pipeline

See `docs/PAPER_REPRODUCTION_RUNBOOK.md` for the end-to-end reproducible pipeline, including:

- submodule/bootstrap steps
- benchmark Docker image build
- procedural trace collection
- SFT dataset assembly
- SFT training
- Prime-RL warm start from the produced SFT adapter
- static benchmark evaluation for base, SFT, and RL models

The full paper pipeline requires Linux for the local-model stages, and in practice CUDA-capable Linux for SFT, Prime-RL, and vLLM-backed local evaluation.

### Evaluate on Static Benchmark

```bash
# Single scenario (1 run)
source .env && uv run python -m src.runner \
  +experiment=eval/benchmark \
  scenario.backend=local_docker \
  agent.model=openai/gpt-5.2 \
  runner.source.scenarios='[01_vuln_suid_gtfo]' \
  runner.runs_per_item=1

# Configured static benchmark
source .env && uv run python -m src.runner \
  +experiment=eval/benchmark \
  scenario.backend=local_docker \
  agent.model=openai/gpt-5.2
```

### Evaluate on Procedural Scenarios

```bash
source .env && uv run python -m src.runner \
  +experiment=eval/paper_procedural \
  scenario.backend=local_docker \
  agent.model=openai/gpt-5.2 \
  runner.max_runs=10
```

### RL Training

```bash
# Full reproducible paper pipeline is documented in docs/PAPER_REPRODUCTION_RUNBOOK.md

# Minimal local SFT
source .env && uv run --group sft python -m src.sft.unsloth.train +experiment=train/paper_qwen3_4b_sft

# Minimal local Prime-RL warm start from an SFT adapter
source .env && uv run --group rl python -m src.rl.prime_rl.train \
  +experiment=train/paper_prime_rl_reward_outcome_cost \
  rl.prime_rl.base_model=Qwen/Qwen3-4B-Instruct-2507 \
  rl.prime_rl.init_adapter_path=/absolute/path/to/sft_run/checkpoints/final \
  rl.prime_rl.scenario_backend=local_docker
```

Reward ablation presets reported in the paper are `train/paper_prime_rl_reward_outcome`, `train/paper_prime_rl_reward_outcome_round`, `train/paper_prime_rl_reward_outcome_cost`, and `train/paper_prime_rl_reward_outcome_round_cost`. See [docs/PAPER_REPRODUCTION_RUNBOOK.md](docs/PAPER_REPRODUCTION_RUNBOOK.md) for the paper-facing config map.

## Procedural Generators

Procedural generators provide the training and validation environments while
the static benchmark remains reserved for evaluation. See
[docs/PROCEDURAL_SCENARIOS.md](docs/PROCEDURAL_SCENARIOS.md) for
the generator design and holdout policy.


## Related Work

- **Linux PrivEsc Benchmark** (Happe et al.): [arXiv:2310.11409](https://arxiv.org/abs/2310.11409)
- **InterCode-CTF, CyBench, AutoPenBench**: Broader pentest benchmarks

## Documentation

- [Procedural Scenarios](docs/PROCEDURAL_SCENARIOS.md): generator design and holdout strategy
- [Paper Reproduction Runbook](docs/PAPER_REPRODUCTION_RUNBOOK.md): end-to-end reproducible local-model pipeline

## License

Research code, see LICENSE file.
