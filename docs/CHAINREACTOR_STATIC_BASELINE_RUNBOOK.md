# ChainReactor Static Baseline Runbook

This runbook reproduces the ChainReactor plan-finding baseline on this repo's static benchmark with the `local_docker` backend.

Eval protocol (sample size, splits, CI reporting) follows `EVAL_PROTOCOL.md`. ChainReactor-specific scope:

- benchmark split: the 12 static scenarios from `load_benchmark_scenarios()`
- backend: `local_docker`
- planner: locally built PowerLifted
- measured outcome: `plan_found` across the 12 static scenarios; execution success is not measured

## Reproducible inputs

- ChainReactor submodule: `external/chainreactor`
- Upstream ChainReactor base commit: `7b1dba8c492bcd3a892404eb76e9696c2becd324`
- Local ChainReactor patch file: `patches/chainreactor-local-docker-ssh-port.patch`
- Local ChainReactor patch helper: `scripts/apply_chainreactor_local_docker_patch.sh`
- PowerLifted source revision: `2bd3bc6ca09d2b239e7699e8bb164703b6153b7d`
- PowerLifted build helper: `scripts/build_powerlifted.sh`
- Harness entrypoint: `src/evaluation/chainreactor_baseline.py`

## Why the full run uses 1800 seconds

The 30-minute budget matches the upstream defaults more closely than the shorter smoke budgets:

- `external/chainreactor/solve_problem.py` hardcodes `--time-limit 1800`
- `external/.powerlifted-src/Apptainer.maidu_sat` uses `--overall-time-limit 30m`
- `external/.powerlifted-src/Apptainer.levitron_sat` uses `--overall-time-limit 30m`
- `external/.powerlifted-src/src/preprocess_h2/planner.cc` uses a 300-second default for h2 mutex preprocessing

The local wrapper keeps a shorter default for smoke runs, so pass `--planner-time-limit 1800` explicitly for the paper-style full sweep.

## One-time setup

```bash
git submodule update --init --recursive
bash scripts/apply_chainreactor_local_docker_patch.sh
uv sync --group chainreactor
```

Build the benchmark images if they are not already present:

```bash
bash external/benchmark-privesc-linux/docker/build.sh
```

Build PowerLifted locally:

```bash
bash -n scripts/build_powerlifted.sh
bash scripts/build_powerlifted.sh
```

`scripts/build_powerlifted.sh` expects:
- `git`
- `cmake`
- `make`
- `python3`
- `gcc` and `g++`
- a usable Boost install

On macOS, the script auto-detects Homebrew Boost. Otherwise set `BOOST_PREFIX` before running it.

## Environment

```bash
source .env
```

The wrapper automatically prefers:
1. `$CHAINREACTOR_PLANNER`
2. `external/powerlifted/bin/powerlifted`
3. `powerlifted` on `PATH`

## Smoke test

```bash
uv run --group chainreactor python -m src.evaluation.chainreactor_baseline \
  --benchmark 01_vuln_suid_gtfo \
  --output-root /tmp/privesc-llm-chainreactor/static_smoke_t1800 \
  --planner-time-limit 1800
```

## Full 30-minute static sweep

```bash
OUTPUT_ROOT="/tmp/privesc-llm-chainreactor/static_full_$(date +%Y%m%d)_t1800"

uv run --group chainreactor python -m src.evaluation.chainreactor_baseline \
  --output-root "$OUTPUT_ROOT" \
  --planner-time-limit 1800
```

The wrapper skips scenarios that already have `run_1.json` under the chosen output root, so rerunning the same command resumes incomplete sweeps.

## Outputs

For each scenario the run writes:
- `run_1.json`
- `run_1_extract.log`
- `run_1_solve.log`
- generated PDDL under `generated_problems_<scenario>_run_1/`
- `plan.1` when a plan is found

Aggregate output:
- `runs.csv`

The CSV and JSON include:
- `status`
- `extract_succeeded`
- `root_problem_exists`
- `plan_found`
- `plan_length`
- `goal_found_time_sec`
- `planner_total_time_sec`
- `planner_time_limit_sec`
- `planner_timed_out`

## Static Scenario Set

The full sweep covers:
- `01_vuln_suid_gtfo`
- `02_vuln_password_in_shell_history`
- `03_vuln_sudo_no_password`
- `05_vuln_sudo_gtfo`
- `06_vuln_docker`
- `07_root_password_reuse_mysql`
- `08_root_password_reuse`
- `09_root_password_root`
- `10_root_allows_lowpriv_to_ssh`
- `11_cron_calling_user_wildcard`
- `12_cron_calling_user_file`
- `13_file_with_root_password`
