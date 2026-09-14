# ACSAC Artifact Evaluation

This artifact verifies released paper evidence on CPU and independently repeats
the main local-model result in fresh GPU-backed benchmark containers.

## Scope

| Claim | Reviewer evidence |
|---|---|
| RQ1: SFT trace design | Recompute six success curves and intervals from 3,000 compact rollout records. |
| RQ2: reward design | Recompute the four reported reward-condition results and audit all 4,000 compact rollout records. |
| RQ3: static benchmark | Reaggregate all 720 headline traces and independently repeat Base, SFT, and PrivEsc-LLM on the static benchmark. |
| RQ4: cost | Recompute trace costs, RTX 5090 serving-cost calibration, recorded training cost through RL checkpoint 300, and amortization. |
| Prompt sensitivity | Recompute twelve conditions from 2,880 compact rollout records. |
| Rebuttal baselines | Reaggregate all 720 released traces for six model/mode conditions. |
| Split and leakage | Verify disjoint profiles and zero model-visible solution-marker hits across 13,200 examples. |
| ChainReactor | Verify plan finding, extraction status, and timeouts across twelve planner records; execution success is not measured. |

The archived workflow deterministically recomputes results from released
records. The live workflow independently evaluates the released Base, SFT, and
PrivEsc-LLM checkpoints in fresh benchmark containers.

Released revisions used by the scripts:

- [evaluation evidence](https://huggingface.co/datasets/sailab-vienna/privesc-llm-evals/tree/5dba67dfd4c16da78a720c16f01e93820cb3471c)
- [SFT and RL adapters](https://huggingface.co/sailab-vienna/privesc-llm-4b/tree/e7a10aadbcc0d0cd721533480d4fd47ee083dc63)
- [dataset and split audit](https://huggingface.co/datasets/sailab-vienna/privesc-llm-data/tree/39a9d2ff37222184fcaf2a6d9a124906e8bb49ad)

The complete evaluated package will be deposited in a permanent public archive
by the final artifact deadline.

## 1. Verify archived evidence

This CPU-only path runs without Docker and requires Linux x86-64, Python 3.12,
uv, 2 CPU cores, 2 GB RAM, approximately 4 GB free disk, and network access for
initial downloads. Our local rehearsal took 63 seconds after installation and
download.

From a fresh clone, run:

```bash
bash scripts/paper/install_analysis.sh
.venv/bin/python scripts/paper/verify_artifact.py
```

Public artifacts download without authentication. If Hugging Face rate-limits
anonymous downloads, run `.venv/bin/hf auth login` and retry.

The verifier downloads pinned evidence, runs metric, path, and audit tests,
reaggregates 720 headline and 720 rebuttal traces, compares released summaries,
verifies immutable hashes, recomputes results from 10,600 scalar records, checks
cost claims, and verifies the released split/leakage audit.

A successful run ends with
`PASS: archived evidence matches all checked paper and rebuttal claims`. It
reproduces the headline r=20 results of 40.8%, 79.2%, and 93.3% for Base, SFT,
and PrivEsc-LLM, respectively; the rebuttal r=20 values of 30.8%, 20.0%, 51.7%,
27.5%, 9.2%, and 99.2%; the $0.00212905 PrivEsc-LLM inference estimate; the
$0.00468650 DeepSeek V4 Flash estimate and 2.2012 ratio; and the $121.08 recorded
training cost through RL checkpoint 300.

## 2. Independently repeat the local-model result

Requirements: a Linux x86-64 host with an NVIDIA GPU and 32 GB
VRAM, 8 CPU cores, 32 GB RAM, 50 GB free disk, rootful Docker, jq, and curl.
Network access is needed for initial downloads. The workflow was rehearsed on
an RTX 5090 with NVIDIA driver 575.51.03.

Run the live workflow on an isolated test host under your control. Scenario 06
requires privileged Docker-in-Docker. Keep host TCP port 5000 available for the
automatically managed image cache.

Run:

```bash
bash scripts/paper/run_artifact_eval.sh
```

The script initializes the benchmark submodule, applies the released patch,
builds and tests the twelve paper scenarios, downloads the pinned adapters,
serves each model through the existing vLLM launcher, and evaluates five runs
per scenario with eight workers. Serving and sampling settings come from the
script, launcher, and checked-in Hydra config. Outputs are written under
`outputs/ae/live/`.

The script requires 60 eligible runs per model and passes when:

- SFT reaches at least 60% at r=20 and at least 1.5 times Base;
- PrivEsc-LLM reaches at least 80% at r=20 and exceeds SFT by at least 5
  percentage points.

Our RTX 5090 rehearsal completed in 3:15:27 and passed. Allow up to one
working day on a fresh host, including downloads and image builds.

## Evidence scope

Headline and rebuttal results use ten runs per scenario and a 60-round horizon,
with r=20 as the primary endpoint. Prompt sensitivity uses twenty runs per
scenario. Counts are rollouts, not distinct generated scenarios. Full traces
are supplied for headline and rebuttal static evaluations; the other experiment
families use compact outcome records with source-file fingerprints.

RQ1 trace-design evidence evaluates the seed-1337 SFT checkpoint at 366/500
(73.2%). The released SFT adapter and RL initialization use seed 2026, confirmed
on a distinct 300-run cohort at 228/300 (76.0%). The verifier checks both
cohorts.

Cost assumptions and scope are documented in [inference cost](docs/TOKEN_COST.md)
and [training cost](docs/TRAINING_COST.md).

The [split/leakage report](https://huggingface.co/datasets/sailab-vienna/privesc-llm-data/blob/39a9d2ff37222184fcaf2a6d9a124906e8bb49ad/leakage_audit/summary.json)
verifies disjoint procedural profiles and zero model-visible solution-marker
hits across 13,200 examples.

The [trace examples](examples/) include the two examples referenced in the
paper. Credentials, usernames, paths, and keys in datasets and traces are
synthetic benchmark artifacts. All privilege-escalation targets are controlled
benchmark containers; the work uses no personal data or human participants.
The paper's ethics discussion provides the full dual-use analysis.
