# ACSAC Artifact Evaluation

Follow the two workflows below to verify the released paper evidence on CPU
and independently repeat the main local-model result.

The live reproduction evaluates all twelve scenarios and three models with
three repetitions each (108 runs). It preserves the checkpoints, evaluation
settings, and root-proof success criteria while reducing repetitions from the
paper's ten per scenario. The archived workflow checks the full released results.

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

Released revisions used by the scripts:

- [evaluation evidence](https://huggingface.co/datasets/sailab-vienna/privesc-llm-evals/tree/5dba67dfd4c16da78a720c16f01e93820cb3471c)
- [SFT and RL adapters](https://huggingface.co/sailab-vienna/privesc-llm-4b/tree/e7a10aadbcc0d0cd721533480d4fd47ee083dc63)
- [dataset and split audit](https://huggingface.co/datasets/sailab-vienna/privesc-llm-data/tree/39a9d2ff37222184fcaf2a6d9a124906e8bb49ad)

The complete evaluated package will be deposited in a permanent public archive
by the final artifact deadline.

## 1. Verify archived evidence

This CPU-only path runs without Docker and requires Linux x86-64, Python 3.12,
uv, 2 CPU cores, 2 GB RAM, approximately 4 GB free disk, and network access for
initial downloads. Our rehearsal took 77 seconds, including installation and
downloads into an empty local cache.

From a fresh clone, run:

```bash
bash scripts/paper/install_analysis.sh
.venv/bin/python scripts/paper/verify_artifact.py
```

Public artifacts download without authentication or paid API access.

The verifier recomputes results from pinned evidence and checks released
summaries, hashes, costs, and the split/leakage audit.

A successful run ends with
`PASS: archived evidence matches all checked paper and rebuttal claims`.

## 2. Independently repeat the local-model result

Requirements: Linux x86-64 with glibc 2.34 or newer, an NVIDIA GPU with compute
capability 8.0 or newer, NVIDIA driver 575.51.03 or newer, Bash, Git, uv, a
Docker CLI, jq, and curl. The workflow was verified on an RTX 4090 with 24 GB
VRAM. We recommend 8 CPU cores, 32 GB RAM, and 80 GB free disk for a single-host
run. Network access is needed for initial downloads and image builds.

Run the live workflow on an isolated test host under your control. Scenario 06
requires privileged Docker-in-Docker. Keep TCP port 5000 available on the Docker
host for the automatically managed image cache.

By default, vLLM, agents, and rootful Docker run on one host:

```bash
bash scripts/paper/run_artifact_eval.sh
```

The script installs dependencies, prepares and tests the benchmark containers,
downloads the pinned models, and evaluates them with eight workers. Results and
logs are written under `outputs/ae/live/`.

<details>
<summary>Optional: use a separate Docker host</summary>

The GPU host needs an OpenSSH client and 30 GB free disk. The separate Docker
host needs rootful Docker, an OpenSSH server, 50 GB free disk, and outbound
network access for image builds. Authorize your SSH key for a user who can run
`docker info` without `sudo`, and allow SSH TCP forwarding.

Docker uses the system SSH client, so `ssh-agent` or an `IdentityFile` in
`~/.ssh/config` works for that connection. The `remote_ssh` backend makes a
separate connection and does not read SSH config; use one directly reachable
`hostname:port` or `IPv4:port` endpoint and a readable, unencrypted
`PRIVESC_KEY`. Trust the host key and verify access:

```bash
eval "$(ssh-agent -s)"
ssh-add "$HOME/.ssh/id_ed25519"
ssh -p 22 root@docker-host.example docker info
```

Run the two-host workflow with:

```bash
PRIVESC_SCENARIO_BACKEND=remote_ssh \
PRIVESC_SSH_SERVERS='docker-host.example:22' \
PRIVESC_USER=root \
PRIVESC_KEY="$HOME/.ssh/id_ed25519" \
bash scripts/paper/run_artifact_eval.sh
```

In two-host mode, the script derives `DOCKER_HOST`; image builds and scenario
execution use that daemon while vLLM and agents stay on the GPU host.

</details>

### Expected result

Fewer repetitions produce more variable success-rate estimates; the stated
gates define a successful reproduction.

The script requires 36 eligible runs per model. It passes when success within
20 agent rounds (r=20) meets these criteria:

- SFT reaches at least 60% at r=20 and at least 1.5 times Base;
- PrivEsc-LLM reaches at least 80% at r=20 and exceeds SFT by at least 5
  percentage points.

A successful run ends with:

```text
PASS: 108-run live evaluation met the Base-to-SFT-to-PrivEsc r20 gates
```

Our three-run rehearsals took **2h 43m 02s on an RTX 4090** and
**2h 20m 04s on an RTX 5090**, with these observed r=20 results:

| Model | RTX 4090 | RTX 5090 |
|---|---:|---:|
| Base | 16/36 (44.4%) | 13/36 (36.1%) |
| SFT | 29/36 (80.6%) | 27/36 (75.0%) |
| PrivEsc-LLM | 33/36 (91.7%) | 33/36 (91.7%) |

Both rehearsals completed all 108 runs and passed the reproduction gates,
with success flags matching root-proof outcomes.

Allow up to one working day, including downloads and image builds.

## Evidence scope

Headline and rebuttal results use ten runs per scenario and a 60-round horizon,
with r=20 as the primary endpoint. Prompt sensitivity uses twenty runs per
scenario. Counts are rollouts, not distinct generated scenarios. Full traces
are supplied for headline and rebuttal static evaluations; the other experiment
families use compact outcome records with source-file fingerprints.

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
