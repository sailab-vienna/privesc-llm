# Training Cost Estimation

The recorded selected SFT runtime and RL training through checkpoint 300 cost
**$121.08** using four H100 GPUs at $2.29 per GPU-hour.

| Stage | Recorded seconds | Estimated cost |
|---|---:|---:|
| Selected SFT, seed 2026 | 6,006.8683 | $15.28 |
| Selected RL, through checkpoint 300 | 41,578.7137 | $105.79 |
| Combined | 47,585.5820 | **$121.08** |

The calculation is `seconds * 4 / 3600 * 2.29`; the combined total is rounded
only after adding both unrounded stage costs.

The [SFT record](https://huggingface.co/datasets/sailab-vienna/privesc-llm-evals/blob/5dba67dfd4c16da78a720c16f01e93820cb3471c/evaluation_artifacts/provenance/cost/training/sft_train_stats.json)
contains the selected run's trainer runtime. The
[RL timing record](https://huggingface.co/datasets/sailab-vienna/privesc-llm-evals/blob/5dba67dfd4c16da78a720c16f01e93820cb3471c/evaluation_artifacts/provenance/cost/training/rl_timing.csv)
covers updates 0 through 299, adapter broadcasts, and checkpoints through 300.
The estimate sums these selected-run timing records.

The estimate amortizes against Claude after 696.9 successful escalations,
approximately 700.

The verifier recomputes both stage costs, the combined total, and the Claude
break-even point:

```bash
.venv/bin/python scripts/paper/verify_artifact.py
```
