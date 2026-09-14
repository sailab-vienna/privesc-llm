# Inference Cost Estimation

The paper reports batched model-serving cost per successful root at the primary
20-round budget. Local-model estimates combine vLLM measurements on one RTX 5090
with token counts from the released static traces.

## Calculation

For effective input and output prices `p_in` and `p_out` per million tokens:

$$
C_{run} = \frac{p_{in} E[N_{in}] + p_{out} E[N_{out}]}{10^6}
$$

$$
C_{root} = \frac{C_{run}}{P(H_{root} \le 20)}
$$

The local hourly-cost assumptions are:

| Input | Value |
|---|---:|
| RTX 5090 MSRP | $1,999 |
| Host excluding GPU | $700 |
| Lifetime and utilization | 3 years, 50% |
| GPU plus host power | 0.675 kW |
| EU non-household electricity | EUR 0.1837/kWh |
| EUR/USD | 1.1595 |

These inputs give $0.2054/hour capital cost, $0.1438/hour electricity,
$0.3492/hour total, and $9.6994e-5/second. The electricity input is the
[Eurostat H2 2025 EU non-household average](https://ec.europa.eu/eurostat/databrowser/view/nrg_pc_205/default/table)
including non-recoverable taxes and levies.

## Serving measurements

The concurrency-16 measurements used for the paper are:

| Model | Wall time | Requests | Input tokens | Output tokens | Throughput |
|---|---:|---:|---:|---:|---:|
| PrivEsc-LLM 4B | 197.5013 s | 48 | 540,739 | 52,497 | 3,003.7 tok/s |
| Gemma4 31B BnB | 516.2649 s | 48 | 191,130 | 15,863 | 400.9 tok/s |

Both use vLLM 0.21.0, `max_model_len=32768`,
`gpu_memory_utilization=0.90`, and `max_num_seqs=64`. Gemma additionally uses
`max_num_batched_tokens=4096` and its BitsAndBytes checkpoint.

Effective prices are fitted by replaying archived agent prompts, regressing
whole-request wall time on input and output tokens, and scaling the coefficients
to the measured concurrency-16 mixed-workload cost:

| Model family | Fit R2 | Input $/1M | Output $/1M |
|---|---:|---:|---:|
| Qwen3 4B | 0.958 | $0.017716 | $0.176256 |
| Gemma4 31B BnB | 0.622 | $0.095859 | $1.899536 |

These prices are applied unchanged to the final evaluation traces.

## Results

All values below are recomputed from the 120 released traces per model through
round 20:

| Model | Success | Cost/successful root |
|---|---:|---:|
| Qwen3 4B Base | 49/120 | $0.00414248 |
| Qwen3 4B SFT | 95/120 | $0.00268980 |
| PrivEsc-LLM 4B | 112/120 | **$0.00212905** |
| Gemma4 31B BnB | 97/120 | $0.01170564 |
| DeepSeek V3.2 | 63/120 | $0.02847646 |
| Claude Opus 4.7 | 120/120 | $0.17587358 |

Gemma and Claude cost 5.4980 and 82.6065 times PrivEsc-LLM respectively.
The rebuttal's DeepSeek V4 Flash baseline costs $0.00468650 per successful root
under its recorded API prices, 2.2012 times the PrivEsc-LLM local estimate.

The numeric inputs are in the pinned
[evaluation artifact](https://huggingface.co/datasets/sailab-vienna/privesc-llm-evals/tree/5dba67dfd4c16da78a720c16f01e93820cb3471c/evaluation_artifacts/provenance/cost/rtx5090).
Run the shared check from the repository root:

```bash
.venv/bin/python scripts/paper/verify_artifact.py
```

This check recomputes the hardware cost, selected benchmark rows, price fits,
final trace costs, and reported ratios.
