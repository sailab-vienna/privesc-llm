# Inference Cost Estimation Methodology

This document defines the paper-facing local inference cost estimate for PrivEsc-LLM. The current primary estimate is a **serving-only, batched vLLM cost** measured on an RTX 5090 and replaying token-length distributions from existing static benchmark traces.

The older RTX 4090 no-batching estimate is retained as historical context, but the recommended paper comparison is the RTX 5090 corporate local-serving estimate below.

## 1. Cost model

For batched serving, prefill and decode are co-scheduled by vLLM, so we report a **mixed-workload token cost** rather than separate input/output token prices. This avoids double-counting the same GPU wall time.

For a benchmark run:

$$ C_{sec} = \frac{C_{hr}}{3600} $$

$$ C_{mixed}^{1M} = C_{sec} \cdot \frac{T_{bench}}{N_{in,bench} + N_{out,bench}} \cdot 10^6 $$

For static benchmark traces at a round budget $R$:

$$ C_{run}(R) = C_{mixed}^{1M} \cdot \frac{\mathbb{E}[N_{in,run}(R) + N_{out,run}(R)]}{10^6} $$

$$ C_{root}(R) = \frac{C_{run}(R)}{P(\text{root} \mid R)} $$

We use $R=20$ for the paper headline cost.

## 2. Hardware cost assumptions

Primary paper setting: **corporate local deployment, RTX 5090 at MSRP, non-household electricity**.

Inputs as of 2026-05-24:

| Input | Value |
|---|---:|
| RTX 5090 MSRP | $1,999 |
| Host system excluding GPU | $700 |
| Lifetime | 3 years |
| Utilization | 50% |
| GPU power | 575 W |
| Host power | 100 W |
| Total power | 0.675 kW |
| EU non-household electricity | €0.1837/kWh |
| EUR/USD | 1.1595 |
| Electricity | $0.2130/kWh |

Hourly cost:

$$ C_{capex} = \frac{1999 + 700}{3 \cdot 365 \cdot 24 \cdot 0.5} = \$0.2054/hr $$

$$ C_{energy} = 0.675 \cdot (0.1837 \cdot 1.1595) = \$0.1438/hr $$

$$ C_{hr} = C_{capex} + C_{energy} = \$0.3492/hr $$

$$ C_{sec} = \$0.3492 / 3600 = \$0.00009699/s $$

## 3. Measured RTX 5090 serving benchmarks

Released cost evidence:

- Headline aggregate summary: `privesc-llm-evals/summaries/runs/paper/04_static_benchmark/comparison/headline/20260524/eval/static/aggregate/final/stats/evaluation_summary_paper.json`
- Headline per-run accounting: `privesc-llm-evals/summaries/runs/paper/04_static_benchmark/comparison/headline/20260524/eval/static/aggregate/final/stats/runs.csv`
- Headline static traces: `privesc-llm-evals/headline_static_traces/`

### 3.1 Final PrivEsc-LLM 4B

Model:

- Base: `Qwen/Qwen3-4B-Instruct-2507`
- LoRA: `privesc-llm-4b/rl_adapter/`

Serving:

- vLLM 0.21.0
- `max_model_len=32768`
- `gpu_memory_utilization=0.90`
- `max_num_seqs=64`
- FlashInfer attention and FlashInfer sampling enabled

Selected benchmark point: concurrency 16, the throughput/latency knee.

| Quantity | Value |
|---|---:|
| Wall time | 197.5013 s |
| Requests | 48 |
| Input tokens | 540,739 |
| Output tokens | 52,497 |
| Mixed tokens | 593,236 |
| Mixed throughput | 3,003.7 tok/s |
| p50 / p95 / p99 latency | 55.47 / 91.93 / 94.69 s |
| Max VRAM | 29,906 MiB |
| Mean GPU utilization | 99.3% |

Mixed token price:

$$ C_{mixed}^{1M} = 0.00009699 \cdot \frac{197.5013}{593236} \cdot 10^6 = \$0.03229/1M $$

Static trace source:

`privesc-llm-evals/headline_static_traces/qwen3-4b-instruct-2507/sft_sd2026_outcome_cost_step300/20260519T163319Z/eval/static/traces/qwen3_4b_sft_unguided_phase_c_lr1p5e-4_r8_sd2026_static`

Released aggregate at $R \le 20$:

| Quantity | Value |
|---|---:|
| Runs | 120 |
| Successes | 112 |
| Success rate | 93.3% |
| Mean cost/run | $0.001987 |
| Expected cost/successful root | $0.002129 |

Cost cross-check:

$$ C_{root} = 0.001987 / 0.9333 = \$0.002129 $$

**Final PrivEsc-LLM 4B headline local serving cost: $0.002129 per successful root at $R \le 20$.**

### 3.2 Gemma4 31B BnB

Model:

- `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`

Serving:

- vLLM 0.21.0
- BitsAndBytes quantized checkpoint
- `max_model_len=32768`
- `gpu_memory_utilization=0.90`
- `max_num_seqs=64`
- `max_num_batched_tokens=4096`
- FlashInfer sampling enabled
- vLLM selected `TRITON_ATTN`; forced FlashInfer attention is invalid for this Gemma4 multimodal configuration (`partial multimodal token full attention not supported`)

Selected benchmark point: concurrency 16, matching the final 4B cost calculation for a fair local price comparison. This is not Gemma4's latency-optimal point; concurrency 16 has substantially higher tail latency than concurrency 1.

| Quantity | Value |
|---|---:|
| Wall time | 516.2649 s |
| Requests | 48 |
| Input tokens | 191,130 |
| Output tokens | 15,863 |
| Mixed tokens | 206,993 |
| Mixed throughput | 400.9 tok/s |
| p50 / p95 / p99 latency | 128.59 / 228.26 / 250.36 s |
| Max VRAM | 30,162 MiB |
| Mean GPU utilization | 99.8% |

Mixed token price:

$$ C_{mixed}^{1M} = 0.00009699 \cdot \frac{516.2649}{206993} \cdot 10^6 = \$0.24191/1M $$

Static trace source:

`privesc-llm-evals/headline_static_traces/gemma4-31b/base/20260418/eval/static/raw/final/traces/gemma4-31b`

Trace statistics at $R \le 20$:

| Quantity | Value |
|---|---:|
| Runs | 120 |
| Successes | 97 |
| Success rate | 80.8% |
| Mean input tokens/run | 35,946.0 |
| Mean output tokens/run | 3,167.3 |
| Mean mixed tokens/run | 39,113.2 |

Cost:

$$ C_{run} = 39113.2 / 10^6 \cdot 0.24191 = \$0.009462 $$

$$ C_{root} = 0.009462 / 0.8083 = \$0.01171 $$

**Gemma4 31B BnB local serving cost: $0.01171 per successful root at $R \le 20$ using the matched concurrency-16 benchmark point.**

## 4. Paper comparison

| Model | Success at $R \le 20$ | Mixed serving $/1M tokens | Cost/run | Cost/successful root |
|---|---:|---:|---:|---:|
| Final PrivEsc-LLM 4B | 93.3% | $0.0323 | $0.00199 | **$0.002129** |
| Gemma4 31B BnB | 80.8% | $0.2419 | $0.00946 | **$0.011706** |

Under matched RTX 5090 MSRP + corporate non-household electricity assumptions and matched concurrency-16 benchmark points, Gemma4 31B BnB is:

$$ \frac{0.011706}{0.002129} = 5.50\times $$

more expensive per successful root than the final PrivEsc-LLM 4B model, while also having lower static benchmark success at $R \le 20$.

## 5. Input/output price split for `src/utils/pricing.py`

The paper-facing aggregate cost is still the measured mixed wall-clock serving cost above. To obtain separate input/output prices for token accounting, we use a small replay benchmark:

1. Reconstruct actual per-turn chat prompts from the final static-evaluation traces.
2. Stratify naturally occurring LLM calls into six buckets by observed output/input token ratio.
3. Replay those real prompts against the same vLLM server at concurrency 16, forcing generation to the observed completion-token count with `ignore_eos`.
4. Fit `wall_s = alpha * input_tokens + beta * output_tokens` across bucket-level whole-request timings.
5. Scale both fitted coefficients to match the full mixed-workload concurrency-16 cost/run above, so the input/output split preserves the measured aggregate serving cost.

Reproduction script, after starting the corresponding vLLM OpenAI-compatible server:

```bash
uv run --no-project python scripts/paper/replay_trace_token_price_bench.py \
  --trace-dir privesc-llm-evals/headline_static_traces/qwen3-4b-instruct-2507/sft_sd2026_outcome_cost_step300/20260519T163319Z/eval/static/traces/qwen3_4b_sft_unguided_phase_c_lr1p5e-4_r8_sd2026_static \
  --model e4_outcome_cost_step_300 \
  --api-base http://127.0.0.1:18082/v1 \
  --out-dir /tmp/privesc-llm-cost-replay/qwen3_4b_5090_trace_replay_phasefit \
  --concurrency 16 \
  --buckets 6 \
  --requests-per-bucket 12 \
  --runs-per-scenario 10 \
  --max-rounds 20 \
  --max-observed-total-tokens 28000 \
  --anchor-cost-per-run-usd 0.0019871163883107236 \
  --seed 2026

uv run --no-project python scripts/paper/replay_trace_token_price_bench.py \
  --trace-dir privesc-llm-evals/headline_static_traces/gemma4-31b/base/20260418/eval/static/raw/final/traces/gemma4-31b \
  --model gemma4_31b_bnb_5090 \
  --api-base http://127.0.0.1:18083/v1 \
  --out-dir /tmp/privesc-llm-cost-replay/gemma4_31b_bnb_5090_trace_replay_phasefit \
  --concurrency 16 \
  --buckets 6 \
  --requests-per-bucket 12 \
  --runs-per-scenario 10 \
  --max-rounds 20 \
  --max-observed-total-tokens 28000 \
  --anchor-cost-per-run-usd 0.009462058302784933 \
  --seed 2026
```

The `28000` observed-token cap excludes only replay calls close enough to the 32k context limit that chat-template overhead can make replay invalid, while preserving the trace-level aggregate cost anchor.

Anchored prices used in `src/utils/pricing.py`:

| Model | Raw fit R² | `input_cost_usd_per_1M` | `output_cost_usd_per_1M` | Preserved cost/root r≤20 |
|---|---:|---:|---:|---:|
| Final PrivEsc-LLM 4B / Qwen3-4B family | 0.958 | $0.017716 | $0.176256 | $0.002129 |
| Gemma4 31B BnB | 0.622 | $0.095859 | $1.899536 | $0.01171 |

These are effective workload prices for the benchmarked vLLM setup, not universal prefill/decode hardware constants.

## 6. Historical RTX 4090 no-batching estimate

The previous estimate used a single-agent RTX 4090 latency model with no batching:

- Input: $0.00135 / 1M input tokens
- Output: $0.974 / 1M output tokens
- Final 4B replay on current headline static traces: $0.00634 per successful root at $R \le 20$

This is useful as a conservative no-batching sensitivity, but it is no longer the primary local serving estimate.

## 7. References

- NVIDIA RTX 5090 MSRP and TGP: NVIDIA GeForce RTX 5090 specifications
- EU non-household electricity: Eurostat, H2 2025 EU average including non-recoverable taxes/levies
- EUR/USD: ECB reference rate, 2026-05-22
- vLLM benchmark artifacts listed in Section 3
