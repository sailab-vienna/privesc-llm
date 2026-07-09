#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def benchmark_eligible(payload: dict[str, Any]) -> bool:
    return payload.get("metadata", {}).get("benchmark_eligible") is True


def item_run_ordinal(payload: dict[str, Any]) -> int:
    value = payload.get("metadata", {}).get("item_run_ordinal")
    return int(value) if value is not None else 10**9


def protocol_traces(trace_dir: Path, runs_per_scenario: int) -> list[tuple[Path, dict[str, Any]]]:
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(trace_dir.glob("*.json")):
        payload = load_json(path)
        if benchmark_eligible(payload):
            grouped.setdefault(str(payload["scenario"]), []).append((path, payload))

    traces = []
    for scenario in sorted(grouped):
        rows = sorted(grouped[scenario], key=lambda item: (item_run_ordinal(item[1]), item[0].name))
        traces.extend(rows[:runs_per_scenario])
    return traces


def trace_summary(traces: list[tuple[Path, dict[str, Any]]], max_rounds: int) -> dict[str, Any]:
    per_run = []
    for _, payload in traces:
        input_tokens = 0
        output_tokens = 0
        for usage in payload.get("llm_usage_by_turn", [])[:max_rounds]:
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
        per_run.append(
            {
                "scenario": payload.get("scenario"),
                "success": bool(payload.get("success") and int(payload.get("turns") or 10**9) <= max_rounds),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
    successes = sum(row["success"] for row in per_run)
    return {
        "runs": len(per_run),
        "successes": successes,
        "success_rate": successes / len(per_run),
        "mean_input_tokens_per_run": statistics.mean(row["input_tokens"] for row in per_run),
        "mean_output_tokens_per_run": statistics.mean(row["output_tokens"] for row in per_run),
    }


def normalized_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        row = dict(message)
        content = row.get("content")
        if isinstance(content, list):
            row["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        normalized.append(row)
    return normalized


def extract_turn_requests(traces: list[tuple[Path, dict[str, Any]]], max_rounds: int, max_observed_total_tokens: int) -> list[dict[str, Any]]:
    requests = []
    for path, payload in traces:
        history = normalized_messages(payload.get("history", []))
        assistant_indices = [idx for idx, message in enumerate(history) if message.get("role") == "assistant"]
        usages = payload.get("llm_usage_by_turn", [])[:max_rounds]
        for turn_index, (assistant_idx, usage) in enumerate(zip(assistant_indices, usages, strict=False), start=1):
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            if input_tokens <= 0 or output_tokens <= 0:
                continue
            if max_observed_total_tokens and input_tokens + output_tokens > max_observed_total_tokens:
                continue
            requests.append(
                {
                    "messages": history[:assistant_idx],
                    "tools": payload.get("tools") or None,
                    "max_tokens": output_tokens,
                    "trace_path": str(path),
                    "scenario": payload.get("scenario"),
                    "turn": turn_index,
                    "observed_input_tokens": input_tokens,
                    "observed_output_tokens": output_tokens,
                    "observed_output_input_ratio": output_tokens / input_tokens,
                }
            )
    return requests


def make_buckets(requests: list[dict[str, Any]], buckets: int, requests_per_bucket: int, seed: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(requests, key=lambda row: row["observed_output_input_ratio"])
    rng = random.Random(seed)
    result = []
    for bucket_idx in range(buckets):
        start = len(ordered) * bucket_idx // buckets
        end = len(ordered) * (bucket_idx + 1) // buckets
        bucket = ordered[start:end]
        if not bucket:
            raise ValueError(f"bucket {bucket_idx} is empty; reduce --buckets or use more trace requests")
        if len(bucket) > requests_per_bucket:
            bucket = rng.sample(bucket, requests_per_bucket)
        rng.shuffle(bucket)
        result.append(bucket)
    return result


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def send_one_sync(url: str, model: str, request: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": request["messages"],
        "tools": request["tools"],
        "max_tokens": request["max_tokens"],
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": True,
    }
    if payload["tools"] is None:
        del payload["tools"]
    body = json.dumps(payload).encode()
    http_request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
            text = response.read().decode()
            usage = (json.loads(text).get("usage") or {})
            return {
                "ok": True,
                "latency_s": time.perf_counter() - started,
                "status": response.status,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "latency_s": time.perf_counter() - started, "status": exc.code, "error": exc.read().decode(errors="replace")[:500]}
    except Exception as exc:
        return {"ok": False, "latency_s": time.perf_counter() - started, "status": None, "error": repr(exc)}


async def run_bucket(url: str, model: str, bucket: list[dict[str, Any]], concurrency: int, timeout_s: float) -> tuple[float, list[dict[str, Any]]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def task(request: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            result = await asyncio.to_thread(send_one_sync, url, model, request, timeout_s)
            public_request = {key: request[key] for key in ("trace_path", "scenario", "turn", "observed_input_tokens", "observed_output_tokens", "observed_output_input_ratio")}
            return {**public_request, **result}

    started = time.perf_counter()
    results = await asyncio.gather(*(task(request) for request in bucket))
    return time.perf_counter() - started, results


def summarize_bucket(bucket_idx: int, bucket: list[dict[str, Any]], wall_s: float, results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if row.get("ok")]
    failed = [row for row in results if not row.get("ok")]
    latencies = [float(row["latency_s"]) for row in ok]
    ratios = [row["observed_output_input_ratio"] for row in bucket]
    input_tokens = sum(int(row.get("prompt_tokens") or 0) for row in ok)
    output_tokens = sum(int(row.get("completion_tokens") or 0) for row in ok)
    return {
        "bucket": bucket_idx,
        "requests": len(results),
        "successes": len(ok),
        "failures": len(failed),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "wall_s": wall_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_tok_per_s": input_tokens / wall_s,
        "output_tok_per_s": output_tokens / wall_s,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
    }


def fit_prices(rows: list[dict[str, Any]], c_sec: float) -> dict[str, Any]:
    s_xx = sum(row["input_tokens"] ** 2 for row in rows)
    s_yy = sum(row["output_tokens"] ** 2 for row in rows)
    s_xy = sum(row["input_tokens"] * row["output_tokens"] for row in rows)
    s_xt = sum(row["input_tokens"] * row["wall_s"] for row in rows)
    s_yt = sum(row["output_tokens"] * row["wall_s"] for row in rows)
    det = s_xx * s_yy - s_xy * s_xy
    if det == 0:
        raise ValueError("cannot fit input/output prices because input and output token counts are collinear")
    alpha = (s_xt * s_yy - s_yt * s_xy) / det
    beta = (s_xx * s_yt - s_xy * s_xt) / det
    predictions = [alpha * row["input_tokens"] + beta * row["output_tokens"] for row in rows]
    mean_wall = statistics.mean(row["wall_s"] for row in rows)
    sse = sum((row["wall_s"] - pred) ** 2 for row, pred in zip(rows, predictions, strict=True))
    sst = sum((row["wall_s"] - mean_wall) ** 2 for row in rows)
    return {
        "model": "wall_s = alpha * input_tokens + beta * output_tokens",
        "alpha_s_per_input_token": alpha,
        "beta_s_per_output_token": beta,
        "input_usd_per_1m_tokens": c_sec * alpha * 1_000_000,
        "output_usd_per_1m_tokens": c_sec * beta * 1_000_000,
        "r2": 1 - sse / sst if sst else math.nan,
        "rows": [dict(row, predicted_wall_s=pred) for row, pred in zip(rows, predictions, strict=True)],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["bucket", "requests", "successes", "failures", "ratio_min", "ratio_max", "wall_s", "input_tokens", "output_tokens", "input_tok_per_s", "output_tok_per_s", "latency_p50_s", "latency_p95_s"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fit effective input/output token prices by replaying real agent LLM-call prompts against a running vLLM OpenAI-compatible server.")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runs-per-scenario", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--buckets", type=int, default=6)
    parser.add_argument("--requests-per-bucket", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--max-observed-total-tokens", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--c-sec", type=float, default=0.00009699401383445375)
    parser.add_argument("--anchor-cost-per-run-usd", type=float, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces = protocol_traces(args.trace_dir, args.runs_per_scenario)
    if not traces:
        raise ValueError(f"no benchmark-eligible traces found in {args.trace_dir}")
    requests = extract_turn_requests(traces, args.max_rounds, args.max_observed_total_tokens)
    if not requests:
        raise ValueError("no replayable turn requests found after filtering")
    buckets = make_buckets(requests, args.buckets, args.requests_per_bucket, args.seed)
    url = args.api_base.rstrip("/") + "/chat/completions"

    bucket_rows = []
    all_results = []
    for idx, bucket in enumerate(buckets):
        wall_s, results = await run_bucket(url, args.model, bucket, args.concurrency, args.timeout_s)
        row = summarize_bucket(idx, bucket, wall_s, results)
        bucket_rows.append(row)
        all_results.extend(dict(result, bucket=idx) for result in results)
        if row["failures"]:
            failure_artifact = {
                "model": args.model,
                "trace_dir": str(args.trace_dir),
                "bucket_rows": bucket_rows,
                "request_results_without_prompts": all_results,
            }
            (args.out_dir / "trace_replay_price_fit_failed.json").write_text(json.dumps(failure_artifact, indent=2))
            raise RuntimeError(f"bucket {idx} had {row['failures']} failed requests; see trace_replay_price_fit_failed.json")

    fit = fit_prices(bucket_rows, args.c_sec)
    summary = trace_summary(traces, args.max_rounds)
    cost_per_run = (
        summary["mean_input_tokens_per_run"] * fit["input_usd_per_1m_tokens"]
        + summary["mean_output_tokens_per_run"] * fit["output_usd_per_1m_tokens"]
    ) / 1_000_000
    fit["expected_cost_per_run_usd"] = cost_per_run
    fit["expected_cost_per_successful_root_usd"] = cost_per_run / summary["success_rate"]
    if args.anchor_cost_per_run_usd is not None:
        scale = args.anchor_cost_per_run_usd / cost_per_run
        fit["anchored_to_cost_per_run_usd"] = args.anchor_cost_per_run_usd
        fit["anchor_scale"] = scale
        fit["anchored_input_usd_per_1m_tokens"] = fit["input_usd_per_1m_tokens"] * scale
        fit["anchored_output_usd_per_1m_tokens"] = fit["output_usd_per_1m_tokens"] * scale
        fit["anchored_expected_cost_per_successful_root_usd"] = args.anchor_cost_per_run_usd / summary["success_rate"]

    write_csv(args.out_dir / "trace_replay_price_fit.csv", bucket_rows)
    artifact = {
        "method": "Replay actual per-turn agent chat prompts from static-evaluation traces; stratify naturally occurring calls by observed output/input ratio; force generation to the observed completion-token count with ignore_eos; fit wall_s = alpha * input_tokens + beta * output_tokens over bucket-level whole-request timings.",
        "model": args.model,
        "trace_dir": str(args.trace_dir),
        "concurrency": args.concurrency,
        "runs_per_scenario": args.runs_per_scenario,
        "max_rounds": args.max_rounds,
        "max_observed_total_tokens": args.max_observed_total_tokens,
        "buckets": args.buckets,
        "requests_per_bucket": args.requests_per_bucket,
        "trace_summary": summary,
        "fit": fit,
        "bucket_rows": bucket_rows,
        "request_results_without_prompts": all_results,
    }
    (args.out_dir / "trace_replay_price_fit.json").write_text(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
