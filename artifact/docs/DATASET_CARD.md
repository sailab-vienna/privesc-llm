# PrivEsc SFT Dataset Card

## Dataset Summary

- **Name**: `privesc_sft`
- **Task**: Supervised fine-tuning traces for Linux privilege-escalation agents
- **Data type**: Multi-turn tool-augmented chat trajectories (`system`, `user`, `assistant`, `tool`)
- **Primary use**: Train/evaluate SFT models on procedural privilege-escalation scenarios
- **Released artifact root**: `sailab-vienna/privesc-llm-data/paper_sft_dataset`
- **Canonical default**: `standard/unguided/deepseek/long_reasoning`

The live procedural training mix is the `training` generator profile:
`capabilities_gtfobins`, `suid_gtfobins`, `sudo_gtfobins`, `cron_wildcard`, `cron_writable_script`, `password_file`, `password_history`, `password_reuse`, `ssh_key_reuse`, `weak_password`.
Validation trace collection and procedural evaluation use the same generator keys with the value-disjoint `holdout` generator profile.
Benchmark-specific exclusion rules are defined in
`src/generators/holdout_manifest.py` and audited by
`conf/experiment/audit/split_leakage.yaml`.

## Dataset Structure

```text
paper_sft_dataset/
  training/
    <scenario>/traces.jsonl
    <scenario>/stats.json
    stats.json
    tools.json
  validation/
    <scenario>/traces.jsonl
    <scenario>/stats.json
    stats.json
    tools.json
```

Generator families in the canonical dataset:

- `capabilities_gtfobins`
- `suid_gtfobins`
- `sudo_gtfobins`
- `cron_wildcard`
- `cron_writable_script`
- `password_file`
- `password_history`
- `password_reuse`
- `ssh_key_reuse`
- `weak_password`

## Split Composition

- **Training**: 2000 examples total (200 per generator)
- **Validation**: 200 examples total (20 per generator)
- **Balancing**: Uniform per-generator counts in both splits

## Data Collection and Provenance

### Collection

- Collection experiments:
  - `conf/experiment/trace/standard_guided_deepseek_training.yaml`
  - `conf/experiment/trace/standard_guided_deepseek_validation.yaml`
  - `conf/experiment/trace/standard_unguided_deepseek_training.yaml`
  - `conf/experiment/trace/standard_unguided_deepseek_validation.yaml`
- Main teacher model: `deepseek/deepseek-v4-flash` via OpenRouter, restricted
  to the DeepSeek provider
- Collection max turns: `15`
### Assembly

- Assembly script: `scripts/collect_and_assemble.sh`
- Assembler module: `src/dataset/privesc/sft.py`
- Configs:
  - `conf/datasets/sft/collection/standard/training.yaml`
  - `conf/datasets/sft/collection/standard/validation.yaml`
  - `conf/datasets/sft/quality/guided.yaml`
  - `conf/datasets/sft/quality/unguided.yaml`

### Split policy

- Training split config:
  - `source_dir: training`
  - `max_per_generator: 200`
  - generator profile: `training`
- Validation split config:
  - `source_dir: validation`
  - `exclude_source_dir: training`
  - `fail_on_split_collision: true`
  - `max_per_generator: 20`
  - generator profile: `holdout`
- Seed policy:
  - Training collection uses deterministic base seed `42`
  - Validation collection uses deterministic base seed `10_000_000`
  - Seeds are derived as `base_seed + run_index`, creating non-overlapping trace IDs under configured run counts
  - Leakage isolation comes from the disjoint generator profiles, not from the seed ranges
- Derived variants keep split ordering, seeds, and filenames from the quality-passing `long_reasoning` source traces. Derivation runs are selected by Hydra configs in `conf/experiment/derive/`. `short_reasoning` uses exactly one DeepSeek structured-output rewrite call per passing trace and records deterministic `sft_reasoning_variant` provenance in trace metadata.

## Quality Filtering and Controls

Filter controls are defined in `conf/datasets/sft/quality/shared.yaml`, with
regime-specific overlays in `guided.yaml` and `unguided.yaml`.

- Turn/token limits:
  - `max_turns: 15`
  - `min_turns: 2`
  - `max_tokens: 32768`
- Behavioral quality:
  - `min_reasoning_length: 60`
  - `max_nudges: 0`
- Shared reject conditions enabled:
  - failed traces
  - empty assistant reasoning
  - HTML entities in assistant/tool text
  - container misconfiguration indicators (expected binaries missing)
  - holdout leakage matches

## Leakage Audit (Assembled Dataset)

The released audit verifies disjoint procedural profiles and zero model-visible
solution-marker hits across 13,200 examples using 14 marker phrases. The audit
results and holdout tables are included in `leakage_audit/` in the data artifact.

## Schema

### Example-level keys

- `messages`
- `scenario`
- `num_tokens`
- `run_id`
- `success`
- `turns`
- `final_message`
- `model`
- `mode`
- `cost`
- `prompt_tokens`
- `completion_tokens`
- `tools`
- `metadata`
- `quality_metrics`

### Message-level keys

- `role`
- `content`
- `tool_calls`
- `tool_call_id`
- `name`

Observed message roles: `system`, `user`, `assistant`, `tool`.

## Integrity and Reproducibility

- The release artifact includes `paper_sft_dataset/`.
- Verify `training/stats.json` and `validation/stats.json` before training; expected split sizes are `2000` and `200`.
- Follow [ARTIFACT.md](../ARTIFACT.md) to verify the pinned release and its audit results.

## Intended Use

- Train SFT policies for privilege-escalation reasoning and tool use in controlled procedural environments.
- Evaluate transfer to held-out static benchmark scenarios using `docs/PAPER_REPRODUCTION_RUNBOOK.md`.

## Out-of-Scope / Misuse Risks

- This dataset encodes exploit-oriented actions and command patterns for privileged access in Linux environments.
- It must not be used for unauthorized access, real-world compromise, or bypassing legal/organizational policy.

## Limitations

- Data reflects procedural generator and prompt design choices in this repository.
- Collected traces are model-generated and may encode model-specific style/bias.

## Maintenance

- Regenerate with:
  - `scripts/collect_and_assemble.sh unguided deepseek training`
  - `scripts/collect_and_assemble.sh unguided deepseek validation`
