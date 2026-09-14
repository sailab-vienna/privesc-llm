import json

import pytest

from src.dataset.privesc.audit import (
    benchmark_holdout_table_rows,
    dataset_visibility_scan,
    eval_scope,
    selection_ledger,
    trace_leakage_summary,
)
from src.generators.profile_audit import generator_profile_diff_rows


def test_generator_profile_diff_rows_reports_disjoint_status():
    training = {"capabilities_allowlist": ["php", "ruby"]}
    validation = {"capabilities_allowlist": ["python3"]}

    rows = generator_profile_diff_rows(training, validation)
    row = next(row for row in rows if row["category"] == "capabilities GTFOBins")

    assert row["status"] == "disjoint"
    assert row["overlap_count"] == 0


def test_generator_profile_diff_rows_reports_overlap_status():
    training = {"capabilities_allowlist": ["php", "ruby"]}
    validation = {"capabilities_allowlist": ["ruby", "python3"]}

    rows = generator_profile_diff_rows(training, validation)
    row = next(row for row in rows if row["category"] == "capabilities GTFOBins")

    assert row["status"] == "overlap"
    assert row["overlap_count"] == 1
    assert json.loads(row["overlap_values"]) == ["ruby"]


def test_dataset_visibility_scan_separates_messages_from_metadata(tmp_path):
    root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="No dataset examples"):
        dataset_visibility_scan(
            root, "**/traces.jsonl", ["secret solution data"], tmp_path
        )
    traces = root / "guided" / "deepseek" / "long_reasoning" / "training" / "gen"
    traces.mkdir(parents=True)
    (traces / "traces.jsonl").write_text("")
    with pytest.raises(ValueError, match="No dataset examples"):
        dataset_visibility_scan(
            root, "**/traces.jsonl", ["secret solution data"], tmp_path
        )
    payload = {
        "messages": [{"role": "assistant", "content": "no hidden guidance here"}],
        "metadata": '{"note": "secret solution data should stay out of messages"}',
    }
    (traces / "traces.jsonl").write_text(json.dumps(payload) + "\n")

    scan = dataset_visibility_scan(
        root,
        "**/traces.jsonl",
        ["secret solution data"],
        tmp_path,
    )

    assert scan["totals"]["examples"] == 1
    assert scan["totals"]["message_hit_examples"] == 0
    assert scan["totals"]["metadata_hit_examples"] == 1


def test_benchmark_holdout_rows_expose_rule_patterns():
    rows = benchmark_holdout_table_rows()

    assert rows
    assert {
        "generator",
        "rule_id",
        "benchmark_cases",
        "category",
        "match",
        "pattern",
        "hard_rejection_class",
    } <= set(rows[0])
    assert any(row["pattern"] == "aim8Du7h" for row in rows)


def test_trace_leakage_summary_reports_filter_counts(tmp_path):
    stats = tmp_path / "stats"
    with pytest.raises(ValueError, match="No trace leakage rows"):
        trace_leakage_summary(stats, "**/trace_leakage.jsonl", tmp_path)
    stats.mkdir()
    (stats / "trace_leakage.jsonl").write_text("")
    with pytest.raises(ValueError, match="No trace leakage rows"):
        trace_leakage_summary(tmp_path, "**/trace_leakage.jsonl", tmp_path)
    (stats / "trace_leakage.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "base_usable": True,
                        "audit_usable": False,
                        "sft_usable": False,
                        "hard_rejection_classes": ["exact_benchmark_password"],
                        "sft_rejection_reasons": ["holdout_leakage"],
                    }
                ),
                json.dumps(
                    {
                        "base_usable": True,
                        "audit_usable": True,
                        "sft_usable": True,
                        "hard_rejection_classes": [],
                        "sft_rejection_reasons": [],
                    }
                ),
            ]
        )
        + "\n"
    )

    summary = trace_leakage_summary(tmp_path, "**/trace_leakage.jsonl", tmp_path)

    assert summary["totals"]["rows"] == 2
    assert summary["totals"]["base_usable"] == 2
    assert summary["totals"]["audit_usable"] == 1
    assert summary["totals"]["sft_usable"] == 1
    assert summary["totals"]["benchmark_holdout_rejected"] == 1
    assert summary["totals"]["hard_rejection_class_counts"] == {
        "exact_benchmark_password": 1
    }
    assert summary["totals"]["sft_rejection_reason_counts"] == {"holdout_leakage": 1}


def test_eval_scope_classifies_configured_eval_roles(tmp_path):
    with pytest.raises(ValueError, match="No evaluation provenance files"):
        selection_ledger(
            tmp_path,
            [],
            procedural_experiment="eval/paper_procedural",
            static_experiment="eval/benchmark",
            static_prefix="eval/paper_static",
            project_dir=tmp_path,
        )
    assert (
        eval_scope(
            "eval/paper_procedural",
            "eval/paper_procedural",
            "eval/benchmark",
            "eval/paper_static",
        )
        == "procedural_validation"
    )
    assert (
        eval_scope(
            "eval/paper_static_qwen3_4b_base",
            "eval/paper_procedural",
            "eval/benchmark",
            "eval/paper_static",
        )
        == "static_benchmark"
    )
