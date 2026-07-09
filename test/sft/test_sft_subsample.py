import json
from pathlib import Path

import pytest

from src.config import SFTSubsampleConfig
from src.dataset.privesc.sft_reporting import calculate_stats
from src.dataset.privesc.sft_subsample import (
    create_balanced_subset,
    expected_output_profile,
)


def _row(scenario: str, index: int) -> dict:
    return {
        "scenario": scenario,
        "messages": [{"role": "user", "content": f"{scenario}-{index}"}],
        "metadata": json.dumps({"generator_name": scenario, "seed": index}),
        "num_tokens": 100 + index,
        "turns": 2,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _source_dataset(project_dir: Path, *, include_validation: bool = True) -> None:
    root = (
        project_dir
        / "outputs/data/privesc_sft/standard/unguided/deepseek/long_reasoning"
    )
    splits = [("training", 4)]
    if include_validation:
        splits.append(("validation", 2))
    for split, count in splits:
        split_root = root / split
        (split_root / "tools.json").parent.mkdir(parents=True, exist_ok=True)
        (split_root / "tools.json").write_text('{"tools": []}\n', encoding="utf-8")
        (split_root / "config_snapshot.json").write_text("{}\n", encoding="utf-8")
        all_rows = []
        for scenario in ("alpha", "beta"):
            rows = [_row(scenario, i) for i in range(count)]
            _write_jsonl(
                split_root / scenario / "traces.jsonl",
                rows,
            )
            (split_root / scenario / "stats.json").write_text(
                json.dumps(calculate_stats(rows)) + "\n",
                encoding="utf-8",
            )
            all_rows.extend(rows)
        (split_root / "stats.json").write_text(
            json.dumps(calculate_stats(all_rows)) + "\n",
            encoding="utf-8",
        )


def _subsample(n: int) -> SFTSubsampleConfig:
    return SFTSubsampleConfig(n=n, output_profile=expected_output_profile(n))


def test_create_balanced_subset_preserves_counts_and_order(tmp_path: Path) -> None:
    _source_dataset(tmp_path)

    manifest = create_balanced_subset(
        project_dir=tmp_path,
        subsample=_subsample(4),
        config_snapshot={"datasets": {"sft": {"profile": "standard"}}},
    )

    root = (
        tmp_path
        / "outputs/data/privesc_sft/data_scaling_n0004/unguided/deepseek/long_reasoning"
    )
    assert manifest["training_num_runs"] == 4
    assert manifest["validation_num_runs"] == 4
    assert manifest["training_counts"] == {"alpha": 2, "beta": 2}
    assert (root / "subset_manifest.json").exists()
    assert (root / "training/stats.json").exists()
    assert (root / "validation/stats.json").exists()
    config_snapshot = json.loads(
        (root / "training/config_snapshot.json").read_text(encoding="utf-8")
    )
    assert config_snapshot["datasets"]["sft"]["profile"] == "data_scaling_n0004"
    assert config_snapshot["datasets"]["sft"]["subsample"]["n"] == 4

    alpha_rows = _read_jsonl(root / "training/alpha/traces.jsonl")
    beta_rows = _read_jsonl(root / "training/beta/traces.jsonl")
    assert [row["metadata"] for row in alpha_rows] == [
        json.dumps({"generator_name": "alpha", "seed": 0}),
        json.dumps({"generator_name": "alpha", "seed": 1}),
    ]
    assert [row["metadata"] for row in beta_rows] == [
        json.dumps({"generator_name": "beta", "seed": 0}),
        json.dumps({"generator_name": "beta", "seed": 1}),
    ]


def test_create_balanced_subset_refuses_existing_output(tmp_path: Path) -> None:
    _source_dataset(tmp_path)
    create_balanced_subset(
        project_dir=tmp_path,
        subsample=_subsample(4),
    )

    with pytest.raises(FileExistsError):
        create_balanced_subset(
            project_dir=tmp_path,
            subsample=_subsample(4),
        )


def test_create_balanced_subset_validates_before_writing(tmp_path: Path) -> None:
    _source_dataset(tmp_path, include_validation=False)

    with pytest.raises(FileNotFoundError, match="Validation dataset"):
        create_balanced_subset(
            project_dir=tmp_path,
            subsample=_subsample(4),
        )

    assert not (
        tmp_path
        / "outputs/data/privesc_sft/data_scaling_n0004/unguided/deepseek/long_reasoning"
    ).exists()


def test_create_balanced_subset_rejects_invalid_scientific_axis(
    tmp_path: Path,
) -> None:
    _source_dataset(tmp_path)

    with pytest.raises(ValueError, match="output_profile"):
        create_balanced_subset(
            project_dir=tmp_path,
            subsample=SFTSubsampleConfig(n=4, output_profile="data_scaling_n0005"),
        )

    with pytest.raises(ValueError, match="divisible"):
        create_balanced_subset(
            project_dir=tmp_path,
            subsample=_subsample(5),
        )
