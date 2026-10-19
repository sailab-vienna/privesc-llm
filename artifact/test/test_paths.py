"""Unit tests for shared path utilities."""

from pathlib import Path

from src.paths import EVALUATION_BASE, model_dir_name, traces_dir, resolve_traces_dir


# -- model_dir_name -----------------------------------------------------------


def test_model_dir_name_bare():
    assert model_dir_name("gpt-5") == "gpt-5"


def test_model_dir_name_strips_provider():
    assert model_dir_name("deepseek/deepseek-v4-flash") == "deepseek-v4-flash"


def test_model_dir_name_strips_colon_suffix():
    assert model_dir_name("deepseek-v4-flash:free") == "deepseek-v4-flash"


def test_model_dir_name_strips_both():
    assert model_dir_name("deepseek/deepseek-v4-flash:free") == "deepseek-v4-flash"


def test_model_dir_name_nested_provider():
    assert model_dir_name("org/sub/model-name:v2") == "model-name"


def test_evaluation_base_constant():
    assert EVALUATION_BASE.endswith("outputs/evals/evaluation")


# -- traces_dir ---------------------------------------------------------------


def test_traces_dir_appends_model():
    result = traces_dir("outputs/traces/trace_collection/training", "gpt-5")
    assert result == "outputs/traces/trace_collection/training/traces/gpt-5"


def test_traces_dir_normalizes_model():
    result = traces_dir(
        "outputs/traces/trace_collection/training",
        "deepseek/deepseek-v4-flash:free",
    )
    assert result.endswith("/traces/deepseek-v4-flash")


# -- resolve_traces_dir -------------------------------------------------------


def test_resolve_traces_dir_found(tmp_path: Path):
    d = tmp_path / "outputs/traces/trace_collection/training/traces/m"
    d.mkdir(parents=True)
    assert resolve_traces_dir(tmp_path, "m", "training") == d


def test_resolve_traces_dir_normalizes_model(tmp_path: Path):
    d = (
        tmp_path
        / "outputs/traces/trace_collection/validation/traces/deepseek-v4-flash"
    )
    d.mkdir(parents=True)
    result = resolve_traces_dir(tmp_path, "deepseek/deepseek-v4-flash:free", "validation")
    assert result == d


def test_resolve_traces_dir_missing(tmp_path: Path):
    assert resolve_traces_dir(tmp_path, "nonexistent", "training") is None


def test_resolve_traces_dir_uses_explicit_trace_root(tmp_path: Path):
    d = (
        tmp_path
        / "outputs/traces/trace_collection/standard/unguided/deepseek/long_reasoning/training/traces/m"
    )
    d.mkdir(parents=True)

    assert (
        resolve_traces_dir(
            tmp_path,
            "m",
            "training",
            trace_root="outputs/traces/trace_collection/standard/unguided/deepseek/long_reasoning",
        )
        == d
    )


def test_resolve_traces_dir_split_isolation(tmp_path: Path):
    d = tmp_path / "outputs/traces/trace_collection/training/traces/m"
    d.mkdir(parents=True)
    assert resolve_traces_dir(tmp_path, "m", "training") is not None
    assert resolve_traces_dir(tmp_path, "m", "validation") is None
