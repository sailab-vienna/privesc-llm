import json
import logging
import os
from pathlib import Path
from typing import Any, TypedDict

from rich import get_console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from src.config import AppConfig
from src.dataset.privesc.collection_stats import (
    audit_trace_payload,
    summary_generators,
    summary_target_per_generator,
)
from src.dataset.privesc.sft_preprocessing import (
    SFTPromptNormalizer,
    record_sft_token_count,
    sft_training_content,
)
from src.dataset.privesc.sft_quality import TraceQualityFilter
from src.dataset.privesc.trace_utils import trace_messages
from src.paths import TRACE_COLLECTION_BASE, resolve_traces_dir

console = get_console()
log = logging.getLogger("privesc_sft")


class AssemblerStats(TypedDict):
    total: int
    passed: int
    filtered: dict[str, int]


class TraceOrdinalPolicy(TypedDict):
    generators: list[str]
    target: int


class TraceOrdinalRecord(TypedDict):
    file: str
    key: tuple[str, int] | None


class DatasetAssembler:
    """Load traces, run the per-trace pipeline, and emit examples."""

    def __init__(self, cfg: AppConfig, quality_filter: TraceQualityFilter):
        self.cfg = cfg
        self.quality_filter = quality_filter
        self.preprocessor = SFTPromptNormalizer(cfg)
        self.stats: AssemblerStats = {"total": 0, "passed": 0, "filtered": {}}

    def load_traces(
        self,
        teacher_model: str,
        base_dir: Path,
        prune_rejected: bool = False,
        source_dir: str = "training",
        exclude_source_dir: str | None = None,
        trace_root: str | None = None,
        max_per_generator: int | None = None,
        fail_on_split_collision: bool = False,
    ) -> list[dict[str, Any]]:
        exclude_keys = self._exclude_keys(
            base_dir,
            teacher_model,
            exclude_source_dir=exclude_source_dir,
            trace_root=trace_root,
        )
        traces_dir = resolve_traces_dir(
            base_dir, teacher_model, source_dir, trace_root=trace_root
        )
        if not traces_dir:
            log.error("Traces directory not found for teacher %s", teacher_model)
            return []

        files = sorted(traces_dir.glob("*.json"))
        if not files:
            log.warning("No json files found in %s", traces_dir)
            return []

        ordinal_policy = self._trace_collection_ordinal_policy(
            base_dir=base_dir,
            source_dir=source_dir,
            trace_root=trace_root,
            max_per_generator=max_per_generator,
        )
        examples: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int]] = set()
        seen_ordinal_keys: set[tuple[str, int]] = set()
        ordinal_records: list[TraceOrdinalRecord] = []
        kept_per_gen: dict[str, int] = {}
        split_collision_count = 0
        split_collision_files: list[str] = []
        pruned_files: list[tuple[str, list[str]]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Processing {len(files)} traces...", total=len(files)
            )

            for run_id, trace_file in enumerate(files):
                try:
                    raw_data = self._read_trace(trace_file)
                    raw_key = self._trace_key(raw_data)
                    if raw_key is not None and raw_key in exclude_keys:
                        reason = "seed_collision_with_excluded_split"
                        self.stats["total"] += 1
                        self._record_filtered_reason(reason)
                        split_collision_count += 1
                        split_collision_files.append(trace_file.name)
                        if prune_rejected and not fail_on_split_collision:
                            self._prune_file(trace_file, [reason])
                            pruned_files.append((trace_file.name, [reason]))
                        progress.advance(task)
                        continue
                    self.stats["total"] += 1
                    passed, metrics, key, trace_data = self.prepare_passing_trace(
                        raw_data
                    )
                    if not passed:
                        reasons = metrics["reasons"]
                        for reason in reasons:
                            self._record_filtered_reason(reason)
                        if prune_rejected:
                            self._prune_file(trace_file, reasons)
                            pruned_files.append((trace_file.name, reasons))
                        progress.advance(task)
                        continue
                    if trace_data is None:
                        raise ValueError("Passing trace missing processed payload")

                    ordinal_key = self._trace_ordinal_key(trace_data)
                    if ordinal_policy is not None:
                        ordinal_records.append(
                            {
                                "file": trace_file.name,
                                "key": ordinal_key,
                            }
                        )
                    if ordinal_key is not None and ordinal_key in seen_ordinal_keys:
                        reason = "duplicate_ordinal_within_split"
                        self._record_filtered_reason(reason)
                        if prune_rejected:
                            self._prune_file(trace_file, [reason])
                            pruned_files.append((trace_file.name, [reason]))
                        progress.advance(task)
                        continue

                    is_duplicate, key = self._is_duplicate_within_split(
                        trace_data, seen_keys
                    )
                    if is_duplicate:
                        reason = "duplicate_seed_within_split"
                        self._record_filtered_reason(reason)
                        if prune_rejected:
                            self._prune_file(trace_file, [reason])
                            pruned_files.append((trace_file.name, [reason]))
                        progress.advance(task)
                        continue

                    if (
                        max_per_generator is not None
                        and max_per_generator > 0
                        and key is not None
                    ):
                        gen = key[0]
                        already = kept_per_gen.get(gen, 0)
                        if already >= max_per_generator:
                            reason = "extra_trace_over_cap"
                            self._record_filtered_reason(reason)
                            if prune_rejected:
                                self._prune_file(trace_file, [reason])
                                pruned_files.append((trace_file.name, [reason]))
                            progress.advance(task)
                            continue
                        kept_per_gen[gen] = already + 1

                    if key is not None:
                        seen_keys.add(key)
                    if ordinal_key is not None:
                        seen_ordinal_keys.add(ordinal_key)
                    example = self._convert_to_example(trace_data, run_id=run_id)
                    example["quality_metrics"] = metrics
                    examples.append(example)
                    self.stats["passed"] += 1

                except Exception as exc:
                    console.log(
                        f"[yellow]Failed to process {trace_file.name}: {exc}[/]"
                    )

                progress.advance(task)

        self._report_pruned_files(pruned_files)
        if fail_on_split_collision and split_collision_count > 0:
            sample_files = ", ".join(split_collision_files[:5])
            raise ValueError(
                "Detected split collisions with excluded split: "
                f"{split_collision_count} trace(s) overlap by (generator_name, seed). "
                "Refusing to assemble dataset. "
                f"Example files: {sample_files}"
            )
        if ordinal_policy is not None:
            self._validate_trace_collection_ordinals(
                ordinal_records,
                generators=ordinal_policy["generators"],
                target=ordinal_policy["target"],
            )
        return examples

    def _trace_collection_ordinal_policy(
        self,
        *,
        base_dir: Path,
        source_dir: str,
        trace_root: str | None,
        max_per_generator: int | None,
    ) -> TraceOrdinalPolicy | None:
        summary = self._load_collection_summary(base_dir, source_dir, trace_root)
        generators = summary_generators(summary) if summary is not None else []
        summary_target = (
            summary_target_per_generator(summary) if summary is not None else None
        )

        runner_cfg = getattr(self.cfg, "runner", None)
        source_cfg = getattr(runner_cfg, "source", None)
        if not generators and (
            getattr(runner_cfg, "mode", None) == "trace_collection"
            and source_cfg is not None
            and getattr(source_cfg, "type", None) == "procedural"
        ):
            generators = [str(name) for name in getattr(source_cfg, "generators", [])]
            summary_target = int(getattr(runner_cfg, "runs_per_item", 0) or 0)

        if max_per_generator is not None and max_per_generator > 0:
            target = int(max_per_generator)
            if summary_target is not None and int(summary_target) != target:
                raise ValueError(
                    "datasets.sft.max_per_generator must match collection target: "
                    f"{target} != {summary_target}"
                )
        else:
            target = int(summary_target or 0)

        if not generators or target <= 0:
            return None
        return {"generators": generators, "target": target}

    def _load_collection_summary(
        self,
        base_dir: Path,
        source_dir: str,
        trace_root: str | None,
    ) -> dict[str, Any] | None:
        root = Path(str(trace_root)) if trace_root else Path(TRACE_COLLECTION_BASE)
        if not root.is_absolute():
            root = base_dir / root
        summary_path = root / source_dir / "stats" / "collection_summary.json"
        if not summary_path.is_file():
            return None
        with open(summary_path, "r") as f:
            summary = json.load(f)
        return summary if isinstance(summary, dict) else None

    def _exclude_keys(
        self,
        base_dir: Path,
        teacher_model: str,
        *,
        exclude_source_dir: str | None,
        trace_root: str | None,
    ) -> set[tuple[str, int]]:
        if exclude_source_dir is None:
            return set()
        traces_dir = resolve_traces_dir(
            base_dir, teacher_model, exclude_source_dir, trace_root=trace_root
        )
        if not traces_dir:
            return set()
        return self._load_exclude_keys(traces_dir)

    def _read_trace(self, trace_file: Path) -> dict[str, Any]:
        with open(trace_file, "r") as f:
            return json.load(f)

    def prepare_passing_trace(
        self,
        raw_data: dict[str, Any],
        *,
        seen_keys: set[tuple[str, int]] | None = None,
        seen_ordinal_keys: set[tuple[str, int]] | None = None,
    ) -> tuple[bool, dict[str, Any], tuple[str, int] | None, dict[str, Any] | None]:
        trace_data = self.preprocessor.normalize_trace_prompts(raw_data)
        record_sft_token_count(trace_data)
        passed, metrics = self.quality_filter.check_trace(trace_data)
        if not passed:
            return False, metrics, None, None

        audit_reasons = self._collection_audit_rejection_reasons(trace_data)
        if audit_reasons:
            return False, {**metrics, "reasons": audit_reasons}, None, None

        key: tuple[str, int] | None = None
        if seen_keys is not None:
            is_duplicate, key = self._is_duplicate_within_split(trace_data, seen_keys)
            if is_duplicate:
                return (
                    False,
                    {**metrics, "reasons": ["duplicate_seed_within_split"]},
                    key,
                    None,
                )

        return True, metrics, key, trace_data

    def _load_exclude_keys(self, traces_dir: Path) -> set[tuple[str, int]]:
        keys: set[tuple[str, int]] = set()
        for path in traces_dir.glob("*.json"):
            with open(path, "r") as f:
                data = json.load(f)
            key = self._trace_key(data)
            if key is not None:
                keys.add(key)
        return keys

    def _trace_key(self, trace_data: dict[str, Any]) -> tuple[str, int] | None:
        metadata = trace_data.get("metadata")
        if not isinstance(metadata, dict):
            return None
        gen = metadata.get("generator_name")
        seed = metadata.get("seed")
        if not isinstance(gen, str) or not gen:
            return None
        if not isinstance(seed, int):
            return None
        return gen, seed

    def _trace_ordinal_key(self, trace_data: dict[str, Any]) -> tuple[str, int] | None:
        metadata = self._trace_metadata(trace_data)
        if not metadata:
            return None
        gen = metadata.get("generator_name")
        ordinal = metadata.get("item_run_ordinal")
        if not isinstance(gen, str) or not gen:
            return None
        if not isinstance(ordinal, int):
            return None
        return gen, ordinal

    def _trace_metadata(self, trace_data: dict[str, Any]) -> dict[str, Any]:
        metadata = trace_data.get("metadata")
        if not isinstance(metadata, dict):
            return {}
        return metadata

    def _is_duplicate_within_split(
        self,
        trace_data: dict[str, Any],
        seen_keys: set[tuple[str, int]],
    ) -> tuple[bool, tuple[str, int] | None]:
        if trace_data.get("mode") != "trace_collection":
            return False, None
        key = self._trace_key(trace_data)
        if key is None:
            return False, None
        return key in seen_keys, key

    def _validate_trace_collection_ordinals(
        self,
        records: list[TraceOrdinalRecord],
        *,
        generators: list[str],
        target: int,
    ) -> None:
        seen: dict[tuple[str, int], TraceOrdinalRecord] = {}
        duplicates: list[tuple[str, int]] = []
        outside: list[tuple[str, int]] = []
        missing_metadata_files: list[str] = []
        for record in records:
            key = record["key"]
            if key is None:
                missing_metadata_files.append(record["file"])
                continue
            generator, ordinal = key
            key = (generator, ordinal)
            if generator not in generators or ordinal < 0 or ordinal >= target:
                outside.append(key)
                continue
            if key in seen:
                duplicates.append(key)
                continue
            seen[key] = record

        missing = [
            (generator, ordinal)
            for generator in generators
            for ordinal in range(target)
            if (generator, ordinal) not in seen
        ]
        if missing or duplicates or outside or missing_metadata_files:
            raise ValueError(
                "Trace collection assembly requires exactly one SFT-usable trace for "
                f"each contiguous item_run_ordinal 0..{target - 1} per generator. "
                f"missing={missing[:10]}, duplicates={duplicates[:10]}, "
                f"outside={outside[:10]}, missing_metadata_files={missing_metadata_files[:10]}"
            )

    def _collection_audit_rejection_reasons(
        self, trace_data: dict[str, Any]
    ) -> list[str]:
        if (
            trace_data.get("mode") != "trace_collection"
            or not self.quality_filter.config.reject_on_holdout_leakage
        ):
            return []
        audit = audit_trace_payload(trace_data)
        if audit["audit_usable"]:
            return []
        classes = audit["hard_rejection_classes"]
        if classes:
            return [f"collection_audit:{class_name}" for class_name in classes]
        return ["collection_audit:not_auditable"]

    def _convert_to_example(
        self, trace_data: dict[str, Any], run_id: int
    ) -> dict[str, Any]:
        messages = []
        token_count = trace_data.get("sft_num_tokens")
        if not isinstance(token_count, int):
            token_count = trace_data.get("total_tokens", 0)
        if not isinstance(token_count, int):
            token_count = 0
        for message in trace_messages(trace_data):
            messages.append(
                {
                    "role": message.get("role", ""),
                    "content": sft_training_content(message),
                    "tool_calls": message.get("tool_calls", []),
                    "tool_call_id": message.get("tool_call_id"),
                    "name": message.get("name"),
                }
            )

        return {
            "messages": messages,
            "scenario": trace_data.get("scenario", "unknown"),
            "num_tokens": token_count,
            "run_id": run_id,
            "success": trace_data.get("success", False),
            "turns": trace_data.get("turns", 0),
            "final_message": trace_data.get("final_message", ""),
            "model": trace_data.get("model", ""),
            "mode": trace_data.get("mode", ""),
            "cost": trace_data.get("cost") or trace_data.get("total_cost", 0.0),
            "prompt_tokens": trace_data.get("prompt_tokens", 0),
            "completion_tokens": trace_data.get("completion_tokens", 0),
            "tools": trace_data.get("tools", []),
            "metadata": json.dumps(trace_data.get("metadata", {})),
        }

    def _record_filtered_reason(self, reason: str) -> None:
        self.stats["filtered"][reason] = self.stats["filtered"].get(reason, 0) + 1

    def _report_pruned_files(self, pruned_files: list[tuple[str, list[str]]]) -> None:
        if not pruned_files:
            return
        console.print(f"\n[bold red]Pruned {len(pruned_files)} traces:[/]")
        for name, reasons in pruned_files:
            console.print(f"  [dim]- {name}: {reasons}[/]")
        console.print()

    def _prune_file(self, path: Path, reasons: list[str]) -> None:
        try:
            os.remove(path)
        except OSError as exc:
            log.warning("Failed to prune %s: %s", path, exc)
