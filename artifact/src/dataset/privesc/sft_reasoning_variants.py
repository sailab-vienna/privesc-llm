import json
import re
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import local
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Protocol, cast

import hydra
from hydra.utils import get_original_cwd
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from src.config import (
    AppConfig,
    plain_config_dict,
    register_with_hydra,
    sft_teacher_model,
)
from src.dataset.privesc.sft_assembly import DatasetAssembler
from src.dataset.privesc.sft_quality import (
    QualityConfig,
    TraceQualityFilter,
    matched_keywords,
)
from src.dataset.privesc.trace_utils import (
    sft_assistant_content,
    trace_messages,
)
from src.paths import (
    assert_no_path_overlap,
    resolve_traces_dir,
    traces_dir as build_traces_dir,
)
from src.utils.langchain import build_chat_openai


NO_REASONING_VARIANT = "no_reasoning"
SHORT_REASONING_VARIANT = "short_reasoning"
LONG_REASONING_VARIANT = "long_reasoning"
REMOVE_ASSISTANT_REASONING_TRANSFORM = "remove_assistant_reasoning"
SHORT_REASONING_TRANSFORM = "rewrite_short_reasoning"
SHORT_REASONING_PROMPT_TEMPLATE = "short_reasoning_rewrite.jinja"
SHORT_REASONING_STRUCTURED_OUTPUT_METHOD = "json_mode"
SHORT_REASONING_REWRITE_MAX_RETRIES = 10
SHORT_REASONING_MIN_SOURCE_CHARS_FOR_LENGTH_CHECK = 100
SHORT_REASONING_ERROR_SNIPPET_CHARS = 500
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
REASONING_FIELD_KEYS = {"reasoning", "reasoning_content"}
REASONING_CONTENT_TYPES = {"reasoning", "reasoning_text"}


class ShortReasoningRewriter(Protocol):
    def rewrite_trace(
        self,
        trace_data: dict[str, Any],
        *,
        source_file: Path,
    ) -> dict[int, str]: ...


class ForbiddenShortReasoningPhraseError(ValueError):
    pass


@dataclass(frozen=True)
class TraceTransformContext:
    source_file: Path
    source_root: Path
    output_root: Path
    variant: str
    source_variant: str
    quality_metrics: dict[str, Any]
    teacher_model: str
    forbidden_short_reasoning_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceTransformResult:
    trace: dict[str, Any]
    llm_calls: int = 0
    assistant_messages_rewritten: int = 0
    reasoning_chars_before: int = 0
    reasoning_chars_after: int = 0


@dataclass(frozen=True)
class TraceTransformJob:
    trace_data: dict[str, Any]
    context: TraceTransformContext


class ShortReasoningItem(BaseModel):
    message_index: int = Field(
        description="Index of the assistant message in the original trace message list."
    )
    short_reasoning: str = Field(
        min_length=1,
        description="Concise reasoning for this exact assistant message.",
    )


class ShortReasoningResponse(BaseModel):
    assistant_reasoning: list[ShortReasoningItem] = Field(
        description="One concise reasoning rewrite for every requested assistant message."
    )


@dataclass
class ReasoningVariantStats:
    total: int = 0
    passed_quality: int = 0
    transformed: int = 0
    reused: int = 0
    regenerated: int = 0
    stale_removed: int = 0
    llm_calls: int = 0
    assistant_messages_rewritten: int = 0
    reasoning_chars_before: int = 0
    reasoning_chars_after: int = 0
    filtered: dict[str, int] = field(default_factory=dict)

    def record_filtered(self, reason: str) -> None:
        self.filtered[reason] = self.filtered.get(reason, 0) + 1

    def record_transform(self, result: TraceTransformResult) -> None:
        self.transformed += 1
        self.regenerated += 1
        self.llm_calls += result.llm_calls
        self.assistant_messages_rewritten += result.assistant_messages_rewritten
        self.reasoning_chars_before += result.reasoning_chars_before
        self.reasoning_chars_after += result.reasoning_chars_after

    def record_reuse(self, result: TraceTransformResult) -> None:
        self.transformed += 1
        self.reused += 1
        self.assistant_messages_rewritten += result.assistant_messages_rewritten
        self.reasoning_chars_before += result.reasoning_chars_before
        self.reasoning_chars_after += result.reasoning_chars_after

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_no_reasoning_trace(
    trace_data: dict[str, Any], *, context: TraceTransformContext
) -> TraceTransformResult:
    transformed = deepcopy(trace_data)
    for message in trace_messages(transformed):
        if message.get("role") == "assistant":
            _remove_assistant_reasoning_message(message)

    _record_provenance(
        transformed,
        source_file=context.source_file,
        source_root=context.source_root,
        output_root=context.output_root,
        variant=context.variant,
        source_variant=context.source_variant,
        quality_metrics=context.quality_metrics,
    )
    _assert_removed_reasoning_invariants(trace_data, transformed)
    return TraceTransformResult(trace=transformed)


class DeepSeekShortReasoningRewriter:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self._local = local()
        self.system_prompt = _short_reasoning_system_prompt()

    def _structured_llm(self) -> Any:
        structured_llm = getattr(self._local, "structured_llm", None)
        if structured_llm is None:
            llm = build_chat_openai(
                api_base=self.api_base,
                api_key=self.api_key,
                model=self.model,
                temperature=0,
                max_retries=0,
                extra_body={
                    "provider": {
                        "only": ["deepseek"],
                        "allow_fallbacks": False,
                    }
                },
            )
            structured_llm = llm.with_structured_output(
                ShortReasoningResponse,
                method=SHORT_REASONING_STRUCTURED_OUTPUT_METHOD,
            )
            self._local.structured_llm = structured_llm
        return structured_llm

    def rewrite_trace(
        self,
        trace_data: dict[str, Any],
        *,
        source_file: Path,
    ) -> dict[int, str]:
        payload, assistant_indices = _short_reasoning_rewrite_payload(trace_data)
        response = self._structured_llm().invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            ]
        )
        return _short_reasoning_response_to_mapping(response, assistant_indices)


def derive_short_reasoning_trace(
    trace_data: dict[str, Any],
    *,
    context: TraceTransformContext,
    rewriter: ShortReasoningRewriter,
) -> TraceTransformResult:
    original_reasoning = _assistant_reasoning_by_index(trace_data)
    rewritten, llm_calls = _rewrite_short_reasoning_with_retries(
        trace_data,
        source_file=context.source_file,
        rewriter=rewriter,
        original_reasoning=original_reasoning,
        forbidden_phrases=context.forbidden_short_reasoning_phrases,
    )

    transformed = deepcopy(trace_data)
    for message_idx, message in enumerate(trace_messages(transformed)):
        if message.get("role") == "assistant":
            _rewrite_short_reasoning_message(message, rewritten[message_idx])

    _record_provenance(
        transformed,
        source_file=context.source_file,
        source_root=context.source_root,
        output_root=context.output_root,
        variant=context.variant,
        source_variant=context.source_variant,
        transform=SHORT_REASONING_TRANSFORM,
        quality_metrics=context.quality_metrics,
        rewrite_model=context.teacher_model,
        rewrite_prompt_template=SHORT_REASONING_PROMPT_TEMPLATE,
        structured_output_method=SHORT_REASONING_STRUCTURED_OUTPUT_METHOD,
    )
    _assert_short_reasoning_invariants(trace_data, transformed)
    return TraceTransformResult(
        trace=transformed,
        llm_calls=llm_calls,
        assistant_messages_rewritten=len(rewritten),
        reasoning_chars_before=sum(len(text) for text in original_reasoning.values()),
        reasoning_chars_after=sum(len(text) for text in rewritten.values()),
    )


def _rewrite_short_reasoning_with_retries(
    trace_data: dict[str, Any],
    *,
    source_file: Path,
    rewriter: ShortReasoningRewriter,
    original_reasoning: dict[int, str],
    forbidden_phrases: tuple[str, ...] = (),
    max_retries: int = SHORT_REASONING_REWRITE_MAX_RETRIES,
) -> tuple[dict[int, str], int]:
    _validate_short_reasoning_sources(original_reasoning)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            rewritten = rewriter.rewrite_trace(trace_data, source_file=source_file)
            _validate_short_reasoning_mapping(
                original_reasoning,
                rewritten,
                source_file=source_file,
                forbidden_phrases=forbidden_phrases,
            )
            return rewritten, attempt + 1
        except ForbiddenShortReasoningPhraseError:
            raise
        except (ValueError, ValidationError) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("short_reasoning rewrite retry loop exited unexpectedly")


def transform_reasoning_variant_split(
    *,
    source_trace_root: Path,
    output_trace_root: Path,
    teacher_model: str,
    source_dir: str,
    quality_filter: TraceQualityFilter,
    variant: str = NO_REASONING_VARIANT,
    source_variant: str = LONG_REASONING_VARIANT,
    exclude_source_dir: str | None = None,
    short_reasoning_rewriter: ShortReasoningRewriter | None = None,
    project_root: Path | None = None,
    workers: int = 1,
    incremental: bool = False,
) -> ReasoningVariantStats:
    if variant not in {NO_REASONING_VARIANT, SHORT_REASONING_VARIANT}:
        raise ValueError(f"Unsupported reasoning variant: {variant}")
    if variant == SHORT_REASONING_VARIANT and short_reasoning_rewriter is None:
        raise ValueError("short_reasoning requires a short_reasoning_rewriter")
    if workers < 1:
        raise ValueError("reasoning_variant_derivation.workers must be >= 1")

    base_dir = project_root or Path.cwd()
    source_root = _resolve_root(source_trace_root, base_dir)
    output_root = _resolve_root(output_trace_root, base_dir)
    _reject_overlapping_roots(source_root, output_root)
    traces_dir = resolve_traces_dir(
        base_dir, teacher_model, source_dir, trace_root=str(source_root)
    )
    if traces_dir is None:
        raise FileNotFoundError(
            f"Missing source traces for {teacher_model} split {source_dir}: {source_root}"
        )

    output_dir = Path(build_traces_dir(str(output_root / source_dir), teacher_model))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    assembler = DatasetAssembler(
        cfg=cast(AppConfig, SimpleNamespace()),
        quality_filter=quality_filter,
    )
    exclude_keys = assembler._exclude_keys(
        base_dir,
        teacher_model,
        exclude_source_dir=exclude_source_dir,
        trace_root=str(source_root),
    )
    forbidden_short_reasoning_phrases = _short_reasoning_forbidden_phrases(
        quality_filter
    )
    if incremental:
        if variant != SHORT_REASONING_VARIANT:
            raise ValueError("incremental derivation is only supported for short_reasoning")
        output_dir.mkdir(parents=True, exist_ok=True)
        return _transform_short_reasoning_incremental(
            traces_dir=traces_dir,
            output_dir=output_dir,
            source_root=source_root,
            output_root=output_root,
            source_dir=source_dir,
            source_variant=source_variant,
            teacher_model=teacher_model,
            assembler=assembler,
            quality_filter=quality_filter,
            exclude_keys=exclude_keys,
            forbidden_short_reasoning_phrases=forbidden_short_reasoning_phrases,
            short_reasoning_rewriter=short_reasoning_rewriter,
            workers=workers,
        )

    temp_output_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    replace_output = False

    seen_keys: set[tuple[str, int]] = set()
    stats = ReasoningVariantStats()

    try:
        with tqdm(desc=f"{variant} {source_dir}", unit="trace") as progress:
            if workers == 1:
                for job in _iter_transform_jobs(
                    traces_dir=traces_dir,
                    assembler=assembler,
                    exclude_keys=exclude_keys,
                    seen_keys=seen_keys,
                    stats=stats,
                    source_root=source_root,
                    output_root=output_root,
                    variant=variant,
                    source_variant=source_variant,
                    teacher_model=teacher_model,
                    forbidden_short_reasoning_phrases=forbidden_short_reasoning_phrases,
                ):
                    result = _derive_trace_variant_job(
                        job,
                        short_reasoning_rewriter=short_reasoning_rewriter,
                    )
                    _write_trace_transform_result(temp_output_dir, job, result)
                    stats.record_transform(result)
                    progress.update(1)
            else:
                pending: dict[Future[TraceTransformResult], TraceTransformJob] = {}

                def record_completed(done: set[Future[TraceTransformResult]]) -> None:
                    for future in done:
                        job = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception:
                            for pending_future in pending:
                                pending_future.cancel()
                            raise
                        _write_trace_transform_result(temp_output_dir, job, result)
                        stats.record_transform(result)
                        progress.update(1)

                executor = ThreadPoolExecutor(max_workers=workers)
                try:
                    for job in _iter_transform_jobs(
                        traces_dir=traces_dir,
                        assembler=assembler,
                        exclude_keys=exclude_keys,
                        seen_keys=seen_keys,
                        stats=stats,
                        source_root=source_root,
                        output_root=output_root,
                        variant=variant,
                        source_variant=source_variant,
                        teacher_model=teacher_model,
                        forbidden_short_reasoning_phrases=forbidden_short_reasoning_phrases,
                    ):
                        future = executor.submit(
                            _derive_trace_variant_job,
                            job,
                            short_reasoning_rewriter=short_reasoning_rewriter,
                        )
                        pending[future] = job
                        if len(pending) >= workers:
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            record_completed(done)

                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        record_completed(done)
                except Exception:
                    for pending_future in pending:
                        pending_future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_output_dir.rename(output_dir)
        replace_output = True
    finally:
        if not replace_output:
            shutil.rmtree(temp_output_dir, ignore_errors=True)

    stats_dir = output_root / source_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / f"{variant}_transform.json").write_text(
        json.dumps(stats.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


def _derive_trace_variant(
    trace_data: dict[str, Any],
    *,
    context: TraceTransformContext,
    short_reasoning_rewriter: ShortReasoningRewriter | None,
) -> TraceTransformResult:
    if context.variant == NO_REASONING_VARIANT:
        return derive_no_reasoning_trace(trace_data, context=context)
    if context.variant == SHORT_REASONING_VARIANT:
        if short_reasoning_rewriter is None:
            raise ValueError("short_reasoning requires a short_reasoning_rewriter")
        return derive_short_reasoning_trace(
            trace_data,
            context=context,
            rewriter=short_reasoning_rewriter,
        )
    raise ValueError(f"Unsupported reasoning variant: {context.variant}")


def _iter_transform_jobs(
    *,
    traces_dir: Path,
    assembler: DatasetAssembler,
    exclude_keys: set[tuple[str, int]],
    seen_keys: set[tuple[str, int]],
    stats: ReasoningVariantStats,
    source_root: Path,
    output_root: Path,
    variant: str,
    source_variant: str,
    teacher_model: str,
    forbidden_short_reasoning_phrases: tuple[str, ...],
) -> Iterable[TraceTransformJob]:
    for trace_file in sorted(traces_dir.glob("*.json")):
        stats.total += 1
        raw_data = assembler._read_trace(trace_file)
        raw_key = assembler._trace_key(raw_data)
        if raw_key is not None and raw_key in exclude_keys:
            stats.record_filtered("seed_collision_with_excluded_split")
            continue

        passed, metrics, key, _ = assembler.prepare_passing_trace(
            raw_data,
            seen_keys=seen_keys,
        )
        if not passed:
            for reason in metrics.get("reasons", []):
                stats.record_filtered(str(reason))
            continue

        stats.passed_quality += 1
        if key is not None:
            seen_keys.add(key)

        yield TraceTransformJob(
            trace_data=raw_data,
            context=TraceTransformContext(
                source_file=trace_file,
                source_root=source_root,
                output_root=output_root,
                variant=variant,
                source_variant=source_variant,
                quality_metrics=metrics,
                teacher_model=teacher_model,
                forbidden_short_reasoning_phrases=forbidden_short_reasoning_phrases,
            ),
        )


def _derive_trace_variant_job(
    job: TraceTransformJob,
    *,
    short_reasoning_rewriter: ShortReasoningRewriter | None,
) -> TraceTransformResult:
    return _derive_trace_variant(
        job.trace_data,
        context=job.context,
        short_reasoning_rewriter=short_reasoning_rewriter,
    )


def _transform_short_reasoning_incremental(
    *,
    traces_dir: Path,
    output_dir: Path,
    source_root: Path,
    output_root: Path,
    source_dir: str,
    source_variant: str,
    teacher_model: str,
    assembler: DatasetAssembler,
    quality_filter: TraceQualityFilter,
    exclude_keys: set[tuple[str, int]],
    forbidden_short_reasoning_phrases: tuple[str, ...],
    short_reasoning_rewriter: ShortReasoningRewriter | None,
    workers: int,
) -> ReasoningVariantStats:
    if short_reasoning_rewriter is None:
        raise ValueError("short_reasoning requires a short_reasoning_rewriter")

    seen_keys: set[tuple[str, int]] = set()
    expected_names: set[str] = set()
    stats = ReasoningVariantStats()

    with tqdm(desc=f"short_reasoning {source_dir}", unit="trace") as progress:
        if workers == 1:
            for job in _iter_transform_jobs(
                traces_dir=traces_dir,
                assembler=assembler,
                exclude_keys=exclude_keys,
                seen_keys=seen_keys,
                stats=stats,
                source_root=source_root,
                output_root=output_root,
                variant=SHORT_REASONING_VARIANT,
                source_variant=source_variant,
                teacher_model=teacher_model,
                forbidden_short_reasoning_phrases=forbidden_short_reasoning_phrases,
            ):
                expected_names.add(job.context.source_file.name)
                existing = _reuse_existing_short_reasoning_result(
                    output_dir / job.context.source_file.name,
                    job=job,
                    quality_filter=quality_filter,
                )
                if existing is not None:
                    stats.record_reuse(existing)
                    progress.update(1)
                    continue
                result = _derive_trace_variant_job(
                    job,
                    short_reasoning_rewriter=short_reasoning_rewriter,
                )
                _write_trace_transform_result(output_dir, job, result)
                stats.record_transform(result)
                progress.update(1)
        else:
            pending: dict[Future[TraceTransformResult], TraceTransformJob] = {}

            def record_completed(done: set[Future[TraceTransformResult]]) -> None:
                for future in done:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        for pending_future in pending:
                            pending_future.cancel()
                        raise
                    _write_trace_transform_result(output_dir, job, result)
                    stats.record_transform(result)
                    progress.update(1)

            executor = ThreadPoolExecutor(max_workers=workers)
            try:
                for job in _iter_transform_jobs(
                    traces_dir=traces_dir,
                    assembler=assembler,
                    exclude_keys=exclude_keys,
                    seen_keys=seen_keys,
                    stats=stats,
                    source_root=source_root,
                    output_root=output_root,
                    variant=SHORT_REASONING_VARIANT,
                    source_variant=source_variant,
                    teacher_model=teacher_model,
                    forbidden_short_reasoning_phrases=forbidden_short_reasoning_phrases,
                ):
                    expected_names.add(job.context.source_file.name)
                    existing = _reuse_existing_short_reasoning_result(
                        output_dir / job.context.source_file.name,
                        job=job,
                        quality_filter=quality_filter,
                    )
                    if existing is not None:
                        stats.record_reuse(existing)
                        progress.update(1)
                        continue
                    future = executor.submit(
                        _derive_trace_variant_job,
                        job,
                        short_reasoning_rewriter=short_reasoning_rewriter,
                    )
                    pending[future] = job
                    if len(pending) >= workers:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        record_completed(done)

                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    record_completed(done)
            except Exception:
                for pending_future in pending:
                    pending_future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

    for stale_file in output_dir.glob("*.json"):
        if stale_file.name not in expected_names:
            stale_file.unlink()
            stats.stale_removed += 1

    stats_dir = output_root / source_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "short_reasoning_transform.json").write_text(
        json.dumps(stats.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


def _reuse_existing_short_reasoning_result(
    output_path: Path,
    *,
    job: TraceTransformJob,
    quality_filter: TraceQualityFilter,
) -> TraceTransformResult | None:
    if not output_path.exists():
        return None
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        metadata = existing.get("metadata")
        provenance = (
            metadata.get("sft_reasoning_variant") if isinstance(metadata, dict) else None
        )
        if not isinstance(provenance, dict):
            return None
        if provenance.get("variant") != SHORT_REASONING_VARIANT:
            return None
        if provenance.get("source_variant") != job.context.source_variant:
            return None
        expected_source_path = job.context.source_file.relative_to(
            job.context.source_root
        ).as_posix()
        if provenance.get("source_trace_path") != expected_source_path:
            return None

        original_reasoning = _assistant_reasoning_by_index(job.trace_data)
        rewritten = _assistant_reasoning_by_index(existing)
        _validate_short_reasoning_mapping(
            original_reasoning,
            rewritten,
            source_file=job.context.source_file,
            forbidden_phrases=job.context.forbidden_short_reasoning_phrases,
        )
        _assert_short_reasoning_invariants(job.trace_data, existing)
        passed, _ = quality_filter.check_trace(existing)
        if not passed:
            return None
        return TraceTransformResult(
            trace=existing,
            assistant_messages_rewritten=len(rewritten),
            reasoning_chars_before=sum(len(text) for text in original_reasoning.values()),
            reasoning_chars_after=sum(len(text) for text in rewritten.values()),
        )
    except (OSError, ValueError, ValidationError, json.JSONDecodeError):
        return None


def _write_trace_transform_result(
    output_dir: Path,
    job: TraceTransformJob,
    result: TraceTransformResult,
) -> None:
    output_path = output_dir / job.context.source_file.name
    output_path.write_text(
        json.dumps(result.trace, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def derive_reasoning_variant_from_config(cfg: AppConfig) -> ReasoningVariantStats:
    derivation = cfg.reasoning_variant_derivation
    variant = str(derivation.variant)
    if cfg.datasets.sft.reasoning_variant != variant:
        raise ValueError(
            "datasets.sft.reasoning_variant must match "
            f"reasoning_variant_derivation.variant: {cfg.datasets.sft.reasoning_variant} != {variant}"
        )

    source_dir = cfg.datasets.sft.source_dir
    if not source_dir:
        raise ValueError("datasets.sft.source_dir must be configured")

    teacher_model = sft_teacher_model(cfg)
    project_root = Path(get_original_cwd())
    source_trace_root = Path(
        derivation.source_trace_root
        or _variant_trace_root(cfg, str(derivation.source_variant))
    )
    output_trace_root = Path(
        derivation.output_trace_root
        or cfg.datasets.sft.trace_root
        or _variant_trace_root(cfg, variant)
    )
    quality_filter = TraceQualityFilter(_quality_config_from_hydra(cfg))
    short_reasoning_rewriter = None
    if variant == SHORT_REASONING_VARIANT:
        if cfg.agent.model != teacher_model:
            raise ValueError(
                f"agent.model must match datasets.sft.teacher_model for short_reasoning: {cfg.agent.model} != {teacher_model}"
            )
        short_reasoning_rewriter = DeepSeekShortReasoningRewriter(
            api_base=cfg.agent.api_base,
            api_key=cfg.agent.api_key,
            model=teacher_model,
        )

    return transform_reasoning_variant_split(
        source_trace_root=source_trace_root,
        output_trace_root=output_trace_root,
        teacher_model=teacher_model,
        source_dir=source_dir,
        exclude_source_dir=cfg.datasets.sft.exclude_source_dir,
        variant=variant,
        source_variant=str(derivation.source_variant),
        quality_filter=quality_filter,
        short_reasoning_rewriter=short_reasoning_rewriter,
        project_root=project_root,
        workers=int(derivation.workers),
        incremental=bool(derivation.incremental),
    )


def _variant_trace_root(cfg: AppConfig, reasoning_variant: str) -> str:
    return "/".join(
        (
            "outputs/traces/trace_collection",
            cfg.datasets.sft.profile,
            cfg.datasets.sft.regime,
            cfg.datasets.sft.teacher,
            reasoning_variant,
        )
    )


def _quality_config_from_hydra(cfg: AppConfig) -> QualityConfig:
    payload = plain_config_dict(
        cfg.datasets.sft.quality,
        error_label="datasets.sft.quality",
    )
    return QualityConfig(**payload)


def _remove_assistant_reasoning_message(message: dict[str, Any]) -> None:
    for key in REASONING_FIELD_KEYS:
        message.pop(key, None)

    additional = message.get("additional_kwargs")
    if isinstance(additional, dict):
        for key in REASONING_FIELD_KEYS:
            additional.pop(key, None)

    message["content"] = ""


def _record_provenance(
    trace_data: dict[str, Any],
    *,
    source_file: Path,
    source_root: Path,
    output_root: Path,
    variant: str,
    source_variant: str,
    quality_metrics: dict[str, Any],
    transform: str = REMOVE_ASSISTANT_REASONING_TRANSFORM,
    rewrite_model: str | None = None,
    rewrite_prompt_template: str | None = None,
    structured_output_method: str | None = None,
) -> None:
    metadata = trace_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        trace_data["metadata"] = metadata
    metadata["sft_reasoning_variant"] = {
        "variant": variant,
        "source_variant": source_variant,
        "transform": transform,
        "source_trace_root": source_root.as_posix(),
        "source_trace_path": source_file.relative_to(source_root).as_posix(),
        "output_trace_root": output_root.as_posix(),
        "quality_metrics": quality_metrics,
    }
    if rewrite_model is not None:
        metadata["sft_reasoning_variant"]["rewrite_model"] = rewrite_model
    if rewrite_prompt_template is not None:
        metadata["sft_reasoning_variant"]["rewrite_prompt_template"] = (
            rewrite_prompt_template
        )
    if structured_output_method is not None:
        metadata["sft_reasoning_variant"]["structured_output_method"] = (
            structured_output_method
        )


def _short_reasoning_system_prompt() -> str:
    env = Environment(loader=FileSystemLoader(PROMPTS_DIR))
    return env.get_template(SHORT_REASONING_PROMPT_TEMPLATE).render()


def _short_reasoning_rewrite_payload(
    trace_data: dict[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    items = [
        _assistant_rewrite_item(message, idx)
        for idx, message in enumerate(trace_messages(trace_data))
        if message.get("role") == "assistant"
    ]
    if not items:
        raise ValueError("short_reasoning requires at least one assistant message")

    return {"assistant_rewrites": items}, [item["message_index"] for item in items]


def _assistant_rewrite_item(
    message: dict[str, Any], assistant_idx: int
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "message_index": assistant_idx,
        "source_reasoning": _assistant_source_reasoning(message),
    }
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        item["tool_calls"] = deepcopy(tool_calls)
    return item


def _assistant_source_reasoning(message: dict[str, Any]) -> str:
    reasoning = sft_assistant_content(message.get("content", "")).strip()
    for key in REASONING_FIELD_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            reasoning = _join_unique_text(reasoning, value)
    additional = message.get("additional_kwargs")
    if isinstance(additional, dict):
        for key in REASONING_FIELD_KEYS:
            value = additional.get(key)
            if isinstance(value, str) and value.strip():
                reasoning = _join_unique_text(reasoning, value)
    return reasoning


def _assistant_reasoning_by_index(trace_data: dict[str, Any]) -> dict[int, str]:
    assistant_reasoning: dict[int, str] = {}
    for idx, message in enumerate(trace_messages(trace_data)):
        if message.get("role") != "assistant":
            continue
        assistant_reasoning[idx] = _assistant_source_reasoning(message)
    return assistant_reasoning


def _join_unique_text(current: str, extra: str) -> str:
    extra = extra.strip()
    if not extra:
        return current
    if not current:
        return extra
    if extra in current:
        return current
    return f"{current}\n\n{extra}"


def _rewrite_short_reasoning_message(
    message: dict[str, Any], short_reasoning: str
) -> None:
    message["content"] = short_reasoning
    for key in REASONING_FIELD_KEYS:
        if key in message:
            message[key] = short_reasoning

    additional = message.get("additional_kwargs")
    if isinstance(additional, dict):
        for key in REASONING_FIELD_KEYS:
            if key in additional:
                additional[key] = short_reasoning


def _short_reasoning_response_to_mapping(
    response: ShortReasoningResponse | dict[str, Any],
    assistant_indices: list[int],
) -> dict[int, str]:
    parsed = (
        response
        if isinstance(response, ShortReasoningResponse)
        else ShortReasoningResponse.model_validate(response)
    )
    result = {
        item.message_index: item.short_reasoning for item in parsed.assistant_reasoning
    }
    if len(result) != len(parsed.assistant_reasoning):
        raise ValueError("short_reasoning response contains duplicate message indices")
    if set(result) != set(assistant_indices):
        raise ValueError("short_reasoning response assistant indices mismatch")
    return result


def _validate_short_reasoning_mapping(
    original_reasoning: dict[int, str],
    rewritten: dict[int, str],
    *,
    source_file: Path,
    forbidden_phrases: tuple[str, ...] = (),
) -> None:
    if set(original_reasoning) != set(rewritten):
        raise ValueError("short_reasoning response assistant indices mismatch")
    for idx, text in rewritten.items():
        source = original_reasoning[idx].strip()
        short = text.strip()
        if not text.strip():
            raise ValueError(f"short_reasoning_empty[{idx}]")
        if THINK_TAG_RE.search(text):
            raise ValueError(f"short_reasoning_contains_think_tag[{idx}]")
        leakage = _matched_forbidden_phrases(short, forbidden_phrases)
        if leakage:
            raise ForbiddenShortReasoningPhraseError(
                f"short_reasoning_forbidden_phrase[{idx}] {source_file} "
                f"({', '.join(leakage)})"
            )
        if not source:
            raise ValueError(f"short_reasoning_missing_source_reasoning[{idx}]")
        if (
            len(source) >= SHORT_REASONING_MIN_SOURCE_CHARS_FOR_LENGTH_CHECK
            and len(short) >= len(source)
        ):
            raise ValueError(
                f"short_reasoning_not_shorter[{idx}] {source_file} "
                f"({len(short)} >= {len(source)}); "
                f"source={_error_snippet(source)!r}; "
                f"rewritten={_error_snippet(short)!r}"
            )


def _error_snippet(text: str) -> str:
    if len(text) <= SHORT_REASONING_ERROR_SNIPPET_CHARS:
        return text
    return text[:SHORT_REASONING_ERROR_SNIPPET_CHARS] + "...<truncated>"


def _validate_short_reasoning_sources(original_reasoning: dict[int, str]) -> None:
    for idx, source in original_reasoning.items():
        if not source.strip():
            raise ValueError(f"short_reasoning_missing_source_reasoning[{idx}]")


def _short_reasoning_forbidden_phrases(
    quality_filter: TraceQualityFilter,
) -> tuple[str, ...]:
    if not quality_filter.config.reject_on_secret_solution_leakage:
        return ()
    return tuple(str(kw) for kw in quality_filter.config.secret_solution_leakage_keywords)


def _matched_forbidden_phrases(text: str, phrases: tuple[str, ...]) -> list[str]:
    return matched_keywords(text, phrases)


def _assert_removed_reasoning_invariants(
    original: dict[str, Any], transformed: dict[str, Any]
) -> None:
    _assert_trace_invariants(
        original,
        transformed,
        assistant_invariant=_assert_reasoning_removed_assistant_invariants,
    )


def _assert_short_reasoning_invariants(
    original: dict[str, Any], transformed: dict[str, Any]
) -> None:
    _assert_trace_invariants(
        original,
        transformed,
        assistant_invariant=_assert_short_assistant_invariants,
    )


def _assert_trace_invariants(
    original: dict[str, Any],
    transformed: dict[str, Any],
    *,
    assistant_invariant: Callable[[int, dict[str, Any], dict[str, Any]], None],
) -> None:
    original_messages = trace_messages(original)
    transformed_messages = trace_messages(transformed)
    if len(original_messages) != len(transformed_messages):
        raise ValueError("message_count_changed")

    for idx, (before, after) in enumerate(zip(original_messages, transformed_messages)):
        if before.get("role") != after.get("role"):
            raise ValueError(f"message_role_changed[{idx}]")
        if before.get("role") != "assistant":
            if before != after:
                raise ValueError(f"non_assistant_message_changed[{idx}]")
            continue
        assistant_invariant(idx, before, after)

    for key in ("success", "turns", "scenario", "mode"):
        if original.get(key) != transformed.get(key):
            raise ValueError(f"trace_field_changed:{key}")

    _assert_metadata_preserved(original.get("metadata"), transformed.get("metadata"))


def _assert_short_assistant_invariants(
    idx: int, before: dict[str, Any], after: dict[str, Any]
) -> None:
    for key, value in before.items():
        if key in {"content", "additional_kwargs", *REASONING_FIELD_KEYS}:
            continue
        if after.get(key) != value:
            raise ValueError(f"assistant_field_changed[{idx}]:{key}")

    rewritten = after.get("content")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError(f"assistant_short_reasoning_missing[{idx}]")
    if THINK_TAG_RE.search(rewritten):
        raise ValueError(f"assistant_short_reasoning_contains_think_tag[{idx}]")

    before_additional = before.get("additional_kwargs")
    after_additional = after.get("additional_kwargs")
    if isinstance(before_additional, dict):
        expected = dict(before_additional)
        for key in REASONING_FIELD_KEYS:
            if key in expected:
                expected[key] = rewritten
        actual = after_additional if isinstance(after_additional, dict) else {}
        if actual != expected:
            raise ValueError(f"assistant_additional_kwargs_changed[{idx}]")

    for key in REASONING_FIELD_KEYS:
        if key in before and after.get(key) != rewritten:
            raise ValueError(f"assistant_reasoning_not_rewritten[{idx}]:{key}")


def _assert_reasoning_removed_assistant_invariants(
    idx: int, before: dict[str, Any], after: dict[str, Any]
) -> None:
    for key, value in before.items():
        if key in {"content", "additional_kwargs", *REASONING_FIELD_KEYS}:
            continue
        if after.get(key) != value:
            raise ValueError(f"assistant_field_changed[{idx}]:{key}")

    if after.get("content") != "":
        raise ValueError(f"assistant_content_remaining[{idx}]")
    if _contains_reasoning_payload(after):
        raise ValueError(f"assistant_reasoning_remaining[{idx}]")

    before_additional = before.get("additional_kwargs")
    after_additional = after.get("additional_kwargs")
    if isinstance(before_additional, dict):
        expected = {
            key: value
            for key, value in before_additional.items()
            if key not in REASONING_FIELD_KEYS
        }
        actual = after_additional if isinstance(after_additional, dict) else {}
        if actual != expected:
            raise ValueError(f"assistant_additional_kwargs_changed[{idx}]")


def _contains_reasoning_payload(value: Any) -> bool:
    if isinstance(value, str):
        return bool(THINK_TAG_RE.search(value))
    if isinstance(value, list):
        return any(_contains_reasoning_payload(item) for item in value)
    if isinstance(value, dict):
        if REASONING_FIELD_KEYS.intersection(value):
            return True
        if value.get("type") in REASONING_CONTENT_TYPES:
            return True
        return any(_contains_reasoning_payload(item) for item in value.values())
    return False


def _assert_metadata_preserved(original: Any, transformed: Any) -> None:
    if not isinstance(original, dict):
        if original != transformed:
            raise ValueError("metadata_changed")
        return
    if not isinstance(transformed, dict):
        raise ValueError("metadata_changed")
    without_provenance = dict(transformed)
    without_provenance.pop("sft_reasoning_variant", None)
    if original != without_provenance:
        raise ValueError("metadata_changed")


def _resolve_root(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _reject_overlapping_roots(source_root: Path, output_root: Path) -> None:
    try:
        assert_no_path_overlap(source_root, output_root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to derive reasoning variants with overlapping roots: {source_root.resolve()} -> {output_root.resolve()}"
        ) from exc


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: AppConfig) -> None:
    stats = derive_reasoning_variant_from_config(cfg)
    print(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    register_with_hydra()
    main()
