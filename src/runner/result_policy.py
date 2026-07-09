from __future__ import annotations

import subprocess
from typing import Any

import asyncssh

from src.gym.agent import SessionTraceError


_EXIT_STATUS_125_TEXT = "process exited with non-zero exit status 125"


def _root_error(error: BaseException) -> BaseException:
    if isinstance(error, SessionTraceError):
        return _root_error(error.original_error)
    return error


def is_exit_status_125_infra_error(error: BaseException | str | None) -> bool:
    if error is None:
        return False
    if isinstance(error, str):
        return _EXIT_STATUS_125_TEXT in error.lower()

    root = _root_error(error)
    if isinstance(root, asyncssh.ProcessError):
        return int(getattr(root, "exit_status", -1)) == 125
    if isinstance(root, subprocess.CalledProcessError):
        return int(getattr(root, "returncode", -1)) == 125
    return _EXIT_STATUS_125_TEXT in str(root).lower()


def is_benchmark_eligible_payload(payload: dict[str, Any]) -> bool:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False

    if metadata.get("benchmark_eligible") is not True:
        return False

    if metadata.get("trace_incomplete") is True:
        return False

    if payload.get("status") != "completed":
        return False

    if payload.get("error"):
        return False

    success = payload.get("success")
    if success is True:
        return True

    if success is not False:
        return False

    prompt_vars = metadata.get("prompt_vars")
    if not isinstance(prompt_vars, dict):
        return False

    max_turns = prompt_vars.get("max_turns")
    if not isinstance(max_turns, int):
        return False

    turns = payload.get("turns")
    return isinstance(turns, int) and turns >= max_turns


def is_trace_collection_resume_payload(payload: dict[str, Any]) -> bool:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False

    if payload.get("mode") != "trace_collection":
        return False

    if metadata.get("trace_incomplete") is True:
        return False

    if payload.get("status") != "completed":
        return False

    if payload.get("error"):
        return False

    return (
        is_benchmark_eligible_payload(payload)
        or metadata.get("pre_repair_rejected") is True
    )


def counts_toward_trace_collection_target(payload: dict[str, Any]) -> bool:
    return is_benchmark_eligible_payload(payload) and payload.get("success") is True
