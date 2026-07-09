from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console

from src.config import SSHConfig, resolve_default_host_path
from src.gym.scenario import PrivEscScenario
from src.rl.benchmark import load_benchmark_scenarios
from src.scenarios.static import StaticScenarioSource

console = Console()

RUN_ID = 1
DEFAULT_OUTPUT_ROOT = "outputs/evals/paper_baselines/chainreactor/static"
SYSTEM_PLANNER_BIN = "powerlifted"
DEFAULT_PLANNER_TIME_LIMIT = 30
FIELDNAMES = [
    "scenario",
    "run_id",
    "status",
    "extract_succeeded",
    "root_problem_exists",
    "plan_found",
    "plan_length",
    "goal_found_time_sec",
    "planner_total_time_sec",
    "planner_time_limit_sec",
    "planner_timed_out",
    "planner_available",
    "result_json",
]


@dataclass(slots=True)
class LoggedCommandResult:
    command: str
    return_code: int
    duration_sec: float
    log_path: str
    output: str


@dataclass(slots=True)
class PlanMetrics:
    action_count: int
    cost: int | None


@dataclass(slots=True)
class ChainReactorRunResult:
    scenario: str
    run_id: int
    status: str
    output_dir: str
    planner_bin: str
    planner_available: bool
    planner_time_limit_sec: int | None = None
    backend: str = "local_docker"
    container_user: str = ""
    extract_succeeded: bool = False
    root_problem_exists: bool = False
    plan_found: bool = False
    plan_length: int | None = None
    plan_cost: int | None = None
    executed_success: bool | None = None
    extract_return_code: int | None = None
    extract_duration_sec: float | None = None
    solve_return_code: int | None = None
    solve_duration_sec: float | None = None
    goal_found_time_sec: float | None = None
    planner_total_time_sec: float | None = None
    planner_timed_out: bool | None = None
    extract_command: str | None = None
    solve_command: str | None = None
    extract_log: str | None = None
    solve_log: str | None = None
    error_log: str | None = None
    root_problem_path: str | None = None
    plan_path: str | None = None
    error: str | None = None

    def csv_row(self, *, output_root: Path) -> dict[str, str]:
        result_json = Path(self.output_dir) / f"run_{self.run_id}.json"
        return {
            "scenario": self.scenario,
            "run_id": str(self.run_id),
            "status": self.status,
            "extract_succeeded": "1" if self.extract_succeeded else "0",
            "root_problem_exists": "1" if self.root_problem_exists else "0",
            "plan_found": "1" if self.plan_found else "0",
            "plan_length": "" if self.plan_length is None else str(self.plan_length),
            "goal_found_time_sec": ""
            if self.goal_found_time_sec is None
            else f"{self.goal_found_time_sec:.5f}",
            "planner_total_time_sec": ""
            if self.planner_total_time_sec is None
            else f"{self.planner_total_time_sec:.5f}",
            "planner_time_limit_sec": ""
            if self.planner_time_limit_sec is None
            else str(self.planner_time_limit_sec),
            "planner_timed_out": ""
            if self.planner_timed_out is None
            else ("1" if self.planner_timed_out else "0"),
            "planner_available": "1" if self.planner_available else "0",
            "result_json": str(result_json.relative_to(output_root)),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _chainreactor_root() -> Path:
    return _repo_root() / "external" / "chainreactor"


def _facts_extractor_path() -> Path:
    return _chainreactor_root() / "facts_extractor.py"


def _domain_path() -> Path:
    return _chainreactor_root() / "domain.pddl"


def _default_planner_bin() -> str:
    env_planner = os.environ.get("CHAINREACTOR_PLANNER")
    if env_planner:
        return env_planner
    local_planner = _repo_root() / "external" / "powerlifted" / "bin" / "powerlifted"
    if local_planner.is_file():
        return str(local_planner)
    return SYSTEM_PLANNER_BIN


def _planner_available(planner_bin: str) -> bool:
    path = Path(planner_bin)
    return path.is_file() or shutil.which(planner_bin) is not None


def _ssh_from_env() -> SSHConfig:
    return SSHConfig(
        user=os.environ.get("PRIVESC_USER", "root"),
        key_path=os.environ.get("PRIVESC_KEY", str(Path.home() / ".ssh/id_rsa")),
        servers=os.environ.get("PRIVESC_SSH_SERVERS", ""),
        host_path=os.environ.get(
            "PRIVESC_HOST_PATH",
            resolve_default_host_path(str(Path.home())),
        ),
    )


def _append_setup_script(existing: str | None, extra: str) -> str:
    if not existing:
        return extra
    return f"{existing.rstrip()}\n\n{extra}"


def build_authorized_keys_setup_script(container_user: str, public_key: str) -> str:
    user = shlex.quote(container_user)
    key = shlex.quote(public_key.strip())
    home_dir = shlex.quote(f"/home/{container_user}")
    return "\n".join(
        [
            f"home_dir=$(getent passwd {user} | cut -d: -f6)",
            f'[ -n "$home_dir" ] || home_dir={home_dir}',
            f'install -d -m 700 -o {user} -g {user} "$home_dir/.ssh"',
            'touch "$home_dir/.ssh/authorized_keys"',
            (
                f'grep -qxF {key} "$home_dir/.ssh/authorized_keys" || '
                f"printf '%s\\n' {key} >> \"$home_dir/.ssh/authorized_keys\""
            ),
            f'chown {user}:{user} "$home_dir/.ssh/authorized_keys"',
            'chmod 600 "$home_dir/.ssh/authorized_keys"',
        ]
    )


def build_chainreactor_prereq_setup_script() -> str:
    return "\n".join(
        [
            "export DEBIAN_FRONTEND=noninteractive",
            "if ! command -v file >/dev/null 2>&1; then",
            "  apt-get update",
            "  apt-get install -y --no-install-recommends file",
            "  rm -rf /var/lib/apt/lists/*",
            "fi",
        ]
    )


def _generate_temp_ssh_keypair(private_key_path: Path) -> str:
    command = [
        "ssh-keygen",
        "-q",
        "-t",
        "ed25519",
        "-N",
        "",
        "-f",
        str(private_key_path),
    ]
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh-keygen failed: {proc.stdout.strip()}")
    return private_key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()


def _run_logged_command(
    argv: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int | None = None,
) -> LoggedCommandResult:
    command = shlex.join(argv)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
        duration = time.perf_counter() - started
        output = proc.stdout or ""
        return_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output = f"[TIMEOUT after {timeout_seconds}s]\n\n{output}"
        return_code = 124
    log_path.write_text(
        f"cmd: {command}\nreturn_code: {return_code}\n\n{output}",
        encoding="utf-8",
    )
    return LoggedCommandResult(
        command=command,
        return_code=return_code,
        duration_sec=duration,
        log_path=str(log_path),
        output=output,
    )


def parse_powerlifted_timings(output: str) -> tuple[float | None, float | None, bool]:
    goal_found_time = None
    planner_total_time = None
    timed_out = "timed out after" in output.lower()

    goal_match = re.search(
        r"^Goal found at:\s*([0-9]+(?:\.[0-9]+)?)$", output, re.MULTILINE
    )
    if goal_match:
        goal_found_time = float(goal_match.group(1))

    total_match = re.search(
        r"^Total time:\s*([0-9]+(?:\.[0-9]+)?)$", output, re.MULTILINE
    )
    if total_match:
        planner_total_time = float(total_match.group(1))

    return goal_found_time, planner_total_time, timed_out


def build_planner_command(
    planner_bin: str,
    *,
    domain_path: Path,
    problem_path: Path,
    time_limit_sec: int,
) -> list[str]:
    return [
        planner_bin,
        "--iteration",
        "alt-bfws1,rff,yannakakis,476",
        "--unit-cost",
        "--preprocess-task",
        "--only-effects-novelty-check",
        "--time-limit",
        str(time_limit_sec),
        "-d",
        str(domain_path.resolve()),
        "-i",
        str(problem_path.resolve()),
    ]


def parse_plan_metrics(plan_path: Path) -> PlanMetrics:
    action_count = 0
    cost: int | None = None
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(";"):
            if stripped.lower().startswith("; cost ="):
                raw_cost = stripped.partition("=")[2].strip()
                if raw_cost.isdigit():
                    cost = int(raw_cost)
            continue
        action_count += 1
    return PlanMetrics(action_count=action_count, cost=cost)


def _append_csv_row(csv_path: Path, row: dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _first_plan_path(run_dir: Path) -> Path | None:
    plan_paths = sorted(path for path in run_dir.glob("plan*") if path.is_file())
    return plan_paths[0] if plan_paths else None


def _relative_to_output_root(path: Path | None, output_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(output_root))
    except ValueError:
        return str(path)


async def _evaluate_scenario(
    scenario_id: str,
    *,
    output_root: Path,
    planner_bin: str,
    planner_time_limit: int,
    skip_solve: bool,
) -> ChainReactorRunResult:
    run_dir = output_root / scenario_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result = ChainReactorRunResult(
        scenario=scenario_id,
        run_id=RUN_ID,
        status="initialized",
        output_dir=str(run_dir),
        planner_bin=planner_bin,
        planner_available=_planner_available(planner_bin),
        planner_time_limit_sec=planner_time_limit,
    )

    json_path = run_dir / f"run_{RUN_ID}.json"
    extract_log_path = run_dir / f"run_{RUN_ID}_extract.log"
    solve_log_path = run_dir / f"run_{RUN_ID}_solve.log"
    error_log_path = run_dir / f"run_{RUN_ID}_error.log"
    label = f"{scenario_id}_run_{RUN_ID}"
    root_problem_path = (
        run_dir / f"generated_problems_{label}" / "micronix-problem-root.pddl"
    )

    try:
        with tempfile.TemporaryDirectory(prefix="chainreactor_") as tmpdir:
            private_key_path = Path(tmpdir) / "id_ed25519"
            public_key = _generate_temp_ssh_keypair(private_key_path)

            source = StaticScenarioSource(scenarios=[scenario_id])
            instance = source.build(0)
            instance.config.backend = "local_docker"
            result.container_user = instance.config.container_user
            instance.config.setup_script = _append_setup_script(
                instance.config.setup_script,
                build_chainreactor_prereq_setup_script(),
            )
            instance.config.setup_script = _append_setup_script(
                instance.config.setup_script,
                build_authorized_keys_setup_script(
                    instance.config.container_user,
                    public_key,
                ),
            )

            async with PrivEscScenario(_ssh_from_env(), instance.config) as scenario:
                extract_command = [
                    sys.executable,
                    str(_facts_extractor_path()),
                    "-p",
                    str(scenario.local_port),
                    "-t",
                    "127.0.0.1",
                    "-d",
                    str(_domain_path()),
                    "-n",
                    label,
                    "-s",
                    "-u",
                    instance.config.container_user,
                    "-k",
                    str(private_key_path),
                ]
                extract_result = _run_logged_command(
                    extract_command,
                    cwd=run_dir,
                    log_path=extract_log_path,
                )

        result.extract_command = extract_result.command
        result.extract_return_code = extract_result.return_code
        result.extract_duration_sec = extract_result.duration_sec
        result.extract_log = _relative_to_output_root(extract_log_path, output_root)
        result.extract_succeeded = extract_result.return_code == 0
        result.root_problem_exists = root_problem_path.exists()
        result.root_problem_path = _relative_to_output_root(
            root_problem_path, output_root
        )

        if not result.root_problem_exists:
            result.status = "root_problem_missing"
            return result

        if skip_solve:
            result.status = "solve_skipped"
            return result

        if not result.planner_available:
            result.status = "planner_missing"
            return result

        solve_command = build_planner_command(
            planner_bin,
            domain_path=_domain_path(),
            problem_path=root_problem_path,
            time_limit_sec=planner_time_limit,
        )
        solve_result = _run_logged_command(
            solve_command,
            cwd=run_dir,
            log_path=solve_log_path,
            timeout_seconds=planner_time_limit + 30,
        )
        result.solve_command = solve_result.command
        result.solve_return_code = solve_result.return_code
        result.solve_duration_sec = solve_result.duration_sec
        result.solve_log = _relative_to_output_root(solve_log_path, output_root)
        (
            result.goal_found_time_sec,
            result.planner_total_time_sec,
            result.planner_timed_out,
        ) = parse_powerlifted_timings(solve_result.output)

        plan_path = _first_plan_path(run_dir)
        if plan_path is None:
            result.status = "no_plan"
            return result

        metrics = parse_plan_metrics(plan_path)
        result.plan_found = True
        result.plan_length = metrics.action_count
        result.plan_cost = metrics.cost
        result.plan_path = _relative_to_output_root(plan_path, output_root)
        result.status = "plan_found"
        return result
    except Exception as exc:
        error_log_path.write_text(traceback.format_exc(), encoding="utf-8")
        result.status = "error"
        result.error = str(exc)
        result.error_log = _relative_to_output_root(error_log_path, output_root)
        return result
    finally:
        json_path.write_text(
            json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a ChainReactor static-benchmark audit against local_docker scenarios. "
            "This produces plan-finding results; benchmark_success remains unset until "
            "plan execution is implemented."
        )
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        help=(
            "Benchmark scenario id under conf/scenarios/<name>.yaml. "
            "Repeat to run multiple scenarios. Defaults to the full static benchmark."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for logs, generated problems, plans, and result JSON files.",
    )
    parser.add_argument(
        "--planner-bin",
        default=_default_planner_bin(),
        help=(
            "Planner binary to use for solving. Defaults to $CHAINREACTOR_PLANNER, "
            "then external/powerlifted/bin/powerlifted if present, otherwise "
            f"{SYSTEM_PLANNER_BIN}."
        ),
    )
    parser.add_argument(
        "--planner-time-limit",
        type=int,
        default=DEFAULT_PLANNER_TIME_LIMIT,
        help=(
            "Planner search time limit per scenario in seconds. "
            f"Default: {DEFAULT_PLANNER_TIME_LIMIT}."
        ),
    )
    parser.add_argument(
        "--skip-solve",
        action="store_true",
        help="Only run fact extraction and PDDL generation; skip planner execution.",
    )
    return parser


async def _main_async() -> None:
    args = _build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scenarios = args.benchmark or load_benchmark_scenarios()
    csv_path = output_root / "runs.csv"

    missing_paths = [
        path
        for path in (_chainreactor_root(), _facts_extractor_path(), _domain_path())
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing ChainReactor files: {missing}")

    results: list[ChainReactorRunResult] = []
    for scenario_id in scenarios:
        json_path = output_root / scenario_id / f"run_{RUN_ID}.json"
        if json_path.exists():
            console.print(f"[yellow]Skipping existing[/] {scenario_id} ({json_path})")
            continue

        console.print(f"[cyan]Running[/] {scenario_id}")
        result = await _evaluate_scenario(
            scenario_id,
            output_root=output_root,
            planner_bin=args.planner_bin,
            planner_time_limit=args.planner_time_limit,
            skip_solve=args.skip_solve,
        )
        _append_csv_row(csv_path, result.csv_row(output_root=output_root))
        results.append(result)
        console.print(
            "  -> "
            f"status={result.status} "
            f"extract={int(result.extract_succeeded)} "
            f"root_problem={int(result.root_problem_exists)} "
            f"plan={int(result.plan_found)}"
        )

    if not results:
        console.print("[yellow]No new scenarios were run.[/]")
        return

    plan_found = sum(1 for result in results if result.plan_found)
    root_problem_exists = sum(1 for result in results if result.root_problem_exists)
    extract_succeeded = sum(1 for result in results if result.extract_succeeded)
    console.print(
        "\n[bold]Summary[/]\n"
        f"  scenarios run: {len(results)}\n"
        f"  extraction succeeded: {extract_succeeded}\n"
        f"  root problems generated: {root_problem_exists}\n"
        f"  plans found: {plan_found}\n"
        f"  csv: {csv_path}"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
