from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import InitVar, dataclass, field

import hydra
import pyte
from hydra.core.hydra_config import HydraConfig
from rich.console import Console

from src.config import AppConfig, ScenarioConfig, SSHConfig, register_with_hydra
from src.gym.backends import (
    TIMEOUT_EXIT_CODE,
    build_backend,
    calculate_command_timeout,
)
from src.tui.agent_panels import AgentLogger

log = logging.getLogger("privesc_scenario")


@dataclass
class ToolResult:
    got_root: bool


@dataclass
class ExecResult(ToolResult):
    command: str
    output: str
    exit_code: int
    timed_out: bool = False
    got_root: bool = field(init=False)
    root_verified: InitVar[bool] = False

    def __post_init__(self, root_verified: bool) -> None:
        self.got_root = root_verified


@dataclass
class AuthResult(ToolResult):
    user: str
    password: str
    message: str = ""
    success: bool = False
    uid: int | None = None
    got_root: bool = field(init=False)
    root_verified: InitVar[bool] = False

    def __post_init__(self, root_verified: bool) -> None:
        self.got_root = self.success and root_verified


def _render_terminal_output(data: str, cols: int = 80, lines: int = 24) -> str:
    screen = pyte.Screen(cols, lines)
    stream = pyte.Stream(screen)
    stream.feed(data)
    return "\n".join(line.strip() for line in screen.display).strip()


class PrivEscScenario:
    def __init__(
        self,
        ssh_cfg: SSHConfig,
        scen_cfg: ScenarioConfig,
        logger: AgentLogger | None = None,
        console: Console | None = None,
    ) -> None:
        if not scen_cfg.image:
            raise ValueError(
                f"Scenario '{scen_cfg.name}' missing required 'image' field. "
                "Implicit image fallback has been removed."
            )
        self._scen_cfg = scen_cfg
        self.logger = logger
        self._console = console
        self._backend = build_backend(ssh_cfg, scen_cfg, self._console_print)
        if scen_cfg.max_parallel_tool_calls < 1:
            raise ValueError("max_parallel_tool_calls must be >= 1")
        self._tool_call_semaphore = asyncio.Semaphore(scen_cfg.max_parallel_tool_calls)

    def _console_print(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        elif self._console:
            self._console.print(message)

    async def __aenter__(self) -> PrivEscScenario:
        await self._backend.start()
        try:
            await self.reset()
        except Exception:
            await self._backend.close()
            raise
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._backend.close()

    async def reset(self) -> None:
        start = time.perf_counter()
        await self._backend.reset()
        total_duration = time.perf_counter() - start
        self._console_print(f"[dim]  Environment ready in {total_duration:.1f}s[/]")

    async def exec_command(self, command: str) -> ExecResult:
        async with self._tool_call_semaphore:
            timeout = calculate_command_timeout(command, self._scen_cfg)
            outcome = await self._backend.run_command(command, timeout)
        return ExecResult(
            command=command,
            output=_render_terminal_output(
                outcome.output,
                cols=self._scen_cfg.term_cols,
                lines=self._scen_cfg.term_rows,
            ),
            exit_code=outcome.exit_code,
            timed_out=outcome.timed_out,
            root_verified=outcome.root_verified,
        )

    async def test_credentials(self, user: str, password: str) -> AuthResult:
        async with self._tool_call_semaphore:
            outcome = await self._backend.test_credentials(user, password)
        return AuthResult(
            user=user,
            password=password,
            success=outcome.success,
            message=outcome.message,
            uid=outcome.uid,
            root_verified=outcome.root_verified,
        )

    async def close(self) -> None:
        await self.__aexit__(None, None, None)

    @property
    def local_port(self) -> int:
        local_port = self._backend.local_port
        if local_port is None:
            raise RuntimeError("Scenario local port not available for current backend")
        return int(local_port)


def get_prompt(exit_code: int) -> str:
    if exit_code == 0:
        return "$ "
    if exit_code == TIMEOUT_EXIT_CODE:
        return "\n[TIMEOUT]$ "
    return f"\n[FAIL:{exit_code}]$ "


async def check_and_log_credentials(sc: PrivEscScenario, lf, user: str, password: str):
    auth_result = await sc.test_credentials(user, password)
    if auth_result.success:
        log.info(
            "Credentials '%s:%s' are valid. Got root: %s",
            auth_result.user,
            auth_result.password,
            auth_result.got_root,
        )
        lf.write(
            f"--- Credentials '{auth_result.user}:{auth_result.password}' are valid. Got root: {auth_result.got_root} ---\n"
        )
    else:
        log.warning(
            "Credentials '%s:%s' are not valid: %s. Got root: %s",
            auth_result.user,
            auth_result.password,
            auth_result.message,
            auth_result.got_root,
        )
        lf.write(
            f"--- Credentials '{auth_result.user}:{auth_result.password}' are not valid: {auth_result.message}. Got root: {auth_result.got_root} ---\n"
        )


async def handle_interaction(sc: PrivEscScenario, log_path: str):
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("--- Session Start ---\n")
        try:
            await check_and_log_credentials(sc, lf, "lowpriv", "trustno1")
            await check_and_log_credentials(sc, lf, "root", "aim8Du7h")
            print("$ ", end="", flush=True)

            while True:
                try:
                    command = input()
                except (EOFError, KeyboardInterrupt):
                    log.info("Exiting...")
                    break

                if not command.strip():
                    print("$ ", end="", flush=True)
                    continue

                lf.write("--- Input ---\n")
                lf.write(command + "\n")
                result = await sc.exec_command(command)
                print(result.output)
                prompt = get_prompt(result.exit_code)
                print(prompt, end="", flush=True)

                lf.write("--- Output ---\n")
                lf.write(result.output.strip() + "\n")
                lf.write(f"--- Exit Code: {result.exit_code} ---\n")
                if result.got_root:
                    log.info("Root shell detected!")
                    lf.write("--- Root Shell Detected ---\n")
                lf.flush()
        finally:
            lf.write("--- Session End ---\n")


async def interactive_session(cfg: AppConfig):
    log_path = f"{HydraConfig.get().runtime.output_dir}/interaction.log"
    log.info("Session will be recorded to %s, Ctrl-D to quit", log_path)
    async with PrivEscScenario(cfg.ssh, cfg.scenario, console=Console()) as sc:
        await handle_interaction(sc, log_path)


@hydra.main(version_base=None, config_path="../../conf", config_name="app")
def main(cfg: AppConfig):
    logging.getLogger("asyncssh").setLevel(logging.WARNING)
    log.info("Starting application with config: %s", cfg)
    asyncio.run(interactive_session(cfg))


if __name__ == "__main__":
    register_with_hydra()
    main()
