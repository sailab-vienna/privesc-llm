import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.config import AppConfig
from src.dataset.privesc.collection_stats import (
    SFTAssemblyTargetChecker,
    build_collection_audit,
    format_target_status,
    holdout_leakage_rejection_reasons,
    missing_total,
    summarize_trace_files,
    summary_generators,
)


def _write_trace(
    directory: Path,
    name: str,
    *,
    generator: str,
    success: bool,
    benchmark_eligible: bool = True,
    pre_repair_reasons: list[str] | None = None,
) -> Path:
    metadata: dict[str, object] = {
        "benchmark_eligible": benchmark_eligible,
        "generator_name": generator,
        "seed": 42,
        "prompt_vars": {"max_turns": 12},
    }
    if pre_repair_reasons is not None:
        metadata["pre_repair_rejected"] = True
        metadata["pre_repair_rejection_reasons"] = pre_repair_reasons

    path = directory / name
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": success,
                "turns": 3 if success else 12,
                "error": None,
                "metadata": metadata,
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _assistant(command: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": (
            "I will run this grounded shell check and use the command output before "
            "choosing any specific escalation path."
        ),
        "tool_calls": [
            {
                "type": "function",
                "id": call_id,
                "function": {
                    "name": "exec_command",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def _tool(
    call_id: str,
    command: str,
    output: str,
    *,
    got_root: bool = False,
    exit_code: int = 0,
) -> dict:
    return {
        "role": "tool",
        "name": "exec_command",
        "tool_call_id": call_id,
        "content": json.dumps(
            {
                "got_root": got_root,
                "command": command,
                "output": output,
                "exit_code": exit_code,
                "timed_out": got_root,
            }
        ),
    }


def _assistant_credentials(user: str, password: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": (
            "I have credential evidence, so I will test this weak credential "
            "hypothesis now."
        ),
        "tool_calls": [
            {
                "type": "function",
                "id": call_id,
                "function": {
                    "name": "test_credentials",
                    "arguments": json.dumps({"user": user, "password": password}),
                },
            }
        ],
    }


def _credential_tool(call_id: str, user: str, password: str) -> dict:
    return {
        "role": "tool",
        "name": "test_credentials",
        "tool_call_id": call_id,
        "content": json.dumps(
            {
                "got_root": True,
                "user": user,
                "password": password,
                "message": "Authentication successful",
                "success": True,
            }
        ),
    }


def _write_behavior_trace(directory: Path) -> Path:
    metadata = {
        "benchmark_eligible": True,
        "generator_name": "suid_gtfobins",
        "seed": 7,
        "binary_path": "/opt/custom/vulnmarker",
        "binary_name": "vulnmarker",
        "exploit_cmds": ["/usr/local/bin/find . -exec /bin/sh -p \\; -quit"],
        "prompt_vars": {"max_turns": 12},
    }
    commands = [
        "stat /opt/custom/vulnmarker 2>/dev/null || true",
        "stat /opt/custom/vulnmarker 2>/dev/null || true",
        "definitely-not-a-command",
        "/usr/local/bin/find . -exec /bin/sh -p \\; -quit",
    ]
    path = directory / "suid_gtfobins_behavior.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 4,
                "error": None,
                "metadata": metadata,
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant(commands[0], "call-1"),
                    _tool("call-1", commands[0], ""),
                    _assistant(commands[1], "call-2"),
                    _tool("call-2", commands[1], "/opt/custom/vulnmarker"),
                    _assistant(commands[2], "call-3"),
                    _tool(
                        "call-3",
                        commands[2],
                        "sh: definitely-not-a-command: command not found",
                        exit_code=127,
                    ),
                    _assistant(commands[3], "call-4"),
                    _tool("call-4", commands[3], "#", got_root=True, exit_code=124),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_root_root_trace(directory: Path) -> Path:
    metadata = {
        "benchmark_eligible": True,
        "generator_name": "weak_password",
        "seed": 9,
        "root_password": "root",
        "prompt_vars": {"max_turns": 12},
    }
    path = directory / "weak_password_root_root.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": metadata,
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant_credentials("root", "root", "call-1"),
                    _credential_tool("call-1", "root", "root"),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_reasoning_only_benchmark_word_trace(directory: Path) -> Path:
    metadata = {
        "benchmark_eligible": True,
        "generator_name": "suid_gtfobins",
        "seed": 10,
        "prompt_vars": {"max_turns": 12},
    }
    path = directory / "suid_gtfobins_reasoning_find.json"
    assistant = _assistant("id", "call-1")
    assistant["content"] = (
        "I will find SUID binaries before selecting an exploit, then verify "
        "whether the observed command output gives root."
    )
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": metadata,
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    assistant,
                    _tool("call-1", "id", "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_reasoning_credential_trace(
    directory: Path, content: str, *, name: str, seed: int
) -> Path:
    metadata = {
        "benchmark_eligible": True,
        "generator_name": "weak_password",
        "seed": seed,
        "prompt_vars": {"max_turns": 12},
    }
    assistant = _assistant("id", "call-1")
    assistant["content"] = content
    path = directory / name
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": metadata,
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    assistant,
                    _tool("call-1", "id", "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_generic_sudo_trace(directory: Path) -> Path:
    metadata = {
        "benchmark_eligible": True,
        "generator_name": "ssh_key_reuse",
        "seed": 34,
        "prompt_vars": {"max_turns": 12},
    }
    path = directory / "ssh_key_reuse_sudo_i.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": metadata,
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant("sudo -i", "call-1"),
                    _tool("call-1", "sudo -i", "#", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_benchmark_binary_trace(directory: Path, *, in_metadata: bool) -> Path:
    metadata = {
        "benchmark_eligible": True,
        "generator_name": "suid_gtfobins",
        "seed": 11 if in_metadata else 12,
        "prompt_vars": {"max_turns": 12},
    }
    if in_metadata:
        metadata["binary_name"] = "find"
    path = directory / f"suid_gtfobins_binary_{in_metadata}.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": metadata,
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant("printf 'find\\n'", "call-1"),
                    _tool("call-1", "printf 'find\\n'", "find\n", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_generic_history_discovery_trace(directory: Path) -> Path:
    history_path = "/home/user/.zsh_history"
    path = directory / "password_history_generic_discovery.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 3,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "password_history",
                    "seed": 16,
                    "root_password": "winter2026!",
                    "history_path": history_path,
                    "artifact_path": history_path,
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant(
                        "find /home/user -maxdepth 2 -type f \\( -name '.bash_history' -o -name '*history' \\)",
                        "call-1",
                    ),
                    _tool("call-1", "find", f"{history_path}\n"),
                    _assistant(f"cat {history_path}", "call-2"),
                    _tool("call-2", f"cat {history_path}", "mysql -p'winter2026!'"),
                    _assistant_credentials("root", "winter2026!", "call-3"),
                    _credential_tool("call-3", "root", "winter2026!"),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_generic_ssh_key_name_discovery_trace(directory: Path) -> Path:
    path = directory / "ssh_key_reuse_generic_key_name.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "ssh_key_reuse",
                    "seed": 23,
                    "key_path": "/home/user/.ssh/deploy_key",
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant(
                        "find /home/user/.ssh -type f -name id_rsa -o -name '*key*'",
                        "call-1",
                    ),
                    _tool("call-1", "find", "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_benchmark_history_command_trace(directory: Path) -> Path:
    path = directory / "password_history_benchmark_command.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "password_history",
                    "seed": 22,
                    "history_path": "/home/user/.zsh_history",
                    "root_password": "winter2026!",
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant("cat /home/lowpriv/.bash_history", "call-1"),
                    _tool(
                        "call-1",
                        "cat /home/lowpriv/.bash_history",
                        "uid=0(root)",
                        got_root=True,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_parallel_credential_batch_trace(directory: Path) -> Path:
    path = directory / "weak_password_parallel_batch.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "weak_password",
                    "seed": 24,
                    "root_password": "changeme",
                    "password_source": "common_password",
                    "credential_candidates": [
                        {
                            "user": "root",
                            "password": "changeme",
                            "password_source": "common_password",
                            "gives_root": True,
                        }
                    ],
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    {
                        "role": "assistant",
                        "content": (
                            "A weak-password hypothesis is plausible, so I will test "
                            "independent credential guesses in one parallel batch now."
                        ),
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call-1",
                                "function": {
                                    "name": "test_credentials",
                                    "arguments": json.dumps(
                                        {"user": "root", "password": "changeme"}
                                    ),
                                },
                            },
                            {
                                "type": "function",
                                "id": "call-2",
                                "function": {
                                    "name": "test_credentials",
                                    "arguments": json.dumps(
                                        {"user": "root", "password": "password123"}
                                    ),
                                },
                            },
                        ],
                    },
                    _credential_tool("call-1", "root", "changeme"),
                    {
                        "role": "tool",
                        "name": "test_credentials",
                        "tool_call_id": "call-2",
                        "content": json.dumps(
                            {
                                "got_root": False,
                                "user": "root",
                                "password": "password123",
                                "message": "Authentication failed",
                                "success": False,
                            }
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_exact_benchmark_exploit_command_trace(directory: Path) -> Path:
    path = directory / "sudo_gtfobins_benchmark_exploit_command.json"
    command = "docker run --rm -it -v /:/mnt alpine chroot /mnt /bin/sh"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "sudo_gtfobins",
                    "seed": 24,
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant(command, "call-1"),
                    _tool("call-1", command, "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_benchmark_shaped_exploit_command_trace(directory: Path) -> Path:
    path = directory / "suid_gtfobins_benchmark_shaped_command.json"
    command = "/opt/custom/customfind . -exec /bin/bash -p \\; -quit"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "suid_gtfobins",
                    "seed": 25,
                    "binary_name": "customfind",
                    "binary_path": "/opt/custom/customfind",
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant(command, "call-1"),
                    _tool("call-1", command, "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_benchmark_history_metadata_trace(directory: Path) -> Path:
    path = directory / "password_history_benchmark_metadata.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "password_history",
                    "seed": 17,
                    "history_path": "/home/lowpriv/.bash_history",
                    "artifact_path": "/home/lowpriv/.bash_history",
                    "root_password": "winter2026!",
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant("id", "call-1"),
                    _tool("call-1", "id", "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_generic_bash_history_metadata_trace(directory: Path) -> Path:
    path = directory / "password_history_generic_bash_metadata.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "password_history",
                    "seed": 18,
                    "history_path": "/home/user/.bash_history",
                    "artifact_path": "/home/user/.bash_history",
                    "root_password": "winter2026!",
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant("id", "call-1"),
                    _tool("call-1", "id", "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_generic_cron_script_name_trace(directory: Path) -> Path:
    path = directory / "cron_wildcard_generic_script_name.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "cron_wildcard",
                    "seed": 18,
                    "backup_dir": "/home/user/uploads",
                    "exploit_script_name": "exploit.sh",
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant("id", "call-1"),
                    _tool("call-1", "id", "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_generic_cron_checkpoint_trace(directory: Path) -> Path:
    backup_dir = "/home/user/uploads"
    command = (
        f"echo 'chmod u+s /bin/bash' > {backup_dir}/exploit.sh && "
        f": > {backup_dir}/'--checkpoint=1' && "
        f": > {backup_dir}/'--checkpoint-action=exec=sh exploit.sh'"
    )
    path = directory / "cron_wildcard_generic_checkpoint.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "cron_wildcard",
                    "seed": 23,
                    "backup_dir": backup_dir,
                    "exploit_script_name": "exploit.sh",
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant(command, "call-1"),
                    _tool("call-1", command, "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_observation_grounded_cron_path_recon_trace(directory: Path) -> Path:
    backup_dir = "/home/user/uploads"
    path = directory / "cron_wildcard_path_recon.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 4,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "cron_wildcard",
                    "seed": 22,
                    "backup_dir": backup_dir,
                    "exploit_script_name": "runme.sh",
                    "exploit_cmds": [
                        f"echo 'chmod u+s /bin/bash' > {backup_dir}/runme.sh",
                        "sleep 65 && /bin/bash -p",
                    ],
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant("ls -la /home/user", "call-1"),
                    _tool("call-1", "ls -la /home/user", "drwxr-xr-x uploads"),
                    _assistant(f"ls -la {backup_dir}", "call-2"),
                    _tool("call-2", f"ls -la {backup_dir}", f"drwxr-xr-x {backup_dir}"),
                    _assistant(
                        f"echo 'chmod u+s /bin/bash' > {backup_dir}/runme.sh",
                        "call-3",
                    ),
                    _tool("call-3", f"echo x > {backup_dir}/runme.sh", ""),
                    _assistant("sleep 65 && /bin/bash -p", "call-4"),
                    _tool(
                        "call-4", "sleep 65 && /bin/bash -p", "bash-5.2#", got_root=True
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_history_template_benchmark_binary_trace(directory: Path) -> Path:
    path = directory / "password_history_template_binary.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "password_history",
                    "seed": 19,
                    "history_template": "docker login -u root -p {password}",
                    "history_path": "/home/user/.zsh_history",
                    "root_password": "winter2026!",
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant("id", "call-1"),
                    _tool("call-1", "id", "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_turn_quality_trace(directory: Path) -> Path:
    path = directory / "weak_password_turn_quality.json"
    grounded_reasoning = (
        "I have enough command output to continue with a concrete shell check now."
    )
    rooted_reasoning = "Root is visible in the previous command output, but this extra command should be rejected."
    rooted_assistant = _assistant("whoami", "call-3")
    rooted_assistant["content"] = rooted_reasoning
    normal_assistant = _assistant("id", "call-2")
    normal_assistant["content"] = grounded_reasoning
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 3,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "weak_password",
                    "seed": 13,
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    {**_assistant("id", "call-1"), "content": ""},
                    _tool("call-1", "id", "uid=1000(user)"),
                    {
                        "role": "assistant",
                        "content": "I need one more note.",
                        "tool_calls": [],
                    },
                    normal_assistant,
                    _tool("call-2", "id", "uid=0(root)", got_root=True),
                    rooted_assistant,
                    _tool("call-3", "whoami", "root", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_text_only_assistant_trace(directory: Path) -> Path:
    path = directory / "weak_password_text_only_assistant.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 2,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "weak_password",
                    "seed": 23,
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    {
                        "role": "assistant",
                        "content": (
                            "I have enough context to make one concise plan before "
                            "running the next grounded command."
                        ),
                        "tool_calls": [],
                    },
                    _assistant("id", "call-1"),
                    _tool("call-1", "id", "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_root_root_metadata_variant_trace(directory: Path) -> Path:
    path = directory / "weak_password_root_root_metadata_variant.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "weak_password",
                    "seed": 14,
                    "prompt_vars": {"max_turns": 12},
                    "intended_target_role": "root",
                    "intended_password": "root",
                    "credential_candidates": [
                        {"user": "root", "password": "root", "gives_root": True}
                    ],
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    _assistant("id", "call-1"),
                    _tool("call-1", "id", "uid=0(root)", got_root=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_password_reuse_known_password_trace(directory: Path) -> Path:
    password = "winter2026!"
    assistant = _assistant_credentials("root", password, "call-1")
    assistant["content"] = (
        "The current login password is known from the session, so a password reuse "
        "check against root is a grounded credential hypothesis."
    )
    path = directory / "password_reuse_known_password.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "password_reuse",
                    "seed": 15,
                    "prompt_vars": {"max_turns": 12},
                    "root_password": password,
                    "user_password": password,
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    assistant,
                    _credential_tool("call-1", "root", password),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_weak_password_common_guess_trace(directory: Path) -> Path:
    password = "changeme"
    assistant = _assistant_credentials("root", password, "call-1")
    assistant["content"] = (
        "No stronger path is visible, so I will test a short common weak-password "
        "guess for root before continuing broader enumeration."
    )
    path = directory / "weak_password_common_guess.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "weak_password",
                    "seed": 20,
                    "root_password": password,
                    "password_source": "common_password",
                    "credential_candidates": [
                        {
                            "user": "root",
                            "password": password,
                            "role": "root",
                            "password_source": "common_password",
                            "gives_root": True,
                        }
                    ],
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    assistant,
                    _credential_tool("call-1", "root", password),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_weak_password_service_guess_trace(directory: Path) -> Path:
    password = "vault"
    assistant = _assistant_credentials("root", password, "call-1")
    assistant["content"] = (
        "No stronger path is visible, so I will test vault as a root password now."
    )
    path = directory / "weak_password_service_guess.json"
    path.write_text(
        json.dumps(
            {
                "mode": "trace_collection",
                "status": "completed",
                "success": True,
                "turns": 1,
                "error": None,
                "metadata": {
                    "benchmark_eligible": True,
                    "generator_name": "weak_password",
                    "seed": 21,
                    "root_password": password,
                    "password_source": "service_name",
                    "credential_candidates": [
                        {
                            "user": "root",
                            "password": password,
                            "role": "root",
                            "password_source": "service_name",
                            "gives_root": True,
                        }
                    ],
                    "prompt_vars": {"max_turns": 12},
                },
                "history": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                    assistant,
                    _credential_tool("call-1", "root", password),
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_collection_stats_report_teacher_solve_and_rejects(tmp_path: Path) -> None:
    paths = [
        _write_trace(
            tmp_path,
            "weak_password_success.json",
            generator="weak_password",
            success=True,
        ),
        _write_trace(
            tmp_path,
            "weak_password_failed.json",
            generator="weak_password",
            success=False,
        ),
        _write_trace(
            tmp_path,
            "weak_password_leak.json",
            generator="weak_password",
            success=False,
            benchmark_eligible=False,
            pre_repair_reasons=["secret_solution_leakage (scenario description)"],
        ),
        _write_trace(
            tmp_path,
            "ssh_key_reuse_adjacent.json",
            generator="ssh_key_reuse",
            success=False,
            benchmark_eligible=False,
            pre_repair_reasons=["holdout_leakage:exact_benchmark_path"],
        ),
    ]

    summary = summarize_trace_files(
        paths, generators=["weak_password", "ssh_key_reuse"]
    )

    weak = summary["by_generator"]["weak_password"]
    assert weak["raw"] == 3
    assert weak["usable"] == 1
    assert weak["teacher_solved"] == 1
    assert weak["teacher_solve_rate"] == pytest.approx(1 / 3)
    assert weak["leakage_rejected"] == 1

    ssh = summary["by_generator"]["ssh_key_reuse"]
    assert ssh["usable"] == 0
    assert ssh["holdout_leakage_rejected"] == 1

    assert summary["total"]["raw"] == 4
    assert summary["total"]["usable"] == 1
    assert summary["total"]["teacher_solve_rate"] == pytest.approx(0.25)
    assert summary["rejection_reasons"] == {
        "secret_solution_leakage (scenario description)": 1,
        "holdout_leakage:exact_benchmark_path": 1,
    }

    assert (
        missing_total(
            summary,
            generators=["weak_password", "ssh_key_reuse"],
            target_per_generator=2,
        )
        == 3
    )
    status = format_target_status(
        summary,
        generators=["weak_password", "ssh_key_reuse"],
        target_per_generator=2,
    )
    assert "usable=1/4" in status
    assert "missing=3" in status


def test_target_status_prefers_sft_usable_count() -> None:
    summary = {
        "total": {
            "raw": 2,
            "usable": 2,
            "sft_usable": 1,
            "teacher_solve_rate": 1.0,
            "leakage_rejected": 0,
            "audit_rejected": 0,
            "live_rejected": 0,
        },
        "by_generator": {
            "cron_writable_script": {
                "raw": 2,
                "usable": 2,
                "sft_usable": 1,
                "teacher_solve_rate": 1.0,
                "leakage_rejected": 0,
                "audit_rejected": 0,
                "live_rejected": 0,
            }
        },
    }

    assert (
        missing_total(
            summary,
            generators=["cron_writable_script"],
            target_per_generator=2,
        )
        == 1
    )
    status = format_target_status(
        summary,
        generators=["cron_writable_script"],
        target_per_generator=2,
    )
    assert "usable=1/2" in status
    assert "missing=1" in status


def test_target_generators_ignore_stale_observed_generators(tmp_path: Path) -> None:
    stale_trace = _write_trace(
        tmp_path,
        "ssh_key_reuse_stale.json",
        generator="ssh_key_reuse",
        success=True,
    )

    summary, _, _ = build_collection_audit(
        [stale_trace],
        generators=["weak_password"],
    )

    assert "ssh_key_reuse" in summary["by_generator"]
    assert summary_generators(summary) == ["weak_password"]
    assert (
        missing_total(
            summary,
            generators=summary_generators(summary),
            target_per_generator=2,
        )
        == 2
    )


def test_sft_target_checker_rejects_empty_history() -> None:
    checker = SFTAssemblyTargetChecker(OmegaConf.structured(AppConfig()))
    result = checker.check({"mode": "trace_collection", "success": True, "history": []})

    assert result == {
        "sft_usable": False,
        "sft_rejection_reasons": ["missing_history"],
    }


def test_sft_target_checker_requires_history() -> None:
    checker = SFTAssemblyTargetChecker(OmegaConf.structured(AppConfig()))

    with pytest.raises(ValueError, match="list-valued 'history'"):
        checker.check({"mode": "trace_collection", "success": True})


def test_collection_stats_report_behavior_and_diversity(tmp_path: Path) -> None:
    summary = summarize_trace_files(
        [_write_behavior_trace(tmp_path)],
        generators=["suid_gtfobins"],
    )

    stats = summary["by_generator"]["suid_gtfobins"]
    assert stats["avg_turns"] == pytest.approx(4.0)
    assert stats["avg_tool_calls"] == pytest.approx(4.0)
    assert stats["median_root_round"] == 4
    assert stats["avg_evidence_to_exploit_latency"] == pytest.approx(2.0)
    assert stats["repeated_command_traces"] == 1
    assert stats["repeated_command_trace_rate"] == pytest.approx(1.0)
    assert stats["invalid_command_traces"] == 1
    assert stats["invalid_command_trace_rate"] == pytest.approx(1.0)
    assert stats["unique_task_ids"] == 1
    assert stats["unique_seeds"] == 1
    assert stats["unique_solution_signature_hashes"] == 1
    assert stats["traces_per_task_mean"] == pytest.approx(1.0)
    assert stats["traces_per_task_max"] == 1

    total = summary["total"]
    assert total["avg_tool_calls"] == pytest.approx(stats["avg_tool_calls"])
    assert total["unique_solution_signature_hashes"] == 1


def test_collection_audit_rejects_root_root_success(tmp_path: Path) -> None:
    trace_path = _write_root_root_trace(tmp_path)
    summary, task_rows, leakage_rows = build_collection_audit(
        [trace_path],
        generators=["weak_password"],
    )

    stats = summary["by_generator"]["weak_password"]
    assert stats["raw"] == 1
    assert stats["teacher_solved"] == 1
    assert stats["usable"] == 0
    assert stats["audit_rejected"] == 1
    assert stats["structured_rejection_classes"]["exact_benchmark_credential"] == 1
    assert summary["structured_rejection_classes"]["exact_benchmark_credential"] == 1
    assert summary["leakage_class_counts"]["benchmark_credential_pair"] == 1

    assert task_rows[0]["expected_solution_signature_hash"]
    assert task_rows[0]["audit_usable"] is False
    assert leakage_rows[0]["hard_rejection_classes"] == ["exact_benchmark_credential"]
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert holdout_leakage_rejection_reasons(payload) == [
        "holdout_leakage:exact_benchmark_credential"
    ]


def test_collection_audit_ignores_benchmark_words_in_reasoning(tmp_path: Path) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_reasoning_only_benchmark_word_trace(tmp_path)],
        generators=["suid_gtfobins"],
    )

    stats = summary["by_generator"]["suid_gtfobins"]
    assert stats["usable"] == 1
    assert "exact_benchmark_binary" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_rejects_exact_credential_in_reasoning(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [
            _write_reasoning_credential_trace(
                tmp_path,
                "The password reuse evidence supports testing root:trustno1 now.",
                name="weak_password_reasoning_trustno1.json",
                seed=31,
            )
        ],
        generators=["weak_password"],
    )

    stats = summary["by_generator"]["weak_password"]
    assert stats["usable"] == 0
    assert "exact_benchmark_credential" in leakage_rows[0]["hard_rejection_classes"]
    assert "benchmark_credential_pair" in leakage_rows[0]["leakage_classes"]


def test_collection_audit_counts_root_root_guess_without_rejecting(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [
            _write_reasoning_credential_trace(
                tmp_path,
                "A weak default-password hypothesis supports trying root:root.",
                name="weak_password_reasoning_root_root.json",
                seed=32,
            )
        ],
        generators=["weak_password"],
    )

    assert summary["by_generator"]["weak_password"]["usable"] == 1
    assert "benchmark_credential_pair" in leakage_rows[0]["leakage_classes"]
    assert "exact_benchmark_credential" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_allows_root_root_ownership_in_reasoning(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [
            _write_reasoning_credential_trace(
                tmp_path,
                "The setup likely used chown root:root on the target binary.",
                name="weak_password_reasoning_chown.json",
                seed=33,
            )
        ],
        generators=["weak_password"],
    )

    assert summary["by_generator"]["weak_password"]["usable"] == 1
    assert "exact_benchmark_credential" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_counts_generic_sudo_shell_without_rejecting(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_generic_sudo_trace(tmp_path)],
        generators=["ssh_key_reuse"],
    )

    assert summary["by_generator"]["ssh_key_reuse"]["usable"] == 1
    assert "benchmark_exploit_fragments" in leakage_rows[0]["leakage_classes"]
    assert (
        "exact_benchmark_exploit_fragment"
        not in leakage_rows[0]["hard_rejection_classes"]
    )


def test_collection_audit_scopes_binary_atoms_to_solution_surfaces(
    tmp_path: Path,
) -> None:
    _, _, output_rows = build_collection_audit(
        [_write_benchmark_binary_trace(tmp_path, in_metadata=False)],
        generators=["suid_gtfobins"],
    )
    assert "exact_benchmark_binary" not in output_rows[0]["hard_rejection_classes"]

    _, _, metadata_rows = build_collection_audit(
        [_write_benchmark_binary_trace(tmp_path, in_metadata=True)],
        generators=["suid_gtfobins"],
    )
    assert "exact_benchmark_binary" in metadata_rows[0]["hard_rejection_classes"]


def test_collection_audit_treats_generic_history_search_as_audit_only(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_generic_history_discovery_trace(tmp_path)],
        generators=["password_history"],
    )

    stats = summary["by_generator"]["password_history"]
    assert stats["usable"] == 1
    assert stats["leakage_class_counts"]["benchmark_filenames"] == 1
    assert "exact_benchmark_path" not in leakage_rows[0]["hard_rejection_classes"]
    assert "oracle_before_evidence" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_treats_generic_ssh_key_search_as_audit_only(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_generic_ssh_key_name_discovery_trace(tmp_path)],
        generators=["ssh_key_reuse"],
    )

    stats = summary["by_generator"]["ssh_key_reuse"]
    assert stats["usable"] == 1
    assert stats["leakage_class_counts"]["benchmark_ssh_key_names"] == 1
    assert "exact_benchmark_path" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_counts_benchmark_artifact_commands_without_rejecting(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_benchmark_history_command_trace(tmp_path)],
        generators=["password_history"],
    )

    assert summary["by_generator"]["password_history"]["usable"] == 1
    assert "benchmark_file_paths" in leakage_rows[0]["leakage_classes"]
    assert "exact_benchmark_path" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_rejects_exact_benchmark_exploit_fragments(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_exact_benchmark_exploit_command_trace(tmp_path)],
        generators=["sudo_gtfobins"],
    )

    assert summary["by_generator"]["sudo_gtfobins"]["usable"] == 0
    assert (
        "exact_benchmark_exploit_fragment"
        in leakage_rows[0]["hard_rejection_classes"]
    )


def test_collection_audit_counts_fuzzy_skeleton_overlap_without_rejecting(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_benchmark_shaped_exploit_command_trace(tmp_path)],
        generators=["suid_gtfobins"],
    )

    assert summary["by_generator"]["suid_gtfobins"]["usable"] == 1
    assert leakage_rows[0]["command_skeleton_overlaps"]
    assert "benchmark_command_skeleton" not in leakage_rows[0]["hard_rejection_classes"]
    assert "benchmark_command_skeleton" not in leakage_rows[0]["leakage_classes"]


def test_collection_audit_allows_parallel_batch_results_after_root(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_parallel_credential_batch_trace(tmp_path)],
        generators=["weak_password"],
    )

    stats = summary["by_generator"]["weak_password"]
    assert stats["usable"] == 1
    assert stats["post_root_continuation_turns"] == 0
    assert "post_root_continuation" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_rejects_benchmark_history_solution_metadata(
    tmp_path: Path,
) -> None:
    _, _, leakage_rows = build_collection_audit(
        [_write_benchmark_history_metadata_trace(tmp_path)],
        generators=["password_history"],
    )

    assert "exact_benchmark_path" in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_allows_generic_bash_history_solution_metadata(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_generic_bash_history_metadata_trace(tmp_path)],
        generators=["password_history"],
    )

    assert summary["by_generator"]["password_history"]["usable"] == 1
    assert "benchmark_filenames" not in leakage_rows[0]["leakage_classes"]
    assert "exact_benchmark_path" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_allows_generic_cron_script_name(tmp_path: Path) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_generic_cron_script_name_trace(tmp_path)],
        generators=["cron_wildcard"],
    )

    assert summary["by_generator"]["cron_wildcard"]["usable"] == 1
    assert "exact_benchmark_path" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_allows_generic_cron_checkpoint_syntax(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_generic_cron_checkpoint_trace(tmp_path)],
        generators=["cron_wildcard"],
    )

    assert summary["by_generator"]["cron_wildcard"]["usable"] == 1
    assert (
        "exact_benchmark_exploit_fragment"
        not in leakage_rows[0]["hard_rejection_classes"]
    )


def test_collection_audit_allows_observation_grounded_path_recon(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_observation_grounded_cron_path_recon_trace(tmp_path)],
        generators=["cron_wildcard"],
    )

    assert summary["by_generator"]["cron_wildcard"]["usable"] == 1
    assert "oracle_before_evidence" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_scopes_binary_atoms_to_exploit_metadata(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_history_template_benchmark_binary_trace(tmp_path)],
        generators=["password_history"],
    )

    assert summary["by_generator"]["password_history"]["usable"] == 1
    assert "exact_benchmark_binary" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_counts_turn_quality_failures(tmp_path: Path) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_turn_quality_trace(tmp_path)],
        generators=["weak_password"],
    )

    stats = summary["by_generator"]["weak_password"]
    assert stats["usable"] == 1
    assert stats["near_empty_reasoning_tool_call_turns"] == 1
    assert stats["assistant_turns_without_tools"] == 1
    assert stats["post_root_continuation_turns"] == 1
    assert stats["near_empty_reasoning_tool_call_traces"] == 1
    assert stats["assistant_turns_without_tools_traces"] == 1
    assert stats["post_root_continuation_traces"] == 1
    assert leakage_rows[0]["hard_rejection_classes"] == []
    assert (
        "assistant_turn_without_tools" not in leakage_rows[0]["hard_rejection_classes"]
    )
    assert (
        "near_empty_reasoning_tool_call_turn" not in leakage_rows[0]["leakage_classes"]
    )
    assert "post_root_continuation" not in leakage_rows[0]["leakage_classes"]


def test_collection_audit_counts_text_only_assistant_turns_without_rejecting(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_text_only_assistant_trace(tmp_path)],
        generators=["weak_password"],
    )

    stats = summary["by_generator"]["weak_password"]
    assert stats["usable"] == 1
    assert stats["assistant_turns_without_tools"] == 1
    assert "assistant_turn_without_tools" not in leakage_rows[0]["leakage_classes"]
    assert (
        "assistant_turn_without_tools" not in leakage_rows[0]["hard_rejection_classes"]
    )


def test_collection_audit_rejects_root_root_metadata_variants(tmp_path: Path) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_root_root_metadata_variant_trace(tmp_path)],
        generators=["weak_password"],
    )

    stats = summary["by_generator"]["weak_password"]
    assert stats["usable"] == 0
    assert stats["structured_rejection_classes"]["exact_benchmark_credential"] == 1
    assert "exact_benchmark_credential" in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_allows_known_password_reuse_guess(tmp_path: Path) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_password_reuse_known_password_trace(tmp_path)],
        generators=["password_reuse"],
    )

    stats = summary["by_generator"]["password_reuse"]
    assert stats["usable"] == 1
    assert "oracle_before_evidence" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_allows_weak_password_common_guess(tmp_path: Path) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_weak_password_common_guess_trace(tmp_path)],
        generators=["weak_password"],
    )

    stats = summary["by_generator"]["weak_password"]
    assert stats["usable"] == 1
    assert "oracle_before_evidence" not in leakage_rows[0]["hard_rejection_classes"]


def test_collection_audit_counts_ungrounded_service_password_guess(
    tmp_path: Path,
) -> None:
    summary, _, leakage_rows = build_collection_audit(
        [_write_weak_password_service_guess_trace(tmp_path)],
        generators=["weak_password"],
    )

    stats = summary["by_generator"]["weak_password"]
    assert stats["usable"] == 1
    assert leakage_rows[0]["oracle_before_evidence"]
    assert "oracle_before_evidence" not in leakage_rows[0]["hard_rejection_classes"]
    assert "oracle_before_evidence" not in leakage_rows[0]["leakage_classes"]
