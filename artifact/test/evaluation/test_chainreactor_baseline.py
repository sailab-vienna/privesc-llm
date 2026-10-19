from pathlib import Path

from src.evaluation.chainreactor_baseline import (
    _append_setup_script,
    _default_planner_bin,
    build_authorized_keys_setup_script,
    build_planner_command,
    parse_plan_metrics,
    parse_powerlifted_timings,
)


def test_append_setup_script_preserves_existing_script() -> None:
    combined = _append_setup_script("echo first", "echo second")
    assert combined == "echo first\n\necho second"


def test_build_authorized_keys_setup_script_targets_container_user_home() -> None:
    script = build_authorized_keys_setup_script(
        "lowpriv",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey chainreactor",
    )
    assert "getent passwd lowpriv" in script
    assert 'home_dir="/home/lowpriv"' not in script
    assert "home_dir=/home/lowpriv" in script
    assert 'install -d -m 700 -o lowpriv -g lowpriv "$home_dir/.ssh"' in script
    assert "authorized_keys" in script
    assert "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey chainreactor" in script


def test_default_planner_bin_respects_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CHAINREACTOR_PLANNER", "/tmp/custom-powerlifted")
    assert _default_planner_bin() == "/tmp/custom-powerlifted"


def test_build_planner_command_matches_chainreactor_solver_contract() -> None:
    command = build_planner_command(
        "/opt/powerlifted",
        domain_path=Path("/tmp/domain.pddl"),
        problem_path=Path("/tmp/problem.pddl"),
        time_limit_sec=30,
    )
    assert command == [
        "/opt/powerlifted",
        "--iteration",
        "alt-bfws1,rff,yannakakis,476",
        "--unit-cost",
        "--preprocess-task",
        "--only-effects-novelty-check",
        "--time-limit",
        "30",
        "-d",
        str(Path("/tmp/domain.pddl").resolve()),
        "-i",
        str(Path("/tmp/problem.pddl").resolve()),
    ]


def test_parse_powerlifted_timings_extracts_goal_and_total_time() -> None:
    output = "\n".join(
        [
            "Goal found at: 9.97039",
            "Total time: 9.97041",
            "Solution found.",
        ]
    )

    goal_found_time, planner_total_time, timed_out = parse_powerlifted_timings(output)

    assert goal_found_time == 9.97039
    assert planner_total_time == 9.97041
    assert timed_out is False


def test_parse_plan_metrics_counts_actions_and_cost(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.1"
    plan_path.write_text(
        "\n".join(
            [
                "(derive_user_can_execute_file lowpriv_u lowpriv_g usr_bin_find)",
                "(spawn_suid_process lowpriv_u root_u lowpriv_g root_g usr_bin_find process)",
                "; cost = 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = parse_plan_metrics(plan_path)

    assert metrics.action_count == 2
    assert metrics.cost == 2
