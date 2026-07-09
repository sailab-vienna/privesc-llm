"""Tests for procedural password artifact generators."""

import logging

import pytest

from src.generators.base import get_generator_file_default
from src.generators.credential_artifact import (
    PasswordFileGenerator,
    PasswordHistoryGenerator,
)
from src.gym.scenario import PrivEscScenario

from test.solution_utils import run_exploit
from .test_generators_base import ssh_config_from_env

log = logging.getLogger(__name__)
logging.getLogger("asyncssh").setLevel(logging.WARNING)

PASSWORD_FILE_CONFIG = get_generator_file_default("password_file", "password_file", {})
PASSWORD_HISTORY_CONFIG = get_generator_file_default(
    "password_history", "password_history", {}
)
FILE_TEMPLATE_LIST = PASSWORD_FILE_CONFIG["file_templates"]
PLACEMENT_LOCS = PASSWORD_FILE_CONFIG["placement_locations"]
HISTORY_TEMPLATES_LIST = PASSWORD_HISTORY_CONFIG["history_templates"]


def _preferred_path_prefixes(kind: str, user_home: str) -> tuple[str, ...]:
    locations = PLACEMENT_LOCS[kind]
    return tuple(
        f"{str(location).format(user_home=user_home).rstrip('/')}/"
        for location in locations
    )


async def _assert_solution_succeeds(scenario) -> None:
    ssh_config = ssh_config_from_env()

    async with PrivEscScenario(ssh_config, scenario.to_scenario_config()) as sc:
        for exploit in scenario.solution["exploit_tool_calls"]:
            await run_exploit(sc, exploit, scenario.category, is_alternative=False)


class TestPasswordArtifactGenerators:
    def test_categories(self):
        assert PasswordFileGenerator().generate(seed=0).category == "password_file"
        assert (
            PasswordHistoryGenerator().generate(seed=0).category == "password_history"
        )

    def test_deterministic(self):
        file_generator = PasswordFileGenerator()
        history_generator = PasswordHistoryGenerator()

        assert (
            file_generator.generate(seed=42).to_dict()
            == file_generator.generate(seed=42).to_dict()
        )
        assert (
            history_generator.generate(seed=42).to_dict()
            == history_generator.generate(seed=42).to_dict()
        )

    def test_different_file_seeds_vary_artifacts(self):
        generator = PasswordFileGenerator()
        artifact_paths = {
            generator.generate(seed=i).metadata["artifact_path"] for i in range(20)
        }
        assert len(artifact_paths) > 1

    def test_password_file_solution_is_simple_cat_and_login(self):
        scenario = PasswordFileGenerator().generate(seed=0)
        steps = scenario.solution["exploit_tool_calls"]

        assert len(steps) == 2
        assert steps[0]["function"] == "exec_command"
        assert steps[0]["arguments"]["command"].startswith("cat ")
        assert steps[1]["function"] == "test_credentials"

    def test_password_history_solution_searches_then_reads_and_logs_in(self):
        scenario = PasswordHistoryGenerator().generate(seed=0)
        steps = scenario.solution["exploit_tool_calls"]

        assert len(steps) == 3
        assert steps[0]["function"] == "exec_command"
        assert (
            steps[0]["arguments"]["command"] == scenario.metadata["artifact_search_cmd"]
        )
        assert steps[1]["function"] == "exec_command"
        assert steps[2]["function"] == "test_credentials"

    def test_file_metadata_tracks_artifact_path(self):
        scenario = PasswordFileGenerator().generate(seed=0)
        assert scenario.metadata["artifact_mode"] == "file"
        assert scenario.metadata["artifact_path"].endswith(
            scenario.metadata["filename"]
        )
        assert "history_path" not in scenario.metadata

    def test_history_metadata_uses_history_path(self):
        scenario = PasswordHistoryGenerator().generate(seed=0)
        assert scenario.metadata["artifact_mode"] == "history"
        assert scenario.metadata["artifact_path"] == scenario.metadata["history_path"]

    def test_mail_artifacts_default_to_home_or_mail_spool(self):
        template = next(
            item
            for item in FILE_TEMPLATE_LIST
            if item["filename"] == "message_{ticket_number}.eml"
        )
        scenario = PasswordFileGenerator(
            config={"file_templates": [template]}
        ).generate(seed=0)
        user_home = f"/home/{scenario.container_user}"
        assert scenario.metadata["template_kind"] == "mail"
        assert scenario.metadata["artifact_path"].startswith(
            _preferred_path_prefixes("mail", user_home)
        )

    def test_mail_spool_artifacts_use_username_mailbox(self):
        template = next(
            item
            for item in FILE_TEMPLATE_LIST
            if item["filename"] == "message_{ticket_number}.eml"
        )
        for mail_root in ("/var/mail", "/var/spool/mail"):
            scenario = PasswordFileGenerator(
                config={
                    "file_templates": [template],
                    "placement_locations": {
                        "mail": [mail_root],
                        "config": ["{user_home}"],
                        "note": ["{user_home}"],
                    },
                },
            ).generate(seed=0)
            assert (
                scenario.metadata["artifact_path"]
                == f"{mail_root}/{scenario.container_user}"
            )
            assert scenario.metadata["filename"] == scenario.container_user
            assert scenario.metadata["template_filename"].endswith(".eml")
            assert f"{mail_root}/*" in scenario.metadata["artifact_search_cmd"]

    def test_shared_service_artifacts_default_to_service_like_locations(self):
        template = next(
            item for item in FILE_TEMPLATE_LIST if item["filename"] == ".env"
        )
        scenario = PasswordFileGenerator(
            config={"file_templates": [template]}
        ).generate(seed=0)
        user_home = f"/home/{scenario.container_user}"
        assert scenario.metadata["template_kind"] == "config"
        assert scenario.metadata["artifact_path"].startswith(
            _preferred_path_prefixes("config", user_home)
        )

    def test_history_filename_comes_from_config(self):
        scenario = PasswordHistoryGenerator().generate(seed=0)
        history_filename = scenario.metadata["history_path"].rsplit("/", 1)[-1]
        assert history_filename in PASSWORD_HISTORY_CONFIG["history_filenames"]

    def test_history_filename_can_be_sampled_from_config_list(self):
        history_filenames = [".bash_history", ".zsh_history"]
        generator = PasswordHistoryGenerator(
            config={"history_filenames": history_filenames}
        )

        sampled_filenames = {
            generator.generate(seed=i).metadata["history_path"].rsplit("/", 1)[-1]
            for i in range(20)
        }

        assert sampled_filenames == set(history_filenames)

    def test_hint_comes_from_config(self):
        file_scenario = PasswordFileGenerator().generate(seed=0)
        history_scenario = PasswordHistoryGenerator().generate(seed=0)

        assert file_scenario.hint == PASSWORD_FILE_CONFIG["hint"]
        assert history_scenario.hint == PASSWORD_HISTORY_CONFIG["hint"]

    def test_history_noise_commands_come_from_config(self):
        noise_commands = [
            "printf history-noise-1",
            "printf history-noise-2",
            "printf history-noise-3",
            "printf history-noise-4",
            "printf history-noise-5",
            "printf history-noise-6",
        ]
        generator = PasswordHistoryGenerator(
            config={
                "history_templates": [HISTORY_TEMPLATES_LIST[0]],
                "history_noise_commands": noise_commands,
            }
        )
        scenario = generator.generate(seed=0)
        assert generator.history_noise_commands == noise_commands
        assert any(command in scenario.setup_script for command in noise_commands)

    def test_file_search_command_comes_from_configured_templates(self):
        generator = PasswordFileGenerator(
            config={
                "file_templates": [
                    {
                        "filename": "service_secret_{ticket_number}.dat",
                        "content": "root password: {password}",
                    }
                ],
                "placement_locations": {
                    "mail": ["{user_home}"],
                    "config": ["{user_home}"],
                    "note": ["{user_home}"],
                },
            }
        )
        scenario = generator.generate(seed=0)
        assert "service_secret_*.dat" in scenario.metadata["artifact_search_cmd"]
        assert "grep -E" not in scenario.metadata["artifact_search_cmd"]

    def test_history_search_command_comes_from_configured_filename(self):
        generator = PasswordHistoryGenerator(
            config={
                "history_filenames": [".operator_credentials"],
                "history_templates": [HISTORY_TEMPLATES_LIST[0]],
            }
        )
        scenario = generator.generate(seed=0)
        assert "-name .operator_credentials" in scenario.metadata["artifact_search_cmd"]


@pytest.mark.asyncio
@pytest.mark.slow
async def test_password_artifact_generators_are_solvable() -> None:
    for generator in (PasswordFileGenerator(), PasswordHistoryGenerator()):
        scenario = generator.generate(seed=0)
        await _assert_solution_succeeds(scenario)


@pytest.mark.parametrize("file_template", FILE_TEMPLATE_LIST)
def test_procedural_password_file_templates(file_template: dict[str, str]):
    generator = PasswordFileGenerator(
        config={
            "file_templates": [file_template],
            "placement_locations": {
                "mail": ["{user_home}"],
                "config": ["{user_home}"],
                "note": ["{user_home}"],
            },
        }
    )
    scenario = generator.generate(seed=0)
    assert scenario.metadata["artifact_mode"] == "file"
    assert scenario.metadata["template_filename"].startswith(
        file_template["filename"].split("{", 1)[0]
    )
    assert scenario.metadata["artifact_path"] in scenario.setup_script
    assert scenario.metadata["root_password"] in scenario.setup_script


@pytest.mark.parametrize(
    ("kind", "template_filename"),
    [
        ("mail", "message_{ticket_number}.eml"),
        ("config", ".env"),
        ("note", "admin_notes.txt"),
    ],
)
def test_procedural_password_file_uses_explicit_kind_locations(
    kind: str,
    template_filename: str,
):
    template = next(
        item for item in FILE_TEMPLATE_LIST if item["filename"] == template_filename
    )
    generator = PasswordFileGenerator(config={"file_templates": [template]})

    scenario = generator.generate(seed=0)
    assert scenario.metadata["artifact_mode"] == "file"
    artifact_path = scenario.metadata["artifact_path"]
    assert scenario.metadata["template_kind"] == kind
    assert any(
        artifact_path.startswith(
            f"{str(location).format(user_home=f'/home/{scenario.container_user}').rstrip('/')}/"
        )
        or artifact_path
        == str(location).format(user_home=f"/home/{scenario.container_user}").rstrip(
            "/"
        )
        for location in PLACEMENT_LOCS[kind]
    )


def test_password_file_template_kind_overrides_filename_heuristic():
    generator = PasswordFileGenerator(
        config={
            "file_templates": [
                {
                    "filename": "operator_notes.env",
                    "kind": "note",
                    "content": "root password: {password}",
                }
            ],
            "placement_locations": {
                "mail": ["{user_home}/mail"],
                "config": ["{user_home}/.config"],
                "note": ["{user_home}/notes"],
            },
        }
    )

    scenario = generator.generate(seed=0)

    assert scenario.metadata["template_kind"] == "note"
    assert f"/home/{scenario.container_user}/notes/" in scenario.metadata[
        "artifact_path"
    ]


def test_password_file_rejects_unknown_template_kind():
    with pytest.raises(ValueError, match="file_templates kind"):
        PasswordFileGenerator(
            config={
                "file_templates": [
                    {
                        "filename": "creds.txt",
                        "kind": "secret",
                        "content": "root password: {password}",
                    }
                ],
                "placement_locations": {
                    "mail": ["{user_home}"],
                    "config": ["{user_home}"],
                    "note": ["{user_home}"],
                },
            }
        )


def test_password_file_fails_when_template_kind_has_no_explicit_location():
    with pytest.raises(ValueError, match="placement_locations.mail"):
        PasswordFileGenerator(
            config={
                "file_templates": [
                    item
                    for item in FILE_TEMPLATE_LIST
                    if item["filename"] == "message_{ticket_number}.eml"
                ],
                "placement_locations": {
                    "mail": [],
                    "config": ["{user_home}"],
                    "note": ["{user_home}"],
                },
            }
        )


def test_password_file_fails_when_selected_kind_has_no_candidate_location():
    with pytest.raises(
        ValueError, match="placement_locations.mail must contain at least one location"
    ):
        PasswordFileGenerator(
            config={
                "file_templates": [
                    item for item in FILE_TEMPLATE_LIST if item["filename"] == "new_mail"
                ],
                "placement_locations": {
                    "mail": [""],
                    "config": ["{user_home}"],
                    "note": ["{user_home}"],
                },
            }
        ).generate(seed=0)


@pytest.mark.parametrize("history_template", HISTORY_TEMPLATES_LIST)
def test_procedural_password_history_templates(history_template: str):
    generator = PasswordHistoryGenerator(
        config={"history_templates": [history_template]}
    )
    scenario = generator.generate(seed=0)
    assert scenario.metadata["artifact_mode"] == "history"
    assert scenario.metadata["history_template"] == history_template
    assert scenario.metadata["root_password"] in scenario.metadata["history_entry"]
    assert scenario.metadata["history_path"] in scenario.setup_script
