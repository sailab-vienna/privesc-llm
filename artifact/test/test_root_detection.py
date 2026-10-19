from src.utils.root_detection import marked_root_proof_matches, root_proof_matches


def test_root_proof_matches_expected_value() -> None:
    assert root_proof_matches("root-proof-123\n", "root-proof-123") is True


def test_prompt_like_output_does_not_count_without_proof() -> None:
    assert root_proof_matches("root@container:/#\n", "root-proof-123") is False
    assert root_proof_matches("#\n", "root-proof-123") is False


def test_marked_root_proof_matches_probe_payload() -> None:
    output = "ignored\nBEGIN\nroot-proof-123\nEND\n"

    assert marked_root_proof_matches(output, "root-proof-123", "BEGIN", "END") is True


def test_marked_root_proof_ignores_echoed_probe_markers() -> None:
    output = (
        "printf '\\nBEGIN\\n'; cat /root/flag.txt; printf '\\nEND\\n'\n"
        "BEGIN\n"
        "root-proof-123\n"
        "END\n"
    )

    assert marked_root_proof_matches(output, "root-proof-123", "BEGIN", "END") is True
