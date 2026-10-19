ROOT_PROOF_PATH = "/root/flag.txt"


def root_proof_matches(output: str, expected_proof: str | None) -> bool:
    return bool(expected_proof) and str(expected_proof) in (output or "")


def marked_root_proof_matches(
    output: str,
    expected_proof: str | None,
    begin_marker: str,
    end_marker: str,
) -> bool:
    if not expected_proof:
        return False

    start = 0
    while True:
        begin = output.find(begin_marker, start)
        if begin < 0:
            return False
        end = output.find(end_marker, begin + len(begin_marker))
        if end < 0:
            return False
        if expected_proof in output[begin + len(begin_marker) : end]:
            return True
        start = begin + len(begin_marker)
