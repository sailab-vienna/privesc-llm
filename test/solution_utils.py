import logging

log = logging.getLogger("test_solutions")


def assert_result_matches_expected(result, expected: dict, command: str):
    """Assert that the result matches the expected values."""
    if hasattr(result, "exit_code"):
        assert result.got_root == expected["got_root"]
        assert result.exit_code == expected["exit_code"]
        for expected_str in expected["output_contains"]:
            assert expected_str in result.output
    else:
        assert result.got_root == expected["got_root"]
        assert result.success == expected["success"]


async def run_exploit(
    scenario, exploit: dict, scenario_name: str, is_alternative: bool = False
):
    """Run a single exploit (primary or alternative)."""
    function_name = exploit["function"]
    arguments = exploit["arguments"]
    exploit_type = "alternative" if is_alternative else "primary"

    if function_name == "exec_command":
        log.info(f"Testing {exploit_type} exploit command: {arguments['command']}")
        result = await scenario.exec_command(arguments["command"])
        assert_result_matches_expected(
            result, exploit["expected_result"], arguments["command"]
        )
        log.info(
            f"✅ {exploit_type.capitalize()} exploit successful: {arguments['command']}"
        )
        return result

    if function_name == "test_credentials":
        log.info(
            f"Testing {exploit_type} credentials: {arguments['user']}:{arguments['password']}"
        )
        result = await scenario.test_credentials(
            arguments["user"], arguments["password"]
        )
        assert_result_matches_expected(
            result,
            exploit["expected_result"],
            f"test_credentials({arguments['user']}, {arguments['password']})",
        )
        log.info(
            f"✅ {exploit_type.capitalize()} credentials successful: {arguments['user']}:{arguments['password']}"
        )
        return result

    raise AssertionError(f"Unsupported exploit function: {function_name}")
