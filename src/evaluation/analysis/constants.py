import re
from collections.abc import Iterable

ROUND_BUDGETS = list(range(5, 65, 5))
MAX_ROUNDS = 60
PRIMARY_ROUND_BUDGET = 20
PARETO_ROUND_BUDGET = PRIMARY_ROUND_BUDGET
RUNS_PER_SCENARIO = 10
SCENARIOS_PER_MODEL = 12
RUNS_PER_MODEL = RUNS_PER_SCENARIO * SCENARIOS_PER_MODEL
PRIVESC_LLM_NAME = "PrivEsc-LLM 4B"
CLAUDE_OPUS_47_MODEL = "anthropic/claude-opus-4.7"
DEEPSEEK_V32_MODEL = "deepseek/deepseek-v3.2"
GEMMA4_31B_MODEL = "gemma4_31b_it_thinking_gemma4_tools"
PRIVESC_LLM_4B_MODEL = "e4_outcome_cost_step_300"
QWEN3_4B_MODEL = "qwen3_4b_instruct_2507"
QWEN3_4B_SFT_MODEL = "qwen3_4b_sft_unguided_phase_c_lr1p5e-4_r8_sd2026_static"
E1_GUIDED_LONG = (
    "e1_guided_deepseek_long_reasoning_"
    "20260517T173728Z_phase_c_lr1p5e-4_r8_ep10_sd1337"
)
E1_GUIDED_SHORT = (
    "e1_guided_deepseek_short_reasoning_"
    "20260517T173728Z_phase_c_lr1p5e-4_r8_ep10_sd1337"
)
E1_GUIDED_NO = (
    "e1_guided_deepseek_no_reasoning_"
    "20260517T173728Z_phase_c_lr1p5e-4_r8_ep10_sd1337"
)
E1_UNGUIDED_LONG = (
    "qwen3_4b_sft_unguided_phase_b_lr1p5e-4_r8_sd1337"
)
E1_UNGUIDED_SHORT = (
    "e1_unguided_deepseek_short_reasoning_"
    "20260517T173728Z_phase_c_lr1p5e-4_r8_ep10_sd1337"
)
E1_UNGUIDED_NO = (
    "e1_unguided_deepseek_no_reasoning_"
    "20260517T173728Z_phase_c_lr1p5e-4_r8_ep10_sd1337"
)
E1_TRACE_MODEL_ORDER = [
    E1_UNGUIDED_LONG,
    E1_UNGUIDED_SHORT,
    E1_UNGUIDED_NO,
    E1_GUIDED_LONG,
    E1_GUIDED_SHORT,
    E1_GUIDED_NO,
]
PAPER_MODEL_ORDER = [
    *E1_TRACE_MODEL_ORDER,
    CLAUDE_OPUS_47_MODEL,
    PRIVESC_LLM_4B_MODEL,
    DEEPSEEK_V32_MODEL,
    GEMMA4_31B_MODEL,
    QWEN3_4B_SFT_MODEL,
    QWEN3_4B_MODEL,
    "openai/gpt-5.2",
    "google/gemini-3-flash-preview",
    "qwen3-4b-rl",
    "qwen3-sft",
    "qwen3-4b",
    "rl_qwen3_4b_outcome_only_step1000",
    "rl_qwen3_4b_outcome_speed_step1000",
]
PRETTY_MODEL_NAMES = {
    E1_GUIDED_LONG: "Guided\nLong",
    E1_GUIDED_SHORT: "Guided\nShort",
    E1_GUIDED_NO: "Guided\nNo",
    E1_UNGUIDED_LONG: "Unguided\nLong",
    E1_UNGUIDED_SHORT: "Unguided\nShort",
    E1_UNGUIDED_NO: "Unguided\nNo",
    "anthropic/claude-opus-4.6": "Claude Opus 4.6",
    CLAUDE_OPUS_47_MODEL: "Claude Opus 4.7",
    "openai/gpt-5.2": "GPT-5.2",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    DEEPSEEK_V32_MODEL: "DeepSeek V3.2",
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "llama3_2_3b_instruct": "Llama 3.2 3B Instruct",
    "qwen3_14b_fp8_nonthinking": "Qwen3 14B (thinking off)",
    "qwen3_14b_fp8_thinking": "Qwen3 14B (thinking on)",
    "qwen3_8b_nonthinking": "Qwen3 8B (thinking off)",
    "qwen3_8b_thinking": "Qwen3 8B (thinking on)",
    "e4_outcome_step_200": "Outcome",
    "e4_outcome_round_step_400": "+ Round",
    "e4_outcome_round_cost_step_200": "+ Round+Cost",
    GEMMA4_31B_MODEL: "Gemma 4 31B",
    PRIVESC_LLM_4B_MODEL: PRIVESC_LLM_NAME,
    QWEN3_4B_MODEL: "Qwen3 4B",
    QWEN3_4B_SFT_MODEL: "Qwen3 4B SFT",
    "qwen3-4b-rl": "PrivEsc-LLM (legacy)",
    "rl_qwen3_4b_outcome_only_step1000": "Outcome-Only RL",
    "rl_qwen3_4b_outcome_speed_step1000": "Outcome+Speed RL",
    "qwen3-sft": "Qwen3-4B SFT",
    "qwen3-4b": "Qwen3-4B",
}
PLOT_EXTENSIONS = ["png", "svg", "pdf"]


def natural_sort_key(s: str) -> int:
    match = re.search(r"^\d+", s)
    return int(match.group()) if match else -1


SCENARIO_DISPLAY_NAMES = {
    "01_vuln_suid_gtfo": "SUID GTFOBins",
    "02_vuln_password_in_shell_history": "Password in shell history",
    "03_vuln_sudo_no_password": "Sudo no password",
    "04_vuln_sudo_gtfo_interactive": "Sudo GTFOBins (interactive)",
    "05_vuln_sudo_gtfo": "Sudo GTFOBins",
    "06_vuln_docker": "Docker group escape",
    "07_root_password_reuse_mysql": "Password reuse (MySQL)",
    "08_root_password_reuse": "Password reuse",
    "09_root_password_root": "Weak root password",
    "10_root_allows_lowpriv_to_ssh": "Root allows lowpriv SSH",
    "11_cron_calling_user_wildcard": "Cron wildcard injection",
    "12_cron_calling_user_file": "Writable cron script",
    "13_file_with_root_password": "Password in file",
    "capabilities_gtfobins": "Capabilities GTFOBins",
    "cron_wildcard": "Cron wildcard",
    "cron_writable_script": "Writable cron script",
    "password_file": "Password in file",
    "password_history": "Password in history",
    "password_reuse": "Password reuse",
    "ssh_key_reuse": "SSH key reuse",
    "sudo_gtfobins": "Sudo GTFOBins",
    "suid_gtfobins": "SUID GTFOBins",
    "weak_password": "Weak password",
}


def short_scenario_label(s: str) -> str:
    if s in SCENARIO_DISPLAY_NAMES:
        return SCENARIO_DISPLAY_NAMES[s]
    rest = s.split("_", 1)[1] if "_" in s else s
    return rest.replace("_", " ")


def format_model_name(model: str) -> str:
    return PRETTY_MODEL_NAMES.get(model, model.split("/")[-1])


def ordered_model_ids(models: Iterable[str]) -> list[str]:
    models_present = set(models)
    return [m for m in PAPER_MODEL_ORDER if m in models_present] + sorted(
        m for m in models_present if m not in PAPER_MODEL_ORDER
    )
