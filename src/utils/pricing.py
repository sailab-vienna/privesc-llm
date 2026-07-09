# src/pricing.py

from typing import Dict, Tuple

from src.evaluation.analysis.constants import (
    GEMMA4_31B_MODEL,
    PRIVESC_LLM_4B_MODEL,
    QWEN3_4B_MODEL,
    QWEN3_4B_SFT_MODEL,
)

# Historical RTX 4090 no-batching estimate retained for sensitivity analyses.
RTX4090_SINGLE_AGENT_IN_COST = 0.00135
RTX4090_SINGLE_AGENT_OUT_COST = 0.974

# Paper-facing RTX 5090 batched serving estimates from docs/TOKEN_COST.md.
# Input/output split is fit from replayed real agent prompts, then scaled to
# match the measured whole-workload concurrency-16 serving cost.
QWEN3_4B_RTX5090_IN_COST = 0.0177161823792666
QWEN3_4B_RTX5090_OUT_COST = 0.17625589535946126
GEMMA4_31B_BNB_RTX5090_IN_COST = 0.09585878227256306
GEMMA4_31B_BNB_RTX5090_OUT_COST = 1.8995359009295438
LOCAL_IN_COST = QWEN3_4B_RTX5090_IN_COST
LOCAL_OUT_COST = QWEN3_4B_RTX5090_OUT_COST
GEMMA4_31B_IN_COST = GEMMA4_31B_BNB_RTX5090_IN_COST
GEMMA4_31B_OUT_COST = GEMMA4_31B_BNB_RTX5090_OUT_COST

# Historical RTX 4090 rental estimate on Vast.ai.
RENTAL_IN_COST = 0.00104
RENTAL_OUT_COST = 0.756

# Pricing per 1,000,000 tokens
# Format: model_name -> (input_cost_usd, output_cost_usd)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "anthropic/claude-opus-4.6": (5.00, 25.00),
    "anthropic/claude-opus-4.7": (5.00, 25.00),
    "deepseek/deepseek-v3.2": (0.28, 0.42),
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "deepseek-v4-flash": (0.14, 0.28),
    "google/gemini-3-flash-preview": (0.50, 3.00),
    GEMMA4_31B_MODEL: (GEMMA4_31B_IN_COST, GEMMA4_31B_OUT_COST),
    "openai/gpt-5.2": (1.75, 14.00),
    "qwen3-4b": (LOCAL_IN_COST, LOCAL_OUT_COST),
    "qwen3-sft": (LOCAL_IN_COST, LOCAL_OUT_COST),
    "qwen3-4b-rl": (LOCAL_IN_COST, LOCAL_OUT_COST),
    "rl_qwen3_4b_outcome_only_step1000": (LOCAL_IN_COST, LOCAL_OUT_COST),
    "rl_qwen3_4b_outcome_speed_step1000": (LOCAL_IN_COST, LOCAL_OUT_COST),
    "qwen/qwen3-4b": (LOCAL_IN_COST, LOCAL_OUT_COST),
    "qwen/qwen3-4b-sft": (LOCAL_IN_COST, LOCAL_OUT_COST),
    "qwen/qwen3-4b-sft-rl": (LOCAL_IN_COST, LOCAL_OUT_COST),
    QWEN3_4B_MODEL: (LOCAL_IN_COST, LOCAL_OUT_COST),
    QWEN3_4B_SFT_MODEL: (
        LOCAL_IN_COST,
        LOCAL_OUT_COST,
    ),
    PRIVESC_LLM_4B_MODEL: (LOCAL_IN_COST, LOCAL_OUT_COST),
    "qwen3_4b_instruct_2507_base_paper_static": (LOCAL_IN_COST, LOCAL_OUT_COST),
    "sft_warm_start_checkpoint500_paper_static_repeat": (LOCAL_IN_COST, LOCAL_OUT_COST),
    "prime_rl_t_20260218_164349_step_1000": (LOCAL_IN_COST, LOCAL_OUT_COST),
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates the cost of a model run based on token counts."""
    if model not in MODEL_PRICING:
        return 0.0

    input_price, output_price = MODEL_PRICING[model]
    return (prompt_tokens * input_price / 1_000_000) + (
        completion_tokens * output_price / 1_000_000
    )
