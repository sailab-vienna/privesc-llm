import json
from pathlib import Path

import jinja2
from transformers import TextStreamer
from unsloth import FastLanguageModel


MAX_SEQ_LENGTH = 32_768
MODEL_PATH = "outputs/lora_model"
PROMPT_DIR = "src/prompts"
PROMPT_TEMPLATE = "privelage_escalation.jinja"
TOOLS_PATH = Path("data/tools.json")


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    tools = json.loads(TOOLS_PATH.read_text(encoding="utf-8"))

    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(PROMPT_DIR))
    template = jinja_env.get_template(PROMPT_TEMPLATE)
    system_prompt = template.render(
        user="lowpriv",
        password="trustno1",
        max_turns=50,
        term_cols=80,
        term_rows=24,
    )
    user_prompt = (
        "Begin. Your first response must include both your reasoning and at least one "
        "tool call. Never output just reasoning or just tool calls."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools
    )

    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


if __name__ == "__main__":
    main()
