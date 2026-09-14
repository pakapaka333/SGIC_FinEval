import argparse
import hashlib
import os
import re
from pathlib import Path

from datasets import load_dataset
from typing import Optional
import pandas as pd
from vllm import LLM, SamplingParams


HUGGINGFACE_DATASET_ID = "DeepResaerch/dr-finance-instruction"
USE_SPLIT = "train"
QUESTION_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompt_templates/question_prompt.txt"
QUESTION_TEMPLATE_REPLACE_KEYS = ["input"]
REFERENCE_ANSWER_KEY = "output"


THINK_PATTERN = {
    "qwen3": r".*</think>(.*)",
    "qwen-3": r".*</think>(.*)",
    "gpt-oss": r".*assistantfinal(.*)",
}
def _get_think_pattern(
    self,
    model_name: str,
) -> Optional[str]:
    for n, p in THINK_PATTERN.items():
        if n in model_name.lower():
            return p
    return None

def _strip_think_tag(
    self,
    text: str,
    think_pattern: str,
) -> str:
    match = re.search(think_pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def main(args):
    ds = load_dataset(
        HUGGINGFACE_DATASET_ID,
        split=USE_SPLIT,
    )
    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    with open(QUESTION_TEMPLATE_PATH, "r") as f:
        question_template = f.read() 
    
    replace_values = [
        {key: ds[key][i] for key in QUESTION_TEMPLATE_REPLACE_KEYS}
        for i in range(len(ds))
    ]
    questions = [
        question_template.format(**vals)
        for vals in replace_values
        ]
    question_ids = [
        hashlib.md5(q.encode("utf-8")).hexdigest()
        for q in questions
    ]

    model = LLM(
        model=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=args.num_gpus,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=args.stop_tokens.split(",") if args.stop_tokens else None,
    )

    print(f"Generating model answers for {len(questions)} questions.")

    if args.apply_chat_template:
        model_inputs = [[{"role": "user", "content": inp}] for inp in questions]
        model_outputs = model.chat(model_inputs, sampling_params)
    else:
        model_inputs = questions
        model_outputs = model.generate(questions, sampling_params)
    
    print(f"Generated {len(model_outputs)} model answers.")

    think_pattern = _get_think_pattern(args, args.model_name)
    if think_pattern is not None:
        model_output_texts = [
            _strip_think_tag(args, output.outputs[0].text, think_pattern)
            for output in model_outputs
        ]
    else:
        model_output_texts = [output.outputs[0].text for output in model_outputs]

    result_df = pd.DataFrame(
        {
            "question_id": question_ids,
            "question": questions,
            "reference_answer": ds[REFERENCE_ANSWER_KEY],
            "model_answer": model_output_texts,
            "model_name": args.model_name,
        }
    )
    result_df.to_json(args.output_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--stop-words", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)           # 0.0 is set for greedy generation.
    parser.add_argument("--max-tokens", type=int, default=128)              # 128 is based on the maximum token length among the reference answers (65 tokens).
    parser.add_argument("--stop-tokens", type=str, default="質問:,回答:")   # stop tokens for avoiding repetition of the question and answer.
    parser.add_argument("--apply-chat-template", action="store_true")
    args = parser.parse_args()

    main(args)
