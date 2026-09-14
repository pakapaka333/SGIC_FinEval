import argparse
import os
from pathlib import Path
from tqdm import tqdm
import time

import pandas as pd
from openai import OpenAI


JUDGE_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompt_templates/judge_prompt.txt"
JUDGE_TEMPLATE_REPLACE_KEYS = ["question", "reference_answer", "model_answer"]


def parse_judgment(text: str):
    text = text.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def main(args):
    answer_df = pd.read_json(args.answer_path, lines=True)
    for key in JUDGE_TEMPLATE_REPLACE_KEYS:
        if key not in answer_df.columns:
            raise ValueError(
                f"Required column, {key}, not found in judge target file, {args.answer_path}."
            )

    with open(JUDGE_TEMPLATE_PATH, "r") as f:
        judge_template = f.read()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    judgments = []
    for i in tqdm(range(len(answer_df)), desc="Generating judgments", mininterval=3.0):
        replace_values = {
            key: answer_df[key].iloc[i]
            for key in JUDGE_TEMPLATE_REPLACE_KEYS
        }
        prompt = judge_template.format(**replace_values)

        judgment = None
        for attempt in range(args.max_retries):
            try:
                response = client.chat.completions.create(
                    model=args.judge_model,
                    max_completion_tokens=2,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )

                judge_cand = response.choices[0].message.content
                judgment = parse_judgment(judge_cand)

                if judgment is not None:
                    # Succeeded in getting a judgment
                    break

                # Failed to get a judgment
                print(
                    f"[WARN] Failed to get a judgment. Retry {attempt + 1}/{args.max_retries} for index {i}: {judge_cand}."
                )

            except Exception as e:
                print(
                    f"[WARN] Something went wrong. Retry {attempt + 1}/{args.max_retries} for index {i}: {e}."
                )

            # Sleep for the retry interval
            time.sleep(args.retry_interval)

        if judgment is None:
            print(f"[WARN] Failed to get a judgment after {args.max_retries} retries for index {i}. Use None as the judgment.")

        judgments.append(judgment)

    print(f"Generated {len(judgments)} judgments")
    result_df = pd.DataFrame(
        {
            "question_id": answer_df["question_id"],
            "model_answer": answer_df["model_answer"],
            "answer_path": args.answer_path,
            "judgment": judgments,
            "judge_model": args.judge_model,
        }
    )
    result_df.to_json(args.output_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--judge-model", type=str, required=True)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-interval", type=float, default=0.5)
    args = parser.parse_args()
    main(args)