import argparse
import json
from multiprocessing import Value
from optparse import Values

import pandas as pd


def main(args):
    judgment_df = pd.read_json(args.judgment_path, lines=True)
    num_samples = len(judgment_df)
    num_true = (judgment_df["judgment"] == True).sum()
    num_false = (judgment_df["judgment"] == False).sum()
    num_none = (judgment_df["judgment"].isna()).sum()
    print(f"True: {num_true}, False: {num_false}, None: {num_none}, Total: {num_samples}")

    acc_all = num_true / num_samples
    num_samples_wo_none = num_samples - num_none
    acc_wo_none = num_true / num_samples_wo_none if num_samples_wo_none != 0 else float("nan")
    print(f"Accuracy: {acc_all:.2f}, Accuracy without None: {acc_wo_none:.2f}")

    result_json = {
        "results": {
            "dr_finance_instruction": {
                "num_samples": int(num_samples),
                "num_true": int(num_true),
                "num_false": int(num_false),
                "num_none": int(num_none),
                "acc_all": acc_all,
                "acc_wo_none": acc_wo_none,
            }
        },
        "versions": {
            "dr_finance_instruction": 1,
        },
        "config": {
            "model": args.model_name,
            "judge_model": args.judge_model,
            "judgment_path": args.judgment_path,
        }
    }
    json.dump(result_json, open(args.output_path, 'w'), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--judge-model", type=str, required=True)
    parser.add_argument("--judgment-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(args)