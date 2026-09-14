import argparse
import json
import os
import re
from typing import Any

_RESULT_FILE_PATTERN = re.compile(r"^results_(\d+)(?:_.+)?\.(json|jsonl)$")


def _iter_result_files_desc(benchmark_results_dir: str):
    """Yield (jobid, path) sorted by jobid desc. Raise error if duplicate jobid exists."""
    jobid_to_path: dict[int, str] = {}

    for e in os.scandir(benchmark_results_dir):
        if not e.is_file():
            continue
        m = _RESULT_FILE_PATTERN.match(e.name)
        if not m:
            continue
        jobid = int(m.group(1))
        if jobid in jobid_to_path:
            raise RuntimeError(
                f"Duplicate jobid detected: {jobid}\n  - {jobid_to_path[jobid]}\n  - {e.path}"
            )
        jobid_to_path[jobid] = e.path

    for jobid in sorted(jobid_to_path.keys(), reverse=True):
        yield jobid, jobid_to_path[jobid]


def get_latest_result_for_key(
    benchmark_results_dir: str, result_key: str | list[str]
) -> tuple[dict[str, Any] | None, str | None, int | None, str | None]:
    """
    Find the latest results_*.(json|jsonl) that contains the given result_key.
    Accepts:
      - results_<jobid>.jsonl / .json
      - results_<jobid>_<timestamp>.jsonl / .json
    `result_key` can be:
      - a single key (str)
      - a list of candidate keys (list[str]); the first one found in the latest
        result file will be used.

    Returns (result_all, path, jobid, hit_result_key).
    If not found, returns (None, None, None, None).
    """
    # Normalize to list[str]
    candidate_keys: list[str]
    if isinstance(result_key, str):
        candidate_keys = [result_key]
    else:
        candidate_keys = list(result_key)

    for jobid, path in _iter_result_files_desc(benchmark_results_dir):
        try:
            with open(path, "r", encoding="utf-8") as f:
                result_all = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        results = result_all.get("results")
        if not isinstance(results, dict):
            continue

        present_keys = [key for key in candidate_keys if key in results]
        if not present_keys:
            continue

        if len(present_keys) > 1:
            print(
                "Warning: multiple result_keys found in latest result file; "
                f"candidates={candidate_keys}, present={present_keys}, "
                f"using={present_keys[0]} "
                f"(dir={benchmark_results_dir}, file={os.path.basename(path)}, jobid={jobid})"
            )

        # Use the first matching key in this (latest) result file.
        return result_all, path, jobid, present_keys[0]

    return None, None, None, None


def main(args):
    aggregate_config = [
        json.loads(line) for line in open(args.config_path, "r", encoding="utf-8")
    ]

    aggregate_results = {}
    used_sources = {}

    for conf in aggregate_config:
        benchmark_dir = os.path.join(args.results_dir, conf["benchmark"])
        result_key_conf = conf["result_key"]
        metric_key_conf = conf["metric_key"]
        display_name = conf["display_name"]

        aggregate_results[display_name] = -1
        used_sources[display_name] = None

        if not os.path.exists(benchmark_dir):
            print(f"benchmark directory {benchmark_dir} does not exist")
            continue

        result_all, path, jobid, hit_result_key = get_latest_result_for_key(
            benchmark_dir, result_key_conf
        )
        if (result_all is None) or (hit_result_key is None):
            print(
                f"No result found for {conf['benchmark']} with result_key={result_key_conf}"
            )
            continue

        result = result_all.get("results", {}).get(hit_result_key, {})
        hit_metric_key = None
        for metric_key in (metric_key_conf if isinstance(metric_key_conf, list) else [metric_key_conf]):
            if metric_key in result:
                hit_metric_key = metric_key
                break
        if hit_metric_key is None:
            print(
                f"Missing metric_key={metric_key_conf} in {conf['benchmark']} result_key={hit_result_key} "
                f"(file={os.path.basename(path) if path else 'unknown'})"
            )
            continue

        aggregate_results[display_name] = result[hit_metric_key]
        used_sources[display_name] = {
            "path": path,
            "jobid": jobid,
            "benchmark": conf["benchmark"],
            "result_key": hit_result_key,
            "metric_key": hit_metric_key,
        }

    oneliner_results = ",".join([str(v) for v in aggregate_results.values()])

    aggregate_results_path = os.path.join(args.results_dir, "aggregate_results.jsonl")
    print(f"Aggregate results: {aggregate_results_path}")
    aggregate_results_body = {
        "aggregate_results": aggregate_results,
        "oneliner_results": oneliner_results,
        "aggregate_config": aggregate_config,
        "used_sources": used_sources,
    }
    with open(aggregate_results_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_results_body, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="aggregate_config.jsonl")
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)