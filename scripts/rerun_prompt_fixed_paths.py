import argparse
import copy
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

from src.prompts import cwq as prompt_cwq  # noqa: E402
from src.utils.llm_backbone import LLM_Backbone  # noqa: E402
from scripts.label_router_configs import (  # noqa: E402
    append_jsonl,
    apply_path_support_guard,
    score_for_benchmark,
    set_openai_key_from_config,
)


DEFAULT_CASE_LINES = "50,61,71,93,1,10,68"


def parse_case_lines(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def selected_attempt(record):
    if record.get("status") == "resolved" and record.get("attempts"):
        return record["attempts"][-1]
    attempts = record.get("attempts") or []
    return max(
        attempts,
        key=lambda attempt: attempt.get("metrics", {})
        .get("prediction_llm", {})
        .get("f1", 0)
        or 0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rerun only the final answer prompt on existing saved reasoning paths."
    )
    parser.add_argument(
        "--input_label_file",
        default="router_labels/RoG-cwq_train_router_labels.jsonl",
    )
    parser.add_argument("--case_lines", default=DEFAULT_CASE_LINES)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--benchmark", default="RoG-cwq")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0125")
    parser.add_argument("--embedding_model", default="text-embedding-3-small")
    parser.add_argument("--openai_timeout", type=float, default=60)
    parser.add_argument("--openai_max_attempts", type=int, default=3)
    parser.add_argument("--require_path_support", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    set_openai_key_from_config()
    if args.output_file is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_file = os.path.join(
            "prompt_debug_results",
            f"{args.benchmark}_fixed_path_prompt_debug_{stamp}.jsonl",
        )

    records = load_jsonl(args.input_label_file)
    llm = LLM_Backbone(args)
    for line_no in parse_case_lines(args.case_lines):
        record = records[line_no - 1]
        attempt = selected_attempt(record)
        reasoning_paths = attempt.get("reasoning_path") or []
        prompt = copy.deepcopy(prompt_cwq.reasoning_prompt)
        prompt["prompt"] = prompt["prompt"].format(
            question=record["question"],
            reasoning_path="\n".join(reasoning_paths),
        )
        prediction = llm.get_completion(prompt).replace("Answer: ", "").strip()
        metrics = score_for_benchmark(prediction, record["ground_truth"], args.benchmark)
        if args.require_path_support:
            metrics = apply_path_support_guard(
                metrics,
                prediction,
                reasoning_paths,
                args.benchmark,
            )

        row = {
            "source_line": line_no,
            "id": record["id"],
            "question": record["question"],
            "ground_truth": record["ground_truth"],
            "old_status": record.get("status"),
            "old_label": record.get("label"),
            "old_config": attempt.get("config"),
            "old_prediction_llm": attempt.get("prediction_llm"),
            "fixed_reasoning_path": reasoning_paths,
            "rerun_prediction_llm": prediction,
            "rerun_metrics": metrics,
            "rerun_is_correct": metrics.get("f1") == 1.0,
        }
        append_jsonl(args.output_file, row)
        print(
            f"line={line_no} id={record['id']} "
            f"correct={row['rerun_is_correct']} pred={prediction!r}"
        )

    print(f"Saved fixed-path prompt debug results to {args.output_file}")


if __name__ == "__main__":
    main()
