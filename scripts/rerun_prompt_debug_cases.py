import argparse
import copy
import json
import os
import time

from label_router_configs import (
    append_jsonl,
    load_train_dataset,
    run_one_config,
    set_openai_key_from_config,
)


DEFAULT_CASE_LINES = "50,61,71,93,1,10,68"


def parse_case_lines(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def config_for_record(record):
    if record.get("label"):
        return record["label"]
    attempts = record.get("attempts") or []
    if not attempts:
        raise ValueError(f"No label or attempts for record id={record.get('id')}")
    best = max(
        attempts,
        key=lambda attempt: attempt.get("metrics", {})
        .get("prediction_llm", {})
        .get("f1", 0)
        or 0,
    )
    return best["config"]


def main():
    parser = argparse.ArgumentParser(
        description="Rerun selected existing router-label cases with the current prompt/code."
    )
    parser.add_argument(
        "--input_label_file",
        default="router_labels/RoG-cwq_train_router_labels.jsonl",
    )
    parser.add_argument("--case_lines", default=DEFAULT_CASE_LINES)
    parser.add_argument(
        "--output_file",
        default=None,
    )
    parser.add_argument("--benchmark", default="RoG-cwq")
    parser.add_argument("--data_path", default="rmanluo")
    parser.add_argument("--crlt_data_dir", default="datasets/crlt")
    parser.add_argument("--save_cache", default="cache")
    parser.add_argument("--output_path", default="results")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0125")
    parser.add_argument("--embedding_model", default="text-embedding-3-small")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--strategy", default="discrete_rating")
    parser.add_argument("--verifier", default="deductive+planning")
    parser.add_argument("--disable_termination_verification", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--add_hop_information", action="store_true")
    parser.add_argument("--N_CPUS", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--openai_timeout", type=float, default=60)
    parser.add_argument("--openai_max_attempts", type=int, default=3)
    parser.add_argument("--require_path_support", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    set_openai_key_from_config()

    if args.output_file is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_file = os.path.join(
            "prompt_debug_results",
            f"{args.benchmark}_prompt_debug_{stamp}.jsonl",
        )

    records = load_jsonl(args.input_label_file)
    case_lines = parse_case_lines(args.case_lines)
    selected = []
    for line_no in case_lines:
        if line_no < 1 or line_no > len(records):
            raise ValueError(f"Line {line_no} is outside 1..{len(records)}")
        selected.append((line_no, records[line_no - 1]))

    dataset = load_train_dataset(args)
    id_to_index = {sample_id: idx for idx, sample_id in enumerate(dataset["id"])}
    navigators = {}

    for line_no, old_record in selected:
        sample_id = old_record["id"]
        if sample_id not in id_to_index:
            raise ValueError(f"Could not find sample id in processed dataset: {sample_id}")
        sample = dataset[id_to_index[sample_id]]
        config = copy.deepcopy(config_for_record(old_record))
        attempt = run_one_config(sample, args, config, navigators)
        row = {
            "source_line": line_no,
            "id": sample_id,
            "question": old_record.get("question"),
            "ground_truth": old_record.get("ground_truth"),
            "old_status": old_record.get("status"),
            "old_label": old_record.get("label"),
            "rerun_config": config,
            "rerun_is_correct": attempt.get("is_correct"),
            "rerun_prediction_llm": attempt.get("prediction_llm"),
            "rerun_prediction_direct_answer": attempt.get("prediction_direct_answer"),
            "rerun_reasoning_path": attempt.get("reasoning_path"),
            "rerun_metrics": attempt.get("metrics"),
        }
        append_jsonl(args.output_file, row)
        print(
            f"line={line_no} id={sample_id} config={config} "
            f"correct={row['rerun_is_correct']} pred={row['rerun_prediction_llm']!r}"
        )

    print(f"Saved prompt debug results to {args.output_file}")


if __name__ == "__main__":
    main()
