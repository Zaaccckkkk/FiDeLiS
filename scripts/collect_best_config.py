import argparse
import csv
import json
import re
import string
from pathlib import Path


def normalize(s):
    s = str(s).lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(prediction, answer):
    return normalize(answer) in normalize(prediction)


def split_prediction(prediction):
    if isinstance(prediction, list):
        return [str(item) for item in prediction if str(item).strip()]
    return [item for item in str(prediction).split("\n") if item.strip()]


def evaluate_prediction(prediction, answers):
    answers = [str(answer) for answer in answers]
    prediction_items = split_prediction(prediction)
    prediction_text = " ".join(prediction_items)
    acc_matches = sum(1 for answer in answers if match(prediction_text, answer))
    acc = acc_matches / len(answers) if answers else 0
    hit = 1 if any(match(prediction_text, answer) for answer in answers) else 0
    if not prediction_items or not answers:
        return acc, hit, 0, 0, 0
    f1_matches = sum(1 for answer in answers if match(prediction_text, answer))
    precision = f1_matches / len(prediction_items)
    recall = f1_matches / len(answers)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return acc, hit, precision, recall, f1


def safe_json(value):
    return json.dumps(value, ensure_ascii=False)


def read_config(run_dir):
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_from_prediction_file(path, model_name):
    name = path.name
    sample_size = None
    dataset = "UNKNOWN"
    marker = f"-{model_name}-"
    if marker in name:
        dataset, tail = name.split(marker, 1)
        sample_part = tail.replace(".jsonl", "")
        try:
            sample_size = int(sample_part)
        except ValueError:
            sample_size = None
    return dataset, sample_size


def config_key(config, prediction_file, model_name):
    inferred_dataset, inferred_sample_size = infer_from_prediction_file(prediction_file, model_name)
    return (
        config.get("dataset", inferred_dataset),
        config.get("sample_size", inferred_sample_size),
        config.get("seed"),
        config.get("model_name", model_name),
        config.get("top_n"),
        config.get("top_k"),
        config.get("max_length"),
    )


def mean(values):
    return sum(values) / len(values) if values else 0


def sort_best_key(row):
    max_length = row["max_length"] if row["max_length"] is not None else 10**9
    top_k = row["top_k"] if row["top_k"] is not None else 10**9
    top_n = row["top_n"] if row["top_n"] is not None else 10**9
    field_rank = 0 if row["prediction_field"] == "prediction_llm" else 1
    return (-row["hit"], -row["f1"], -row["acc"], max_length, top_k, top_n, field_rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results_grid")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--output_prefix", type=str, default="grid_eval")
    parser.add_argument("--cal_f1", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    args = parser.parse_args()

    results_path = Path(args.results_path)
    model_root = results_path / args.model_name
    if not model_root.exists():
        raise FileNotFoundError(f"Missing model result directory: {model_root}")

    run_summary_rows = []
    per_sample_rows = []
    seen_run_configs = set()

    run_dirs = sorted([p for p in model_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        prediction_files = sorted(run_dir.glob("*.jsonl"))
        prediction_files = [p for p in prediction_files if not p.name.endswith("_error.jsonl") and "detailed_eval" not in p.name]
        if not prediction_files:
            continue
        config = read_config(run_dir)

        for prediction_file in prediction_files:
            key = config_key(config, prediction_file, args.model_name)
            if key in seen_run_configs:
                print(f"WARNING: duplicate run config skipped: {prediction_file}")
                continue
            seen_run_configs.add(key)

            inferred_dataset, inferred_sample_size = infer_from_prediction_file(prediction_file, args.model_name)
            dataset = config.get("dataset", inferred_dataset)
            sample_size = config.get("sample_size", inferred_sample_size)
            seed = config.get("seed")
            model_name = config.get("model_name", args.model_name)
            top_n = config.get("top_n")
            top_k = config.get("top_k")
            max_length = config.get("max_length")

            seen_ids = set()
            llm_metrics = {"acc": [], "hit": [], "f1": []}
            direct_metrics = {"acc": [], "hit": [], "f1": []}
            num_examples = 0

            with prediction_file.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_id = row.get("id")
                    if row_id in seen_ids:
                        print(f"WARNING: duplicate id inside {prediction_file} line {line_number}; keeping first occurrence: {row_id}")
                        continue
                    seen_ids.add(row_id)
                    num_examples += 1
                    answers = row.get("ground_truth", [])

                    for field in ["prediction_llm", "prediction_direct_answer"]:
                        acc, hit, precision, recall, f1 = evaluate_prediction(row.get(field, ""), answers)
                        metric_bucket = llm_metrics if field == "prediction_llm" else direct_metrics
                        metric_bucket["acc"].append(acc)
                        metric_bucket["hit"].append(hit)
                        metric_bucket["f1"].append(f1)
                        per_sample_rows.append({
                            "dataset": dataset,
                            "id": row_id,
                            "question": row.get("question", ""),
                            "ground_truth": safe_json(answers),
                            "sample_size": sample_size,
                            "seed": seed,
                            "model_name": model_name,
                            "top_n": top_n,
                            "top_k": top_k,
                            "max_length": max_length,
                            "prediction_field": field,
                            "prediction": row.get(field, ""),
                            "acc": acc,
                            "hit": hit,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "reasoning_path": safe_json(row.get("reasoning_path", "")),
                            "ground_path": safe_json(row.get("ground_path", "")),
                            "q_entities": safe_json(row.get("q_entities", "")),
                            "hop": row.get("hop", ""),
                            "run_dir": str(run_dir),
                        })

            run_summary_rows.append({
                "dataset": dataset,
                "sample_size": sample_size,
                "seed": seed,
                "model_name": model_name,
                "top_n": top_n,
                "top_k": top_k,
                "max_length": max_length,
                "num_examples": num_examples,
                "mean_acc_llm": mean(llm_metrics["acc"]),
                "mean_hit_llm": mean(llm_metrics["hit"]),
                "mean_f1_llm": mean(llm_metrics["f1"]),
                "mean_acc_direct": mean(direct_metrics["acc"]),
                "mean_hit_direct": mean(direct_metrics["hit"]),
                "mean_f1_direct": mean(direct_metrics["f1"]),
                "prediction_file": str(prediction_file),
                "run_dir": str(run_dir),
            })

    best_rows = []
    grouped = {}
    for row in per_sample_rows:
        grouped.setdefault((row["dataset"], row["id"]), []).append(row)
    for (_dataset, _row_id), candidates in sorted(grouped.items()):
        best = sorted(candidates, key=sort_best_key)[0]
        best_rows.append({
            "dataset": best["dataset"],
            "id": best["id"],
            "question": best["question"],
            "ground_truth": best["ground_truth"],
            "best_prediction_field": best["prediction_field"],
            "best_prediction": best["prediction"],
            "best_top_n": best["top_n"],
            "best_top_k": best["top_k"],
            "best_max_length": best["max_length"],
            "best_acc": best["acc"],
            "best_hit": best["hit"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "best_f1": best["f1"],
            "reasoning_path": best["reasoning_path"],
            "ground_path": best["ground_path"],
            "run_dir": best["run_dir"],
        })

    outputs = [
        (results_path / f"{args.output_prefix}_run_summary.csv", run_summary_rows),
        (results_path / f"{args.output_prefix}_per_sample_all_runs.csv", per_sample_rows),
        (results_path / f"{args.output_prefix}_best_config_per_sample.csv", best_rows),
    ]
    for output_file, rows in outputs:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with output_file.open("w", newline="", encoding="utf-8") as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        print(f"Wrote {output_file} rows={len(rows)}")

    jsonl_path = results_path / f"{args.output_prefix}_best_config_per_sample.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in best_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {jsonl_path} rows={len(best_rows)}")


if __name__ == "__main__":
    main()
