import argparse
import copy
import datetime
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

from main import prepare_dataset  # noqa: E402
from src.evaluate_results import eval_acc, eval_f1, eval_hit, normalize  # noqa: E402
from src.llm_navigator import LLM_Navigator  # noqa: E402


MAX_LENGTHS = [1, 2, 3, 4]
TOP_KS = [1, 2, 3, 5]
BOOLEAN_ANSWERS = {"true", "false"}


def config_grid():
    configs = [
        {"max_length": max_length, "top_k": top_k, "cost": max_length * top_k}
        for max_length in MAX_LENGTHS
        for top_k in TOP_KS
    ]
    return sorted(configs, key=lambda item: (item["cost"], item["max_length"], item["top_k"]))


def prepare_crlt_sample(sample):
    ground_paths = []
    for step in sample["reasoning_steps"]:
        facts = step.get("facts used in this step")
        if facts is None:
            continue
        if isinstance(facts, list):
            ground_paths.extend(facts)
        else:
            ground_paths.append(facts)

    sample["ground_paths"] = ground_paths
    return sample


def load_seen_ids(output_file, rerun_unresolved=False):
    seen = set()
    if not os.path.exists(output_file):
        return seen

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rerun_unresolved and item.get("status") == "unresolved":
                continue
            sample_id = item.get("id")
            if sample_id is not None:
                seen.add(sample_id)
    return seen


def load_train_dataset(args):
    cache_name = f"{args.benchmark}_train_processed"
    cached_dataset_path = os.path.join(args.save_cache, cache_name)
    if os.path.exists(cached_dataset_path):
        dataset = load_from_disk(cached_dataset_path)
        if args.benchmark != "CL-LT-KGQA":
            return dataset
    else:
        if args.benchmark in ["RoG-webqsp", "RoG-cwq"]:
            if os.path.isdir(args.data_path):
                input_file = os.path.join(args.data_path, args.benchmark)
            else:
                input_file = f"{args.data_path.rstrip('/')}/{args.benchmark}"
            dataset = load_dataset(input_file, split="train", cache_dir=args.save_cache)
            dataset = dataset.map(prepare_dataset, num_proc=args.N_CPUS if args.N_CPUS > 1 else None)
        elif args.benchmark == "CL-LT-KGQA":
            input_file = os.path.join(args.crlt_data_dir, "CR-LT-QA-Wikidata-Cache.jsonl")
            dataset = load_dataset("json", data_files=input_file, split="train", cache_dir=args.save_cache)
            dataset = dataset.map(prepare_crlt_sample, num_proc=args.N_CPUS if args.N_CPUS > 1 else None)
        else:
            raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    dataset = dataset.filter(
        lambda x: x.get("hop") > 0
        and x.get("question") != ""
        and len(x.get("q_entity")) > 0
        and len(x.get("a_entity")) > 0
        and len(x.get("ground_paths")) > 0,
        num_proc=args.N_CPUS if args.N_CPUS > 1 else None,
    )
    dataset = dataset.filter(
        lambda x: x.get("q_entity") is not None,
        num_proc=args.N_CPUS if args.N_CPUS > 1 else None,
    )
    if args.benchmark == "CL-LT-KGQA":
        dataset = dataset.filter(
            lambda x: is_boolean_answer(x.get("a_entity")),
            num_proc=args.N_CPUS if args.N_CPUS > 1 else None,
        )

    if not os.path.exists(cached_dataset_path):
        os.makedirs(cached_dataset_path, exist_ok=True)
        dataset.save_to_disk(cached_dataset_path)
    return dataset


def make_fidelis_args(args, max_length, top_k):
    return SimpleNamespace(
        N_CPUS=args.N_CPUS,
        sample=-1,
        data_path=args.data_path,
        d=args.benchmark,
        save_cache=args.save_cache,
        split="train",
        output_path=args.output_path,
        model_name=args.model_name,
        top_n=args.top_n,
        top_k=top_k,
        max_length=max_length,
        strategy=args.strategy,
        squeeze=True,
        verifier=args.verifier,
        embedding_model=args.embedding_model,
        add_hop_information=args.add_hop_information,
        generate_embeddings=False,
        alpha=args.alpha,
        debug=args.debug,
        openai_timeout=args.openai_timeout,
        openai_max_attempts=args.openai_max_attempts,
    )


def split_predictions(prediction):
    if prediction is None:
        return []
    if isinstance(prediction, list):
        return prediction
    return [item for item in prediction.split("\n") if item.strip()]


def is_boolean_answer(answer):
    if not isinstance(answer, list) or len(answer) != 1:
        return False
    return normalize(str(answer[0])) in BOOLEAN_ANSWERS


def extract_boolean_prediction(prediction):
    values = []
    for item in split_predictions(prediction):
        normalized = normalize(str(item))
        if normalized in BOOLEAN_ANSWERS:
            values.append(normalized)

    unique_values = set(values)
    if len(unique_values) == 1:
        return values[0]
    return None


def score_prediction(prediction, answer):
    prediction_list = split_predictions(prediction)
    f1, precision, recall = eval_f1(prediction_list, answer)
    prediction_str = " ".join(prediction_list)
    return {
        "acc": eval_acc(prediction_str, answer),
        "hit": eval_hit(prediction_str, answer),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def score_boolean_prediction(prediction, answer):
    gold = normalize(str(answer[0])) if is_boolean_answer(answer) else None
    predicted = extract_boolean_prediction(prediction)
    is_correct = gold is not None and predicted == gold
    score = 1.0 if is_correct else 0.0
    return {
        "acc": score,
        "hit": int(is_correct),
        "f1": score,
        "precision": score,
        "recall": score,
        "prediction_boolean": predicted,
        "ground_truth_boolean": gold,
    }


def score_for_benchmark(prediction, answer, benchmark):
    if benchmark == "CL-LT-KGQA" and is_boolean_answer(answer):
        return score_boolean_prediction(prediction, answer)
    return score_prediction(prediction, answer)


def run_one_config(sample, args, config, navigators):
    key = (config["max_length"], config["top_k"])
    if key not in navigators:
        fidelis_args = make_fidelis_args(args, config["max_length"], config["top_k"])
        navigators[key] = LLM_Navigator(fidelis_args)

    started_at = time.time()
    res, _ = navigators[key].beam_search(sample)
    llm_metrics = score_for_benchmark(res.get("prediction_llm", ""), res["ground_truth"], args.benchmark)
    direct_metrics = score_for_benchmark(
        res.get("prediction_direct_answer", ""), res["ground_truth"], args.benchmark
    )

    return {
        "config": copy.deepcopy(config),
        "runtime_sec": round(time.time() - started_at, 3),
        "is_correct": llm_metrics["f1"] == 1.0,
        "prediction_llm": res.get("prediction_llm", ""),
        "prediction_direct_answer": res.get("prediction_direct_answer", ""),
        "reasoning_path": res.get("reasoning_path", []),
        "metrics": {
            "prediction_llm": llm_metrics,
            "prediction_direct_answer": direct_metrics,
        },
    }


def append_jsonl(path, item):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=True) + "\n")
        f.flush()


def read_jsonl(path):
    rows = []
    bad_rows = []
    if not os.path.exists(path):
        return rows, bad_rows
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad_rows.append({"line": line_no, "error": str(exc)})
    return rows, bad_rows


def is_usable_label(item):
    attempts = item.get("attempts") if isinstance(item, dict) else None
    return (
        isinstance(item, dict)
        and item.get("id") is not None
        and item.get("question") is not None
        and "ground_truth" in item
        and isinstance(attempts, list)
        and len(attempts) > 0
        and not any(isinstance(attempt, dict) and attempt.get("error") for attempt in attempts)
        and item.get("status") in {"resolved", "unresolved"}
    )


def unique_usable_labels(path):
    rows, bad_rows = read_jsonl(path)
    usable = []
    seen = set()
    duplicate_ids = []
    for row in rows:
        sample_id = row.get("id") if isinstance(row, dict) else None
        if sample_id in seen:
            duplicate_ids.append(sample_id)
            continue
        if is_usable_label(row):
            usable.append(row)
            seen.add(sample_id)
    return usable, bad_rows, duplicate_ids


def load_skipped_ids(path):
    rows, _ = read_jsonl(path)
    return {row.get("id") for row in rows if isinstance(row, dict) and row.get("id") is not None}


def timestamp():
    return datetime.datetime.now().isoformat(timespec="seconds")


def archive_existing_outputs(args):
    if not args.fresh:
        return None
    archive_dir = os.path.join(args.output_dir, "archive", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    paths = {
        args.output_label_file,
        args.runtime_error_file,
        args.skipped_ids_file,
        args.output_label_file.replace(".jsonl", "_errors.jsonl"),
    }
    moved = []
    for path in paths:
        if os.path.exists(path):
            os.makedirs(archive_dir, exist_ok=True)
            dest = os.path.join(archive_dir, os.path.basename(path))
            shutil.move(path, dest)
            moved.append({"from": path, "to": dest})
    if moved:
        print(f"Archived existing router-label files to {archive_dir}")
    return archive_dir if moved else None


def clean_existing_label_file(args):
    rows, bad_rows = read_jsonl(args.output_label_file)
    if not rows and not bad_rows:
        return

    usable = []
    removed = []
    seen = set()
    for row in rows:
        sample_id = row.get("id") if isinstance(row, dict) else None
        if sample_id in seen:
            removed.append((row, "duplicate_label_id"))
            continue
        if is_usable_label(row):
            usable.append(row)
            seen.add(sample_id)
        else:
            removed.append((row, "existing_runtime_error_or_incomplete_label"))

    if not removed and not bad_rows:
        return

    archive_dir = os.path.join(args.output_dir, "archive", "cleaned_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(archive_dir, exist_ok=True)
    archived_label = os.path.join(archive_dir, os.path.basename(args.output_label_file))
    shutil.copy2(args.output_label_file, archived_label)

    with open(args.output_label_file, "w", encoding="utf-8", newline="\n") as f:
        for row in usable:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    existing_skipped = load_skipped_ids(args.skipped_ids_file)
    for row, reason in removed:
        if not isinstance(row, dict):
            continue
        sample_id = row.get("id")
        if sample_id is None:
            continue
        attempts = row.get("attempts") if isinstance(row.get("attempts"), list) else []
        first_error = next((attempt for attempt in attempts if isinstance(attempt, dict) and attempt.get("error")), {})
        error_text = first_error.get("error") or reason
        error_item = {
            "benchmark": row.get("benchmark", args.benchmark),
            "split": row.get("split", "train"),
            "id": sample_id,
            "question": row.get("question"),
            "config": first_error.get("config"),
            "error_type": "ExistingRuntimeErrorLabel",
            "error": error_text,
            "traceback": None,
            "timestamp": timestamp(),
            "output_paths": {
                "output_label_file": args.output_label_file,
                "runtime_error_file": args.runtime_error_file,
                "skipped_ids_file": args.skipped_ids_file,
            },
            "successful_attempts_before_failure": len([attempt for attempt in attempts if isinstance(attempt, dict) and not attempt.get("error")]),
            "partial_attempts": attempts,
            "source": "clean_existing_label_file",
            "archived_label_file": archived_label,
        }
        append_jsonl(args.runtime_error_file, error_item)
        if sample_id not in existing_skipped:
            append_jsonl(args.skipped_ids_file, {
                "id": sample_id,
                "question": row.get("question"),
                "reason": "runtime_error",
                "error_type": "ExistingRuntimeErrorLabel",
                "error": error_text,
                "timestamp": timestamp(),
            })
            existing_skipped.add(sample_id)

    print(f"Cleaned existing label file: kept {len(usable)} usable rows, removed {len(removed)} non-usable rows.")
    print(f"Archived pre-clean label file to {archived_label}")


def set_openai_key_from_config():
    if os.environ.get("OPENAI_API_KEY"):
        return
    for path in ["config.json", os.path.join("..", "FiDeLiS-master", "config.json")]:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        api_key = config.get("OPENAI_API_KEY")
        if api_key and api_key != "your-api-key-here":
            os.environ["OPENAI_API_KEY"] = api_key
            return


def validate_outputs(args):
    labels, label_bad_rows, label_duplicate_ids = unique_usable_labels(args.output_label_file)
    skipped_rows, skipped_bad_rows = read_jsonl(args.skipped_ids_file)
    skipped_ids = [row.get("id") for row in skipped_rows if isinstance(row, dict) and row.get("id") is not None]
    skipped_duplicate_ids = [sample_id for sample_id, count in __import__("collections").Counter(skipped_ids).items() if count > 1]
    resolved_count = sum(1 for row in labels if row.get("status") == "resolved")
    unresolved_count = sum(1 for row in labels if row.get("status") == "unresolved")
    summary = {
        "output_label_file": args.output_label_file,
        "runtime_error_file": args.runtime_error_file,
        "skipped_ids_file": args.skipped_ids_file,
        "target_count": args.target_count,
        "usable_label_rows": len(labels),
        "resolved_count": resolved_count,
        "unresolved_count": unresolved_count,
        "skipped_runtime_error_samples": len(set(skipped_ids)),
        "duplicate_ids_in_label_file": label_duplicate_ids,
        "duplicate_ids_in_skipped_file": skipped_duplicate_ids,
        "bad_json_rows": {
            "label_file": label_bad_rows,
            "skipped_ids_file": skipped_bad_rows,
        },
        "target_count_reached": len(labels) >= args.target_count,
        "timestamp": timestamp(),
    }
    summary_path = os.path.join(args.output_dir, "label_generation_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print("Validation summary")
    print(f"  usable label rows: {summary['usable_label_rows']}")
    print(f"  resolved count: {resolved_count}")
    print(f"  unresolved count: {unresolved_count}")
    print(f"  skipped runtime-error samples: {summary['skipped_runtime_error_samples']}")
    print(f"  duplicate ids in label file: {len(label_duplicate_ids)}")
    print(f"  duplicate ids in skipped file: {len(skipped_duplicate_ids)}")
    print(f"  bad JSON rows: {len(label_bad_rows) + len(skipped_bad_rows)}")
    print(f"  target_count reached: {summary['target_count_reached']}")
    print(f"  summary: {summary_path}")
    return summary


def label_samples(args):
    set_openai_key_from_config()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_file and args.output_label_file == parser_default_output_label_file():
        args.output_label_file = args.output_file
    archive_existing_outputs(args)
    skip_ids = {item.strip() for item in args.skip_ids.split(",") if item.strip()}
    if args.target_count <= 0:
        print(f"No labels requested. Output file would be {args.output_label_file}")
        return

    if args.resume:
        clean_existing_label_file(args)
    existing_labels, _, _ = unique_usable_labels(args.output_label_file)
    seen_ids = {row["id"] for row in existing_labels} if args.resume else set()
    skipped_ids = load_skipped_ids(args.skipped_ids_file) if args.resume and not args.retry_skipped else set()
    dataset = load_train_dataset(args)
    sample_indices = list(range(len(dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(sample_indices)
    if args.start_index:
        sample_indices = sample_indices[args.start_index:]

    configs = config_grid()
    navigators = {}

    initial_count = len(existing_labels) if args.resume else 0
    usable_count = initial_count
    scanned_count = 0
    pbar = tqdm(total=args.target_count, initial=min(usable_count, args.target_count), desc="Labeling router configs")

    for sample_idx in sample_indices:
        if usable_count >= args.target_count:
            break
        if args.max_candidates is not None and scanned_count >= args.max_candidates:
            break
        sample = dataset[sample_idx]
        sample_id = sample.get("id")
        if sample_id in skip_ids or sample_id in seen_ids or sample_id in skipped_ids:
            continue

        scanned_count += 1
        attempts = []
        label = None
        status = "unresolved"
        runtime_error = None

        for config in configs:
            try:
                attempt = run_one_config(sample, args, config, navigators)
                attempts.append(attempt)
            except Exception as e:
                runtime_error = e
                error_item = {
                    "benchmark": args.benchmark,
                    "split": "train",
                    "id": sample_id,
                    "question": sample.get("question"),
                    "config": {
                        "max_length": config["max_length"],
                        "top_k": config["top_k"],
                        "cost": config["cost"],
                    },
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": timestamp(),
                    "output_paths": {
                        "output_label_file": args.output_label_file,
                        "runtime_error_file": args.runtime_error_file,
                        "skipped_ids_file": args.skipped_ids_file,
                    },
                    "successful_attempts_before_failure": len(attempts),
                    "partial_attempts": attempts,
                }
                append_jsonl(args.runtime_error_file, error_item)
                skipped_item = {
                    "id": sample_id,
                    "question": sample.get("question"),
                    "reason": "runtime_error",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "timestamp": timestamp(),
                }
                append_jsonl(args.skipped_ids_file, skipped_item)
                skipped_ids.add(sample_id)
                if args.stop_on_error:
                    raise
                break

            if attempt["is_correct"]:
                label = {
                    "max_length": config["max_length"],
                    "top_k": config["top_k"],
                    "cost": config["cost"],
                }
                status = "resolved"
                break

        if runtime_error is not None and args.skip_runtime_errors:
            continue

        record = {
            "benchmark": args.benchmark,
            "split": "train",
            "id": sample_id,
            "question": sample.get("question"),
            "q_entities": sample.get("q_entity"),
            "ground_truth": sample.get("a_entity"),
            "hop": sample.get("hop"),
            "status": status,
            "label": label,
            "attempts": attempts,
        }
        append_jsonl(args.output_label_file, record)
        seen_ids.add(sample_id)
        usable_count += 1
        pbar.update(1)

    pbar.close()
    added_count = usable_count - initial_count
    print(f"Saved {added_count} new usable labels to {args.output_label_file}")
    if usable_count < args.target_count:
        print(f"Only reached {usable_count} usable labels before stopping.")
    validate_outputs(args)


def parser_default_output_label_file():
    return os.path.join("router_labels", "RoG-webqsp_train_router_labels.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label router configs by running FiDeLiS on train split with max_length/top_k grid."
    )
    parser.add_argument(
        "--benchmark",
        "-d",
        choices=["RoG-webqsp", "RoG-cwq", "CL-LT-KGQA"],
        default="RoG-webqsp",
    )
    parser.add_argument("--num_samples", "--sample", type=int, default=10)
    parser.add_argument("--target_count", type=int, default=100)
    parser.add_argument("--max_candidates", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument("--crlt_data_dir", type=str, default="datasets/crlt")
    parser.add_argument("--save_cache", type=str, default="cache")
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="router_labels")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--output_label_file", type=str, default=parser_default_output_label_file())
    parser.add_argument("--runtime_error_file", type=str, default=os.path.join("router_labels", "RoG-webqsp_train_router_runtime_errors.jsonl"))
    parser.add_argument("--skipped_ids_file", type=str, default=os.path.join("router_labels", "RoG-webqsp_train_router_skipped_ids.jsonl"))
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--strategy", type=str, default="discrete_rating")
    parser.add_argument("--verifier", type=str, default="deductive+planning")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--add_hop_information", action="store_true")
    parser.add_argument("--N_CPUS", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--openai_timeout", type=float, default=60)
    parser.add_argument("--openai_max_attempts", type=int, default=3)
    parser.add_argument("--rerun_unresolved", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--skip_runtime_errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry_skipped", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip_ids", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.environ.setdefault("WANDB_MODE", "disabled")
    label_samples(parse_args())
