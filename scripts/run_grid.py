import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = {"id", "question", "prediction_llm", "prediction_direct_answer", "ground_truth"}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_csv_ints(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def choose_python_cmd(explicit=None):
    if explicit:
        return explicit
    if platform.system().lower().startswith("win"):
        return "py"
    return "python"


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_config_matches(config, dataset, sample_size, seed, model_name, top_n, top_k, max_length):
    return (
        config.get("dataset") == dataset
        and int(config.get("sample_size", -1)) == sample_size
        and int(config.get("seed", -999999)) == seed
        and config.get("model_name") == model_name
        and int(config.get("top_n", -1)) == top_n
        and int(config.get("top_k", -1)) == top_k
        and int(config.get("max_length", -1)) == max_length
    )


def validate_prediction_file(path, expected_sample_size=None):
    if not path.exists() or path.stat().st_size == 0:
        return False, {"num_rows": 0, "duplicate_ids": 0, "error": "missing_or_empty"}

    seen = set()
    duplicate_ids = 0
    num_rows = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = REQUIRED_FIELDS - set(row)
                if missing:
                    return False, {"num_rows": num_rows, "duplicate_ids": duplicate_ids, "error": f"missing_fields:{sorted(missing)}"}
                row_id = row["id"]
                if row_id in seen:
                    duplicate_ids += 1
                seen.add(row_id)
                num_rows += 1
    except Exception as exc:
        return False, {"num_rows": num_rows, "duplicate_ids": duplicate_ids, "error": f"invalid_json:{exc}"}

    if duplicate_ids:
        return False, {"num_rows": num_rows, "duplicate_ids": duplicate_ids, "error": "duplicate_ids"}
    if expected_sample_size is not None and num_rows < expected_sample_size:
        return False, {"num_rows": num_rows, "duplicate_ids": duplicate_ids, "error": "too_few_rows"}
    return True, {"num_rows": num_rows, "duplicate_ids": duplicate_ids, "error": ""}


def find_completed_run(output_path, model_name, dataset, sample_size, seed, top_n, top_k, max_length):
    model_root = Path(output_path) / model_name
    if not model_root.exists():
        return None
    for run_dir in sorted([p for p in model_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not run_config_matches(config, dataset, sample_size, seed, model_name, top_n, top_k, max_length):
            continue
        prediction_file = run_dir / f"{dataset}-{model_name}-{sample_size}.jsonl"
        valid, _ = validate_prediction_file(prediction_file, expected_sample_size=sample_size)
        if valid:
            return prediction_file
    return None


def newest_run_dir(output_path, model_name, started_before):
    model_root = Path(output_path) / model_name
    if not model_root.exists():
        return None
    candidates = [p for p in model_root.iterdir() if p.is_dir() and p.stat().st_mtime >= started_before]
    if not candidates:
        candidates = [p for p in model_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_command(args, dataset, top_k, max_length, python_cmd):
    command = [
        python_cmd,
        "main.py",
        "--d",
        dataset,
        "--sample",
        str(args.sample_size),
        "--seed",
        str(args.seed),
        "--model_name",
        args.model_name,
        "--embedding_model",
        args.embedding_model,
        "--top_n",
        str(args.top_n),
        "--top_k",
        str(top_k),
        "--max_length",
        str(max_length),
        "--output_path",
        args.output_path,
    ]
    if dataset in {"RoG-webqsp", "RoG-cwq"}:
        command += ["--split", args.split]
    if args.add_hop_information:
        command.append("--add_hop_information")
    if args.debug:
        command.append("--debug")
    return command


def write_run_config(run_dir, args, dataset, top_k, max_length, command):
    config = {
        "dataset": dataset,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "model_name": args.model_name,
        "embedding_model": args.embedding_model,
        "top_n": args.top_n,
        "top_k": top_k,
        "max_length": max_length,
        "split": args.split if dataset in {"RoG-webqsp", "RoG-cwq"} else None,
        "command": command,
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    parser.add_argument("--output_path", type=str, default="results_grid")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--top_k_values", type=str, default="1,2,3")
    parser.add_argument("--max_length_values", type=str, default="1,2,3")
    parser.add_argument("--datasets", type=str, default="CL-LT-KGQA,RoG-webqsp,RoG-cwq")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--add_hop_information", type=parse_bool, default=True)
    parser.add_argument("--debug", type=parse_bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python_cmd", type=str, default=None)
    parser.add_argument("--sleep_seconds", type=float, default=0)
    parser.add_argument("--max_retries", type=int, default=2)
    args = parser.parse_args()

    python_cmd = choose_python_cmd(args.python_cmd)
    output_path = Path(args.output_path)
    manifest_path = output_path / "grid_manifest.jsonl"
    log_dir = output_path / "logs"
    output_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    datasets = parse_csv_strings(args.datasets)
    top_k_values = parse_csv_ints(args.top_k_values)
    max_length_values = parse_csv_ints(args.max_length_values)
    failed_runs = 0

    for dataset in datasets:
        for top_k in top_k_values:
            for max_length in max_length_values:
                command = build_command(args, dataset, top_k, max_length, python_cmd)
                command_str = " ".join(command)
                print(f"\nCOMMAND: {command_str}", flush=True)

                row = {
                    "dataset": dataset,
                    "sample_size": args.sample_size,
                    "seed": args.seed,
                    "model_name": args.model_name,
                    "top_n": args.top_n,
                    "top_k": top_k,
                    "max_length": max_length,
                    "command": command,
                    "status": "PENDING",
                    "start_time": now_iso(),
                    "end_time": None,
                    "return_code": None,
                }

                if args.resume:
                    completed = find_completed_run(
                        args.output_path, args.model_name, dataset, args.sample_size, args.seed, args.top_n, top_k, max_length
                    )
                    if completed:
                        row["status"] = "SKIPPED_RESUME"
                        row["end_time"] = now_iso()
                        row["return_code"] = 0
                        row["prediction_file"] = str(completed)
                        append_jsonl(manifest_path, row)
                        print(f"SKIP resume: valid prediction file exists at {completed}", flush=True)
                        continue

                if args.dry_run:
                    row["status"] = "DRY_RUN"
                    row["end_time"] = now_iso()
                    row["return_code"] = 0
                    append_jsonl(manifest_path, row)
                    continue

                final_completed = None
                for attempt in range(args.max_retries + 1):
                    print(f"Attempt {attempt + 1}/{args.max_retries + 1}", flush=True)
                    started_before = time.time()
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    row["return_code"] = completed.returncode
                    if completed.stdout:
                        print(completed.stdout, flush=True)
                    if completed.stderr:
                        print(completed.stderr, flush=True)

                    if completed.returncode == 0:
                        run_dir = newest_run_dir(args.output_path, args.model_name, started_before)
                        if run_dir:
                            write_run_config(run_dir, args, dataset, top_k, max_length, command)
                            prediction_file = run_dir / f"{dataset}-{args.model_name}-{args.sample_size}.jsonl"
                            valid, validation = validate_prediction_file(prediction_file, expected_sample_size=args.sample_size)
                            if valid:
                                final_completed = prediction_file
                                row["status"] = "OK"
                                row["prediction_file"] = str(prediction_file)
                                break
                            row["status"] = f"FAILED_VALIDATION:{validation.get('error')}"
                        else:
                            row["status"] = "FAILED_VALIDATION:no_run_dir"
                    else:
                        row["status"] = "FAILED_RUNTIME"

                    log_path = log_dir / f"{dataset}_topk{top_k}_depth{max_length}.log"
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"\n=== {now_iso()} attempt {attempt + 1} command ===\n{command_str}\n")
                        f.write(f"\n--- stdout ---\n{completed.stdout or ''}\n")
                        f.write(f"\n--- stderr ---\n{completed.stderr or ''}\n")
                        f.write(f"\nstatus={row['status']} return_code={completed.returncode}\n")

                    if attempt < args.max_retries:
                        time.sleep(max(args.sleep_seconds, 0))

                row["end_time"] = now_iso()
                append_jsonl(manifest_path, row)
                if final_completed is None:
                    failed_runs += 1
                    print(f"RUN FAILED: {command_str}", flush=True)
                if args.sleep_seconds:
                    time.sleep(args.sleep_seconds)

    if failed_runs:
        raise SystemExit(f"{failed_runs} grid run(s) failed. See {log_dir} for captured stdout/stderr.")


if __name__ == "__main__":
    main()
