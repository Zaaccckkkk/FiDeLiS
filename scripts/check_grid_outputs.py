import argparse
import csv
import json
from pathlib import Path


REQUIRED_FIELDS = {"id", "question", "prediction_llm", "prediction_direct_answer", "ground_truth"}


def validate_prediction_file(path, expected_sample_size=None):
    result = {
        "prediction_file": str(path),
        "num_rows": 0,
        "duplicate_ids": 0,
        "empty_llm": 0,
        "empty_direct": 0,
        "invalid_json": 0,
        "missing_fields": 0,
        "status": "OK",
    }
    if not path.exists():
        result["status"] = "MISSING"
        return result
    if path.stat().st_size == 0:
        result["status"] = "EMPTY"
        return result

    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                result["invalid_json"] += 1
                continue
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                result["missing_fields"] += 1
            row_id = row.get("id")
            if row_id in seen:
                result["duplicate_ids"] += 1
            seen.add(row_id)
            if not str(row.get("prediction_llm", "")).strip():
                result["empty_llm"] += 1
            if not str(row.get("prediction_direct_answer", "")).strip():
                result["empty_direct"] += 1
            result["num_rows"] += 1

    failures = []
    if result["num_rows"] == 0:
        failures.append("ZERO_ROWS")
    if result["invalid_json"]:
        failures.append("INVALID_JSON")
    if result["missing_fields"]:
        failures.append("MISSING_FIELDS")
    if result["duplicate_ids"]:
        failures.append("DUPLICATE_IDS")
    if expected_sample_size is not None and result["num_rows"] < expected_sample_size:
        failures.append("TOO_FEW_ROWS")
    result["status"] = "OK" if not failures else "|".join(failures)
    return result


def count_error_lines(prediction_file):
    error_file = prediction_file.with_name(prediction_file.stem + "_error.jsonl")
    if not error_file.exists():
        return 0, False, []
    lines = [line.rstrip("\n") for line in error_file.open("r", encoding="utf-8") if line.strip()]
    return len(lines), True, lines[:3]


def is_prediction_jsonl(path):
    if path.name == "selected_ids.jsonl":
        return False
    if path.name.endswith("_error.jsonl"):
        return False
    if "detailed_eval" in path.name:
        return False
    return path.suffix == ".jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results_grid")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--expected_sample_size", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results_path = Path(args.results_path)
    model_root = results_path / args.model_name
    rows = []
    print("dataset\trun_dir\tnum_rows\tduplicate_ids\tempty_llm\tempty_direct\terror_lines\tstatus")

    if not model_root.exists():
        print(f"NO_RUNS\t{model_root}\t0\t0\t0\t0\t0\tMISSING_MODEL_ROOT")
        if args.strict:
            raise SystemExit(1)
        return

    for run_dir in sorted([p for p in model_root.iterdir() if p.is_dir()]):
        prediction_files = sorted(run_dir.glob("*.jsonl"))
        prediction_files = [p for p in prediction_files if is_prediction_jsonl(p)]
        if not prediction_files:
            row = {
                "dataset": "UNKNOWN",
                "run_dir": str(run_dir),
                "prediction_file": "",
                "num_rows": 0,
                "duplicate_ids": 0,
                "empty_llm": 0,
                "empty_direct": 0,
                "error_lines": 0,
                "status": "MISSING_PREDICTION",
            }
            rows.append(row)
            print("\t".join(str(row[k]) for k in ["dataset", "run_dir", "num_rows", "duplicate_ids", "empty_llm", "empty_direct", "error_lines", "status"]))
            continue

        for prediction_file in prediction_files:
            dataset = prediction_file.name.split(f"-{args.model_name}-")[0]
            validation = validate_prediction_file(prediction_file, args.expected_sample_size)
            error_lines, error_exists, first_errors = count_error_lines(prediction_file)
            row = {
                "dataset": dataset,
                "run_dir": str(run_dir),
                "prediction_file": str(prediction_file),
                "num_rows": validation["num_rows"],
                "duplicate_ids": validation["duplicate_ids"],
                "empty_llm": validation["empty_llm"],
                "empty_direct": validation["empty_direct"],
                "error_lines": error_lines,
                "error_file_exists": error_exists,
                "status": validation["status"],
            }
            rows.append(row)
            print("\t".join(str(row[k]) for k in ["dataset", "run_dir", "num_rows", "duplicate_ids", "empty_llm", "empty_direct", "error_lines", "status"]))
            for error in first_errors:
                print(f"  first_error: {error[:500]}")

    csv_path = results_path / "grid_output_check.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "run_dir", "prediction_file", "num_rows", "duplicate_ids", "empty_llm",
            "empty_direct", "error_lines", "error_file_exists", "status"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    if args.strict:
        bad = [row for row in rows if row["status"] != "OK"]
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
