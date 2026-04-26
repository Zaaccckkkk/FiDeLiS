import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODEL_NAME = "gpt-3.5-turbo-0125"
OUTPUT_ROOT = REPO_ROOT / "results_final_test"
REQUIRED_COLUMNS = {
    "id",
    "question",
    "prediction_llm",
    "prediction_direct_answer",
    "ground_truth",
}


def print_section(title):
    print(f"\n=== {title} ===", flush=True)


def load_config_key():
    config_path = REPO_ROOT / "config.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("OPENAI_API_KEY")
    except Exception as exc:
        print(f"WARNING: Could not read config.json: {exc}", flush=True)
        return None


def has_api_key():
    key = os.environ.get("OPENAI_API_KEY") or load_config_key()
    return bool(key and key != "your-api-key-here"), bool(key)


def choose_python_runner():
    py = shutil.which("py")
    if py:
        try:
            subprocess.run([py, "--version"], check=True, capture_output=True, text=True)
            return [py]
        except Exception:
            pass
    return [sys.executable]


def run_subprocess(command, diagnosis):
    print(f"RUN: {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.stdout.strip():
            print(completed.stdout, flush=True)
        if completed.stderr.strip():
            print(completed.stderr, flush=True)
        return completed
    except subprocess.CalledProcessError as exc:
        print("\nFAILED COMMAND:", " ".join(command), flush=True)
        print("DIAGNOSIS:", diagnosis, flush=True)
        print("\n--- stdout ---", flush=True)
        print(exc.stdout or "", flush=True)
        print("\n--- stderr ---", flush=True)
        print(exc.stderr or "", flush=True)
        if "model" in (exc.stderr or "").lower() and MODEL_NAME in (exc.stderr or ""):
            print(
                "Model hint: if gpt-3.5-turbo-0125 is unavailable for your account, retry with --model_name gpt-4o-mini.",
                flush=True,
            )
        raise


def check_environment(runner):
    print_section("Environment Checks")
    print(f"Python: {sys.version}", flush=True)
    for module_name in ["openai", "datasets", "networkx", "numpy", "tqdm"]:
        importlib.import_module(module_name)
        print(f"import {module_name}: OK", flush=True)

    run_subprocess(
        runner + ["main.py", "--help"],
        "main.py help failed, so the entrypoint is not importable/runnable.",
    )

    valid_key, any_key = has_api_key()
    if not valid_key:
        if any_key:
            raise RuntimeError("OPENAI_API_KEY is present but still uses the placeholder value.")
        raise RuntimeError("OPENAI_API_KEY is missing from the environment and config.json.")
    print("OPENAI_API_KEY: present (value hidden)", flush=True)


def print_first_question(prefix, row):
    row_id = row.get("id", "<no id>") if isinstance(row, dict) else "<no id>"
    question = row.get("question") or row.get("query") or "<no question>"
    print(f"{prefix} first id/question: {row_id} / {question}", flush=True)


def check_datasets():
    print_section("Dataset Loading Checks")
    from datasets import load_dataset

    status = {}
    for name in ["RoG-webqsp", "RoG-cwq"]:
        dataset_id = f"rmanluo/{name}"
        ds = load_dataset(dataset_id, split="test", cache_dir=str(REPO_ROOT / "datasets" / "cache"))
        print(f"{dataset_id}: size={len(ds)} columns={ds.column_names}", flush=True)
        if len(ds):
            print_first_question(dataset_id, ds[0])
        status[name] = "OK"

    crlt_path = REPO_ROOT / "datasets" / "crlt" / "CR-LT-QA.json"
    if not crlt_path.exists():
        raise FileNotFoundError(f"Missing local CR-LT file: {crlt_path}")
    with crlt_path.open("r", encoding="utf-8") as f:
        crlt = json.load(f)
    keys = sorted(crlt[0].keys()) if crlt else []
    print(f"CL-LT-KGQA local: size={len(crlt)} keys={keys}", flush=True)
    if crlt:
        print_first_question("CL-LT-KGQA", crlt[0])
    status["CL-LT-KGQA"] = "OK"
    return status


def evaluation_sanity_check():
    print_section("Evaluation Sanity Check")
    from src.evaluate_results import eval_result

    row = {
        "id": "test1",
        "question": "what does jamaican people speak",
        "q_entities": ["Jamaica"],
        "reasoning_path": ["Jamaica -> location.country.languages_spoken -> Jamaican English"],
        "ground_path": [["location.country.languages_spoken"]],
        "prediction_llm": "Jamaican English\nJamaican Creole English Language",
        "prediction_direct_answer": "Jamaican English\nJamaican Creole English Language\nKingston",
        "ground_truth": ["Jamaican English", "Jamaican Creole English Language"],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        prediction_file = Path(tmpdir) / "prediction.jsonl"
        with prediction_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        llm_result, direct_result = eval_result(str(prediction_file), cal_f1=True)

    eval_status = "OK"
    if llm_result.get("Hit") != 100.0 or llm_result.get("Accuracy") != 100.0 or llm_result.get("F1") != 100.0:
        print(f"WARNING: prediction_llm metrics were not the expected 100s: {llm_result}", flush=True)
        eval_status = "WARN"
    if not direct_result:
        print("WARNING: prediction_direct_answer was not evaluated.", flush=True)
        eval_status = "WARN"
    print(f"eval_result prediction_llm: {llm_result}", flush=True)
    print(f"eval_result prediction_direct_answer: {direct_result}", flush=True)
    return eval_status


def newest_output_dir(model_name):
    model_root = OUTPUT_ROOT / model_name
    if not model_root.exists():
        raise FileNotFoundError(f"Missing model output root: {model_root}")
    candidates = [p for p in model_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No output directories found under: {model_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def validate_output(dataset_name, sample_size, model_name):
    output_dir = newest_output_dir(model_name)
    prediction_file = output_dir / f"{dataset_name}-{model_name}-{sample_size}.jsonl"
    if not prediction_file.exists():
        raise FileNotFoundError(f"Missing prediction file: {prediction_file}")
    if prediction_file.stat().st_size == 0:
        raise RuntimeError(f"Prediction file is empty: {prediction_file}")

    rows = []
    with prediction_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {prediction_file} line {line_number}: {exc}") from exc
            missing = REQUIRED_COLUMNS - set(row.keys())
            if missing:
                raise RuntimeError(f"Missing columns in {prediction_file} line {line_number}: {sorted(missing)}")
            rows.append(row)

    ids = [row["id"] for row in rows]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    empty_predictions = [row["id"] for row in rows if not str(row.get("prediction_llm", "")).strip()]
    error_file = prediction_file.with_name(prediction_file.stem + "_error.jsonl")
    error_exists = error_file.exists()
    error_empty = error_exists and error_file.stat().st_size == 0

    print(
        f"{dataset_name}: rows={len(rows)} duplicates={len(duplicate_ids)} "
        f"empty_prediction_llm={len(empty_predictions)} error_file_exists={error_exists} error_file_empty={error_empty}",
        flush=True,
    )
    return {
        "prediction_file": str(prediction_file),
        "num_rows": len(rows),
        "duplicate_ids": len(duplicate_ids),
        "empty_predictions": len(empty_predictions),
        "error_file_exists": error_exists,
        "error_file_empty": error_empty,
    }


def dataset_command(dataset_name, sample_size, top_n, top_k, max_length, runner):
    command = runner + [
        "main.py",
        "--d",
        dataset_name,
    ]
    if dataset_name in {"RoG-webqsp", "RoG-cwq"}:
        command += ["--split", "test"]
    command += [
        "--sample",
        str(sample_size),
        "--model_name",
        MODEL_NAME,
        "--top_n",
        str(top_n),
        "--top_k",
        str(top_k),
        "--max_length",
        str(max_length),
        "--add_hop_information",
        "--debug",
        "--output_path",
        str(OUTPUT_ROOT),
    ]
    return command


def run_inference_suite(sample_size, top_n, top_k, max_length, runner, preprocessing_status, eval_status):
    print_section(f"Inference Smoke Tests sample={sample_size}")
    rows = []
    for dataset_name in ["CL-LT-KGQA", "RoG-webqsp", "RoG-cwq"]:
        command = dataset_command(dataset_name, sample_size, top_n, top_k, max_length, runner)
        row = {
            "dataset": dataset_name,
            "sample_size": sample_size,
            "preprocessing_status": preprocessing_status.get(dataset_name, "UNKNOWN"),
            "inference_status": "FAILED",
            "prediction_file": "",
            "num_rows": 0,
            "duplicate_ids": 0,
            "empty_predictions": 0,
            "eval_status": eval_status,
        }
        try:
            run_subprocess(command, f"{dataset_name} inference crashed.")
            validation = validate_output(dataset_name, sample_size, MODEL_NAME)
            row.update(validation)
            row["inference_status"] = "OK"
        except Exception:
            rows.append(row)
            raise
        rows.append(row)
    return rows


def print_summary(rows):
    print_section("Final Summary")
    headers = [
        "dataset",
        "sample_size",
        "preprocessing_status",
        "inference_status",
        "prediction_file",
        "num_rows",
        "duplicate_ids",
        "empty_predictions",
        "eval_status",
    ]
    print("\t".join(headers), flush=True)
    for row in rows:
        print("\t".join(str(row.get(header, "")) for header in headers), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_three_sample", action="store_true")
    args = parser.parse_args()

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    runner = choose_python_runner()
    check_environment(runner)
    preprocessing_status = check_datasets()
    eval_status = evaluation_sanity_check()

    all_rows = run_inference_suite(
        sample_size=1,
        top_n=5,
        top_k=1,
        max_length=1,
        runner=runner,
        preprocessing_status=preprocessing_status,
        eval_status=eval_status,
    )
    if args.run_three_sample:
        all_rows.extend(
            run_inference_suite(
                sample_size=3,
                top_n=10,
                top_k=2,
                max_length=2,
                runner=runner,
                preprocessing_status=preprocessing_status,
                eval_status=eval_status,
            )
        )

    print_summary(all_rows)


if __name__ == "__main__":
    main()
