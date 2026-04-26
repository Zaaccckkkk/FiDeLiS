import argparse
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def choose_python_cmd(explicit=None):
    if explicit:
        return explicit
    if platform.system().lower().startswith("win"):
        return "py"
    return "python"


def run_step(command, title):
    print(f"\n=== {title} ===")
    print("COMMAND:", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_smoke_test", action="store_true")
    parser.add_argument("--run_three_sample_smoke", action="store_true")
    parser.add_argument("--python_cmd", type=str, default=None)
    args = parser.parse_args()

    python_cmd = choose_python_cmd(args.python_cmd)

    if not args.skip_smoke_test:
        smoke_command = [python_cmd, "scripts/final_smoke_test.py"]
        if args.run_three_sample_smoke:
            smoke_command.append("--run_three_sample")
        run_step(smoke_command, "Step 1: Smoke Test")

    grid_command = [
        python_cmd,
        "scripts/run_grid.py",
        "--sample_size",
        str(args.sample_size),
        "--seed",
        str(args.seed),
        "--model_name",
        args.model_name,
        "--embedding_model",
        args.embedding_model,
        "--output_path",
        args.output_path,
        "--top_n",
        str(args.top_n),
        "--top_k_values",
        args.top_k_values,
        "--max_length_values",
        args.max_length_values,
        "--datasets",
        args.datasets,
    ]
    if args.resume:
        grid_command.append("--resume")
    if args.dry_run:
        grid_command.append("--dry_run")
    run_step(grid_command, "Step 2: Grid Search")

    check_command = [
        python_cmd,
        "scripts/check_grid_outputs.py",
        "--results_path",
        args.output_path,
        "--model_name",
        args.model_name,
        "--expected_sample_size",
        str(args.sample_size),
        "--strict",
    ]
    run_step(check_command, "Step 3: Check Grid Outputs")

    collect_command = [
        python_cmd,
        "scripts/collect_best_config.py",
        "--results_path",
        args.output_path,
        "--model_name",
        args.model_name,
    ]
    run_step(collect_command, "Step 4: Collect Best Config")

    print("\n=== Final Summary ===")
    print(f"Grid results: {Path(args.output_path).resolve()}")
    print(f"Run summary: {Path(args.output_path).resolve() / 'grid_eval_run_summary.csv'}")
    print(f"Per-sample all runs: {Path(args.output_path).resolve() / 'grid_eval_per_sample_all_runs.csv'}")
    print(f"Best config per sample: {Path(args.output_path).resolve() / 'grid_eval_best_config_per_sample.csv'}")
    print(f"Best config JSONL: {Path(args.output_path).resolve() / 'grid_eval_best_config_per_sample.jsonl'}")


if __name__ == "__main__":
    main()
