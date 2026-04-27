import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main


def load_ids(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line)["id"] for line in f if line.strip()]


def select_ids(seed, sample_size):
    args = SimpleNamespace(
        d="CL-LT-KGQA",
        crlt_path=str(REPO_ROOT / "datasets" / "crlt" / "CR-LT-QA.json"),
        sample=sample_size,
        seed=seed,
    )
    dataset = main.load_local_crlt_dataset(args)
    with tempfile.TemporaryDirectory() as tmpdir:
        selected = main.select_sample(dataset, args, tmpdir)
        selected_ids_path = Path(tmpdir) / "selected_ids.jsonl"
        ids_from_file = load_ids(selected_ids_path)
        ids_from_dataset = [row["id"] for row in selected]
        if ids_from_file != ids_from_dataset:
            raise AssertionError("selected_ids.jsonl does not match the sampled dataset order.")
        return ids_from_file


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args()

    seed_42_first = select_ids(42, args.sample)
    seed_42_second = select_ids(42, args.sample)
    seed_123 = select_ids(123, args.sample)

    print(f"seed=42 run 1: {seed_42_first}")
    print(f"seed=42 run 2: {seed_42_second}")
    print(f"seed=123:     {seed_123}")

    if seed_42_first != seed_42_second:
        raise AssertionError("Same seed did not produce the same selected ids.")
    if seed_42_first == seed_123:
        raise AssertionError("Different seeds produced the same selected ids.")

    print("Sampling reproducibility check passed.")


if __name__ == "__main__":
    main_cli()
