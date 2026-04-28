import argparse
from pathlib import Path

from analyze_router_labels import load_jsonl, print_short_summary, summarize


def print_one(path):
    records = load_jsonl(path)
    summary = summarize(records)
    print("=" * 80)
    print_short_summary(path, summary)
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze every router label JSONL file.")
    parser.add_argument("--label_dir", default="router_labels")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    files = sorted(label_dir.glob("*_train_router_labels.jsonl"))
    if not files:
        print(f"No router label files found in {label_dir}")
        return

    for path in files:
        print_one(path)


if __name__ == "__main__":
    main()
