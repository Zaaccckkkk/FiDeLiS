import argparse
import json
from pathlib import Path


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def attempt_f1(attempt):
    return (
        attempt.get("metrics", {})
        .get("prediction_llm", {})
        .get("f1", 0)
        or 0
    )


def select_weak_label(attempts, min_f1, min_margin):
    valid_attempts = [
        attempt
        for attempt in attempts
        if attempt.get("config") and "error" not in attempt
    ]
    if not valid_attempts:
        return None

    ranked = sorted(
        valid_attempts,
        key=lambda attempt: (
            -attempt_f1(attempt),
            attempt["config"]["cost"],
            attempt["config"]["max_length"],
            attempt["config"]["top_k"],
        ),
    )
    best = ranked[0]
    best_score = attempt_f1(best)
    second_score = attempt_f1(ranked[1]) if len(ranked) > 1 else 0
    margin = best_score - second_score

    if best_score < min_f1 or margin < min_margin:
        return None

    config = best["config"]
    return {
        "label": {
            "max_length": config["max_length"],
            "top_k": config["top_k"],
            "cost": config["cost"],
        },
        "best_f1": best_score,
        "second_best_f1": second_score,
        "margin": margin,
        "reason": f"best_f1>={min_f1} and margin>={min_margin}",
    }


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Promote clear best unresolved attempts to weak router labels."
    )
    parser.add_argument("input_file")
    parser.add_argument("--output_file")
    parser.add_argument("--min_f1", type=float, default=0.5)
    parser.add_argument("--min_margin", type=float, default=0.25)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = (
        Path(args.output_file)
        if args.output_file
        else input_path.with_name(f"{input_path.stem}_weak.jsonl")
    )

    records = load_jsonl(input_path)
    promoted = 0
    for record in records:
        if record.get("status") != "unresolved":
            continue
        weak_label = select_weak_label(
            record.get("attempts") or [],
            min_f1=args.min_f1,
            min_margin=args.min_margin,
        )
        if not weak_label:
            continue

        record["status"] = "weak_resolved"
        record["label"] = weak_label["label"]
        record["weak_label"] = weak_label
        promoted += 1

    write_jsonl(output_path, records)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Promoted unresolved samples: {promoted}")


if __name__ == "__main__":
    main()
