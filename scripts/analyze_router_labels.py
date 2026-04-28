import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return records


def pct(part, total):
    if total == 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def label_key(label):
    if not label:
        return "unresolved"
    return f"max_length={label['max_length']}, top_k={label['top_k']}, cost={label['cost']}"


def best_attempt(record):
    attempts = record.get("attempts") or []
    if not attempts:
        return None
    return max(
        attempts,
        key=lambda attempt: attempt.get("metrics", {})
        .get("prediction_llm", {})
        .get("f1", -1),
    )


def attempt_metric(attempt, metric):
    if not attempt:
        return None
    return attempt.get("metrics", {}).get("prediction_llm", {}).get(metric)


def summarize(records):
    total = len(records)
    status_counts = Counter(record.get("status", "missing") for record in records)
    label_counts = Counter(label_key(record.get("label")) for record in records)
    hop_counts = Counter(record.get("hop") for record in records)
    attempt_counts = Counter(len(record.get("attempts") or []) for record in records)

    resolved = [record for record in records if record.get("status") == "resolved"]
    weak_resolved = [record for record in records if record.get("status") == "weak_resolved"]
    unresolved = [record for record in records if record.get("status") == "unresolved"]

    best_f1_values = []
    best_configs = Counter()
    boolean_pairs = Counter()
    for record in records:
        best = best_attempt(record)
        if best:
            f1 = attempt_metric(best, "f1")
            if f1 is not None:
                best_f1_values.append(f1)
            config = best.get("config")
            if config:
                best_configs[label_key(config)] += 1
            metrics = best.get("metrics", {}).get("prediction_llm", {})
            pred_bool = metrics.get("prediction_boolean")
            gold_bool = metrics.get("ground_truth_boolean")
            if pred_bool is not None or gold_bool is not None:
                boolean_pairs[(str(gold_bool), str(pred_bool))] += 1

    avg_attempts = (
        sum(len(record.get("attempts") or []) for record in records) / total
        if total
        else 0
    )
    avg_best_f1 = sum(best_f1_values) / len(best_f1_values) if best_f1_values else 0

    return {
        "total": total,
        "status_counts": dict(status_counts),
        "label_counts": dict(label_counts),
        "hop_counts": dict(sorted(hop_counts.items(), key=lambda item: str(item[0]))),
        "attempt_counts": dict(sorted(attempt_counts.items())),
        "resolved_count": len(resolved),
        "weak_resolved_count": len(weak_resolved),
        "unresolved_count": len(unresolved),
        "resolved_rate": len(resolved) / total if total else 0,
        "avg_attempts": avg_attempts,
        "avg_best_f1": avg_best_f1,
        "best_config_counts": dict(best_configs),
        "boolean_pairs": {f"gold={k[0]}, pred={k[1]}": v for k, v in boolean_pairs.items()},
    }


def print_counter(title, counter, total=None, limit=None):
    print(f"\n{title}")
    items = counter.most_common(limit)
    if not items:
        print("  (none)")
        return
    denominator = total if total is not None else sum(counter.values())
    for key, value in items:
        print(f"  {key}: {value} ({pct(value, denominator)})")


def print_short_summary(input_path, summary):
    total = summary["total"]
    print(f"File: {input_path}")
    print(f"Total records: {total}")
    print(
        "Resolved: "
        f"{summary['resolved_count']} ({pct(summary['resolved_count'], total)})"
    )
    print(
        "Unresolved: "
        f"{summary['unresolved_count']} ({pct(summary['unresolved_count'], total)})"
    )
    if summary.get("weak_resolved_count"):
        print(
            "Weak resolved: "
            f"{summary['weak_resolved_count']} ({pct(summary['weak_resolved_count'], total)})"
        )
    print(f"Average attempts per sample: {summary['avg_attempts']:.2f}")
    print(f"Average best F1: {summary['avg_best_f1']:.3f}")

    print_counter("Status Distribution", Counter(summary["status_counts"]), total)
    print_counter("Chosen Label Distribution", Counter(summary["label_counts"]), total)


def print_examples(title, records, limit):
    print(f"\n{title}")
    if not records:
        print("  (none)")
        return
    for record in records[:limit]:
        best = best_attempt(record)
        best_config = best.get("config") if best else None
        best_f1 = attempt_metric(best, "f1")
        prediction = best.get("prediction_llm") if best else None
        print(f"  - id: {record.get('id')}")
        print(f"    question: {record.get('question')}")
        print(f"    gold: {record.get('ground_truth')}")
        print(f"    status: {record.get('status')}, label: {record.get('label')}")
        print(f"    best_config: {best_config}, best_f1: {best_f1}")
        print(f"    best_prediction: {prediction}")


def main():
    parser = argparse.ArgumentParser(description="Analyze router label JSONL files.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="router_labels/CL-LT-KGQA_train_router_labels.jsonl",
    )
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed distributions and examples.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    records = load_jsonl(input_path)
    summary = summarize(records)

    total = summary["total"]
    print_short_summary(input_path, summary)

    if args.verbose:
        print_counter("Best Config Distribution", Counter(summary["best_config_counts"]), total, limit=20)
        print_counter("Hop Distribution", Counter(summary["hop_counts"]), total)
        print_counter("Attempt Count Distribution", Counter(summary["attempt_counts"]), total)
        print_counter("Boolean Gold/Prediction Pairs", Counter(summary["boolean_pairs"]), total)

        unresolved = [record for record in records if record.get("status") != "resolved"]
        resolved = [record for record in records if record.get("status") == "resolved"]
        unresolved_sorted = sorted(
            unresolved,
            key=lambda record: attempt_metric(best_attempt(record), "f1") or 0,
            reverse=True,
        )
        resolved_sorted = sorted(
            resolved,
            key=lambda record: (record.get("label") or {}).get("cost", 0),
        )

        print_examples("Resolved Examples", resolved_sorted, args.examples)
        print_examples("Top Unresolved Examples By Best F1", unresolved_sorted, args.examples)


if __name__ == "__main__":
    main()
