from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List


FIELDS = [
   "dataset",
   "sample_id",
   "source_index",
   "task_type",
   "question",
   "ground_truth",
   "prediction_text",
   "predictions",
   "hit",
   "f1",
   "precision",
   "recall",
   "accuracy",
   "status",
   "model",
   "source",
   "created_at",
   "error",
]


def read_jsonl(path: Path) -> Iterable[dict]:
   with path.open("r") as handle:
      for line in handle:
         line = line.strip()
         if line:
            yield json.loads(line)


def clean_cell(value) -> str:
   if value is None:
      return ""
   return str(value).replace("\r", "\\r").replace("\n", "\\n")


def flatten_record(record: dict) -> dict:
   metrics = record.get("metrics", {})
   return {
      "dataset": clean_cell(record.get("dataset", "")),
      "sample_id": clean_cell(record.get("sample_id", "")),
      "source_index": clean_cell(record.get("source_index", "")),
      "task_type": clean_cell(record.get("task_type", "")),
      "question": clean_cell(record.get("question", "")),
      "ground_truth": clean_cell(" | ".join(str(item) for item in record.get("ground_truth", []))),
      "prediction_text": clean_cell(record.get("prediction_text", "")),
      "predictions": clean_cell(" | ".join(str(item) for item in record.get("predictions", []))),
      "hit": metrics.get("hit", ""),
      "f1": metrics.get("f1", ""),
      "precision": metrics.get("precision", ""),
      "recall": metrics.get("recall", ""),
      "accuracy": metrics.get("accuracy", ""),
      "status": clean_cell(record.get("status", "")),
      "model": clean_cell(record.get("model", "")),
      "source": clean_cell(record.get("source", "")),
      "created_at": clean_cell(record.get("created_at", "")),
      "error": clean_cell(record.get("error", "")),
   }


def write_csv(path: Path, rows: List[dict]):
   path.parent.mkdir(parents=True, exist_ok=True)
   with path.open("w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=FIELDS)
      writer.writeheader()
      writer.writerows(rows)


def export_results(input_dir: Path, output_dir: Path):
   all_rows: List[dict] = []
   for result_path in sorted(input_dir.glob("predictions_*.jsonl")):
      dataset = result_path.stem.replace("predictions_", "")
      rows = [flatten_record(record) for record in read_jsonl(result_path)]
      write_csv(output_dir / f"{dataset}_examples.csv", rows)
      all_rows.extend(rows)
   write_csv(output_dir / "all_examples.csv", all_rows)


def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--input_dir", default="results/vanilla_zero_shot/predictions")
   parser.add_argument("--output_dir", default="results/vanilla_zero_shot/tables")
   return parser.parse_args()


def main():
   args = parse_args()
   export_results(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
   main()
