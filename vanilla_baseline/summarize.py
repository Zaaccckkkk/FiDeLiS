from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from vanilla_baseline.data_loading import read_jsonl


def _histogram(values: List[float], bins: int = 10) -> List[dict]:
   if not values:
      return []
   counts = [0 for _ in range(bins)]
   for value in values:
      index = min(int(value * bins), bins - 1)
      counts[index] += 1
   return [
      {
         "bin_start": i / bins,
         "bin_end": (i + 1) / bins,
         "count": count,
      }
      for i, count in enumerate(counts)
   ]


def summarize_records(records: List[dict]) -> Dict:
   successful = [record for record in records if record.get("status") == "ok"]
   summary = {
      "num_records": len(records),
      "num_successful": len(successful),
      "num_failed": len(records) - len(successful),
   }
   for metric in ["hit", "f1", "precision", "recall", "accuracy"]:
      values = [float(record["metrics"][metric]) for record in successful if "metrics" in record]
      summary[metric] = {
         "mean": mean(values) if values else 0.0,
         "histogram": _histogram(values, bins=10),
      }
   return summary


def write_plots(records: List[dict], output_dir: Path, dataset: str):
   try:
      import matplotlib.pyplot as plt
   except Exception:
      return

   successful = [record for record in records if record.get("status") == "ok"]
   output_dir.mkdir(parents=True, exist_ok=True)
   for metric in ["hit", "f1", "precision", "recall"]:
      values = [float(record["metrics"][metric]) for record in successful if "metrics" in record]
      if not values:
         continue
      plt.figure()
      plt.hist(values, bins=10, range=(0.0, 1.0))
      plt.xlabel(metric)
      plt.ylabel("count")
      plt.title(f"{dataset} {metric} distribution")
      plt.tight_layout()
      plt.savefig(output_dir / f"{dataset}_{metric}_hist.png", dpi=180)
      plt.close()


def summarize_file(result_path: Path, summary_path: Path, plot_dir: Path):
   records = read_jsonl(result_path)
   dataset = result_path.stem.replace("predictions_", "")
   summary = summarize_records(records)
   summary_path.parent.mkdir(parents=True, exist_ok=True)
   with summary_path.open("w") as handle:
      json.dump(summary, handle, indent=2)
   write_plots(records, plot_dir, dataset)
   return summary


def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--result_path", required=True)
   parser.add_argument("--summary_path", required=True)
   parser.add_argument("--plot_dir", required=True)
   return parser.parse_args()


def main():
   args = parse_args()
   summarize_file(Path(args.result_path), Path(args.summary_path), Path(args.plot_dir))


if __name__ == "__main__":
   main()

