from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List

from vanilla_baseline.data_loading import (
   VanillaSample,
   append_jsonl,
   load_dataset_by_name,
   load_or_extend_manifest,
   read_jsonl,
)
from vanilla_baseline.metrics import compute_metrics
from vanilla_baseline.openai_runner import OpenAIZeroShotRunner, load_api_key
from vanilla_baseline.summarize import summarize_file


def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--datasets", nargs="+", default=["webqsp", "cwq", "crlt"], choices=["webqsp", "cwq", "crlt"])
   parser.add_argument("--sample_size", type=int, default=300)
   parser.add_argument("--seed", type=int, default=17)
   parser.add_argument("--split", type=str, default="test")
   parser.add_argument("--output_dir", type=str, default="results/vanilla_zero_shot")
   parser.add_argument("--config_path", type=str, default="vanilla_baseline/config.example.json")
   parser.add_argument("--api_key", type=str, default=None)
   parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125")
   parser.add_argument("--temperature", type=float, default=0.0)
   parser.add_argument("--hf_cache_dir", type=str, default=None)
   parser.add_argument("--webqsp_path", type=str, default=None)
   parser.add_argument("--cwq_path", type=str, default=None)
   parser.add_argument("--crlt_dir", type=str, default="datasets/crlt")
   parser.add_argument("--load_limit", type=int, default=-1)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--rerun_failed", action="store_true")
   parser.add_argument("--dry_run", action="store_true")
   return parser.parse_args()


def existing_status_by_id(result_path: Path) -> Dict[str, str]:
   status = {}
   for record in read_jsonl(result_path):
      sample_id = record.get("sample_id")
      if sample_id:
         status[sample_id] = record.get("status", "ok")
   return status


def should_skip(sample: VanillaSample, status_by_id: Dict[str, str], force: bool, rerun_failed: bool) -> bool:
   if force:
      return False
   status = status_by_id.get(sample.sample_id)
   if status is None:
      return False
   if status != "ok" and rerun_failed:
      return False
   return True


def make_record(sample: VanillaSample, model: str, response: dict, status: str, error: str | None = None) -> dict:
   predictions = response.get("predictions", [])
   metrics = compute_metrics(predictions, sample.answers) if status == "ok" else {}
   return {
      "dataset": sample.dataset,
      "sample_id": sample.sample_id,
      "source_index": sample.source_index,
      "question": sample.question,
      "ground_truth": sample.answers,
      "task_type": sample.task_type,
      "source": sample.source,
      "metadata": sample.metadata,
      "model": model,
      "status": status,
      "prediction_text": response.get("prediction_text", ""),
      "predictions": predictions,
      "metrics": metrics,
      "usage": response.get("usage", {}),
      "error": error,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
   }


def run_dataset(args, dataset_name: str, runner: OpenAIZeroShotRunner | None):
   output_dir = Path(args.output_dir)
   manifest_path = output_dir / "manifests" / f"{dataset_name}_seed{args.seed}.jsonl"
   result_path = output_dir / "predictions" / f"predictions_{dataset_name}.jsonl"
   summary_path = output_dir / "summaries" / f"{dataset_name}_summary.json"
   plot_dir = output_dir / "plots"

   universe = load_dataset_by_name(
      dataset_name=dataset_name,
      split=args.split,
      hf_cache_dir=args.hf_cache_dir,
      webqsp_path=args.webqsp_path,
      cwq_path=args.cwq_path,
      crlt_dir=args.crlt_dir,
      load_limit=args.load_limit,
   )
   sample_size = min(args.sample_size, len(universe))
   samples = load_or_extend_manifest(manifest_path, universe, sample_size=sample_size, seed=args.seed)

   if args.dry_run:
      print(json.dumps({"dataset": dataset_name, "manifest": str(manifest_path), "num_samples": len(samples)}, indent=2))
      return

   status_by_id = existing_status_by_id(result_path)
   for sample in samples:
      if should_skip(sample, status_by_id, args.force, args.rerun_failed):
         continue
      try:
         assert runner is not None
         response = runner.run(sample)
         record = make_record(sample, args.model, response, status="ok")
      except Exception as exc:
         record = make_record(sample, args.model, {}, status="error", error=str(exc))
      append_jsonl(result_path, record)
      status_by_id[sample.sample_id] = record["status"]

   summary = summarize_file(result_path, summary_path, plot_dir)
   print(json.dumps({"dataset": dataset_name, "result_path": str(result_path), "summary": summary}, indent=2))


def main():
   args = parse_args()
   runner = None
   if not args.dry_run:
      api_key = load_api_key(args.config_path, args.api_key)
      runner = OpenAIZeroShotRunner(api_key=api_key, model=args.model, temperature=args.temperature)

   for dataset_name in args.datasets:
      run_dataset(args, dataset_name, runner)


if __name__ == "__main__":
   main()

