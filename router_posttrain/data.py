from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
   from datasets import load_from_disk
except Exception:  # pragma: no cover - optional at import time
   load_from_disk = None


WH_WORDS = ("what", "which", "who", "where", "when", "why", "how")


@dataclass
class RouterSample:
   id: str
   question: str
   q_entity: List[str]
   a_entity: List[str]
   graph: list
   ground_paths: list
   hop: int = 0
   dataset_name: Optional[str] = None


def _normalize_sample(item: Dict, dataset_name: Optional[str]) -> RouterSample:
   return RouterSample(
      id=str(item["id"]),
      question=item["question"],
      q_entity=list(item.get("q_entity", [])),
      a_entity=list(item.get("a_entity", [])),
      graph=item.get("graph", []),
      ground_paths=item.get("ground_paths", item.get("ground_path", [])),
      hop=int(item.get("hop", 0)),
      dataset_name=dataset_name,
   )


def load_router_samples(path: str, dataset_name: Optional[str] = None, limit: int = -1) -> List[RouterSample]:
   source = Path(path)
   samples: List[RouterSample] = []

   if source.is_dir():
      if load_from_disk is None:
         raise RuntimeError("datasets.load_from_disk is unavailable but a dataset directory was provided.")
      dataset = load_from_disk(str(source))
      iterable = dataset if limit < 0 else dataset.select(range(min(limit, len(dataset))))
      for item in iterable:
         samples.append(_normalize_sample(item, dataset_name))
      return samples

   if source.suffix == ".jsonl":
      with source.open("r") as handle:
         for line in handle:
            samples.append(_normalize_sample(json.loads(line), dataset_name))
            if 0 <= limit == len(samples):
               break
      return samples

   if source.suffix == ".json":
      with source.open("r") as handle:
         raw = json.load(handle)
      for item in raw[:None if limit < 0 else limit]:
         samples.append(_normalize_sample(item, dataset_name))
      return samples

   raise ValueError(f"Unsupported dataset format: {source}")


def build_scalar_features(sample: RouterSample) -> torch.Tensor:
   question = sample.question.strip().lower()
   tokens = question.split()
   is_yes_no = float(question.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "could ", "was ", "were ")))
   wh_flags = [float(question.startswith(f"{wh} ")) for wh in WH_WORDS]
   features = [
      float(len(tokens)),
      float(len(sample.q_entity)),
      float(sample.hop),
      is_yes_no,
      float("compare" in question or "before" in question or "after" in question),
      float("most" in question or "least" in question or "largest" in question or "smallest" in question),
      *wh_flags,
   ]
   return torch.tensor(features, dtype=torch.float32)

