from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Optional


HF_REQUIRED_COLUMNS = ["id", "question", "query", "a_entity", "answer", "answers", "q_entity", "hop"]


@dataclass
class VanillaSample:
   dataset: str
   sample_id: str
   source_index: int
   question: str
   answers: List[str]
   task_type: str
   source: str
   metadata: Dict

   def to_json(self) -> dict:
      return asdict(self)


def _as_answer_list(value) -> List[str]:
   if isinstance(value, list):
      return [str(item) for item in value]
   if isinstance(value, bool):
      return ["True" if value else "False"]
   if value is None:
      return []
   return [str(value)]


def _load_jsonl(path: Path) -> List[dict]:
   items = []
   with path.open("r") as handle:
      for line in handle:
         line = line.strip()
         if line:
            items.append(json.loads(line))
   return items


def _load_json(path: Path):
   with path.open("r") as handle:
      return json.load(handle)


def _import_hf_datasets():
   repo_root = str(Path(__file__).resolve().parents[1])
   original_path = list(sys.path)
   original_module = sys.modules.pop("datasets", None)
   try:
      sys.path = [entry for entry in sys.path if entry not in ("", repo_root)]
      try:
         return importlib.import_module("datasets")
      except ModuleNotFoundError as exc:
         raise RuntimeError(
            "Hugging Face `datasets` is required for WebQSP/CWQ loading. "
            "Install the project requirements or run `pip install datasets`, "
            "or pass --webqsp_path/--cwq_path to use local normalized files."
         ) from exc
   finally:
      sys.path = original_path
      if original_module is not None and "datasets" not in sys.modules:
         sys.modules["datasets"] = original_module


def _normalize_hf_item(dataset_name: str, item: dict, index: int, source: str) -> VanillaSample:
   question = item.get("question") or item.get("query")
   answers = item.get("a_entity") or item.get("answer") or item.get("answers")
   sample_id = str(item.get("id", f"{dataset_name}:{index}"))
   if question is None:
      raise ValueError(f"Could not find question field in {dataset_name} sample {index}: {item.keys()}")

   return VanillaSample(
      dataset=dataset_name,
      sample_id=sample_id,
      source_index=index,
      question=str(question),
      answers=_as_answer_list(answers),
      task_type="qa",
      source=source,
      metadata={
         "q_entity": item.get("q_entity", []),
         "hop": item.get("hop"),
      },
   )


def _hf_rog_cache_slugs(hf_name: str) -> tuple[str, str, str]:
   owner, repo = hf_name.split("/", 1)
   repo_slug = repo.replace("RoG-", "ro_g-").lower()
   dataset_cache_dir = f"{owner.lower()}___{repo_slug}"
   hub_cache_dir = f"datasets--{owner}--{repo}"
   return repo_slug, dataset_cache_dir, hub_cache_dir


def _read_arrow_rows(path: Path) -> List[dict]:
   try:
      import pyarrow as pa
      import pyarrow.ipc as ipc
   except ModuleNotFoundError as exc:
      raise RuntimeError("Direct cache loading requires `pyarrow`. Run `pip install pyarrow`.") from exc

   with pa.memory_map(str(path), "r") as source:
      try:
         reader = ipc.open_stream(source)
      except pa.ArrowInvalid:
         source.seek(0)
         reader = ipc.open_file(source)
      table = reader.read_all()
      columns = [column for column in HF_REQUIRED_COLUMNS if column in table.column_names]
      return table.select(columns).to_pylist()


def _read_parquet_rows(path: Path) -> List[dict]:
   try:
      import pyarrow.parquet as pq
   except ModuleNotFoundError as exc:
      raise RuntimeError("Direct cache loading requires `pyarrow`. Run `pip install pyarrow`.") from exc

   parquet_file = pq.ParquetFile(path)
   columns = [column for column in HF_REQUIRED_COLUMNS if column in parquet_file.schema_arrow.names]
   return parquet_file.read(columns=columns).to_pylist()


def _local_hf_cache_candidates(hf_name: str, split: str, cache_dir: Optional[str]) -> List[Path]:
   repo_slug, dataset_cache_dir, hub_cache_dir = _hf_rog_cache_slugs(hf_name)
   paths: List[Path] = []

   dataset_roots = []
   if cache_dir:
      dataset_roots.append(Path(cache_dir).expanduser())
   dataset_roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
   for root in dataset_roots:
      paths.extend(root.glob(f"{dataset_cache_dir}/default/*/*/{repo_slug}-{split}-*.arrow"))
      if not paths:
         paths.extend(root.glob(f"{dataset_cache_dir}/**/{repo_slug}-{split}-*.arrow"))

   hub_root = Path.home() / ".cache" / "huggingface" / "hub" / hub_cache_dir
   paths.extend(hub_root.glob(f"snapshots/*/data/{split}-*.parquet"))
   return sorted(set(paths))


def _download_hf_parquet_candidates(hf_name: str, split: str) -> List[Path]:
   try:
      from huggingface_hub import snapshot_download
   except ModuleNotFoundError:
      return []

   snapshot_path = Path(
      snapshot_download(
         repo_id=hf_name,
         repo_type="dataset",
         allow_patterns=f"data/{split}-*.parquet",
      )
   )
   return sorted(snapshot_path.glob(f"data/{split}-*.parquet"))


def _load_hf_rog_from_cached_files(
   dataset_name: str,
   hf_name: str,
   split: str,
   cache_dir: Optional[str],
   limit: int,
) -> List[VanillaSample]:
   files = _local_hf_cache_candidates(hf_name, split, cache_dir)
   if not files:
      files = _download_hf_parquet_candidates(hf_name, split)
   if not files:
      raise RuntimeError(
         f"Could not find cached Arrow/parquet shards for {hf_name} split={split}. "
         "Retry after the Hugging Face download finishes or pass a local --cwq_path/--webqsp_path."
      )

   samples: List[VanillaSample] = []
   for path in files:
      rows = _read_arrow_rows(path) if path.suffix == ".arrow" else _read_parquet_rows(path)
      for row in rows:
         if limit >= 0 and len(samples) >= limit:
            return samples
         samples.append(_normalize_hf_item(dataset_name, row, len(samples), str(path)))
   return samples


def load_hf_rog_dataset(
   dataset_name: str,
   hf_name: str,
   split: str,
   cache_dir: Optional[str],
   limit: int = -1,
) -> List[VanillaSample]:
   if _local_hf_cache_candidates(hf_name, split, cache_dir):
      return _load_hf_rog_from_cached_files(dataset_name, hf_name, split, cache_dir, limit)

   hf_datasets = _import_hf_datasets()
   try:
      dataset = hf_datasets.load_dataset(hf_name, split=split, cache_dir=cache_dir)
   except RuntimeError as exc:
      if "RLock objects should only be shared" not in str(exc):
         raise
      return _load_hf_rog_from_cached_files(dataset_name, hf_name, split, cache_dir, limit)

   if limit >= 0:
      if hasattr(dataset, "select"):
         dataset = dataset.select(range(min(limit, len(dataset))))
      else:
         dataset = dataset.take(limit)
   return [_normalize_hf_item(dataset_name, item, index, hf_name) for index, item in enumerate(dataset)]


def load_local_normalized_dataset(dataset_name: str, path: str, limit: int = -1) -> List[VanillaSample]:
   source = Path(path)
   if source.suffix == ".jsonl":
      raw_items = _load_jsonl(source)
   elif source.suffix == ".json":
      raw_items = _load_json(source)
   else:
      hf_datasets = _import_hf_datasets()
      dataset = hf_datasets.load_from_disk(str(source))
      raw_items = list(dataset)

   if limit >= 0:
      raw_items = raw_items[:limit]

   samples = []
   for index, item in enumerate(raw_items):
      samples.append(_normalize_hf_item(dataset_name, item, index, str(source)))
   return samples


def load_local_crlt(crlt_dir: str, limit: int = -1) -> List[VanillaSample]:
   root = Path(crlt_dir)
   qa_path = root / "CR-LT-QA.json"
   cv_path = root / "CR-LT-ClaimVerification.json"
   samples: List[VanillaSample] = []

   for task_name, path in [("qa", qa_path), ("claim_verification", cv_path)]:
      raw_items = _load_json(path)
      for index, item in enumerate(raw_items):
         answer = item.get("answer")
         if isinstance(answer, bool):
            answers = ["True" if answer else "False"]
         else:
            answers = _as_answer_list(answer)
         samples.append(
            VanillaSample(
               dataset="crlt",
               sample_id=f"crlt-{task_name}-{item.get('id', index)}",
               source_index=len(samples),
               question=str(item["query"]),
               answers=answers,
               task_type=task_name,
               source=str(path),
               metadata={
                  "raw_id": item.get("id"),
                  "reasoning_strategy": item.get("Reasoning Strategy", []),
                  "kg_entities": item.get("KG Entities", {}),
               },
            )
         )

   if limit >= 0:
      samples = samples[:limit]
   return samples


def load_dataset_by_name(
   dataset_name: str,
   split: str,
   hf_cache_dir: Optional[str],
   webqsp_path: Optional[str] = None,
   cwq_path: Optional[str] = None,
   crlt_dir: str = "datasets/crlt",
   load_limit: int = -1,
) -> List[VanillaSample]:
   if dataset_name == "webqsp":
      if webqsp_path:
         return load_local_normalized_dataset("webqsp", webqsp_path, load_limit)
      return load_hf_rog_dataset("webqsp", "rmanluo/RoG-webqsp", split, hf_cache_dir, load_limit)
   if dataset_name == "cwq":
      if cwq_path:
         return load_local_normalized_dataset("cwq", cwq_path, load_limit)
      return load_hf_rog_dataset("cwq", "rmanluo/RoG-cwq", split, hf_cache_dir, load_limit)
   if dataset_name == "crlt":
      return load_local_crlt(crlt_dir, load_limit)
   raise ValueError(f"Unknown dataset: {dataset_name}")


def read_jsonl(path: Path) -> List[dict]:
   if not path.exists():
      return []
   return _load_jsonl(path)


def write_jsonl(path: Path, items: Iterable[dict]):
   path.parent.mkdir(parents=True, exist_ok=True)
   with path.open("w") as handle:
      for item in items:
         handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, item: dict):
   path.parent.mkdir(parents=True, exist_ok=True)
   with path.open("a") as handle:
      handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_or_extend_manifest(
   manifest_path: Path,
   universe: List[VanillaSample],
   sample_size: int,
   seed: int,
) -> List[VanillaSample]:
   by_id = {sample.sample_id: sample for sample in universe}
   existing = read_jsonl(manifest_path)
   chosen_ids = []
   for item in existing:
      sample_id = item["sample_id"]
      if sample_id in by_id and sample_id not in chosen_ids:
         chosen_ids.append(sample_id)

   if len(chosen_ids) < sample_size:
      rng = random.Random(seed)
      remaining = [sample.sample_id for sample in universe if sample.sample_id not in chosen_ids]
      rng.shuffle(remaining)
      chosen_ids.extend(remaining[: sample_size - len(chosen_ids)])

   chosen_ids = chosen_ids[:sample_size]
   chosen_samples = [by_id[sample_id] for sample_id in chosen_ids]
   write_jsonl(manifest_path, [sample.to_json() for sample in chosen_samples])
   return chosen_samples
