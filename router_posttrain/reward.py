from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, List, Sequence

from src.evaluate_results import eval_f1, eval_hit


@dataclass
class RewardWeights:
   answer_hit: float = 0.50
   answer_f1: float = 0.20
   path_support: float = 0.15
   efficiency: float = 0.15


@dataclass
class EfficiencyBudgets:
   runtime_seconds: float = 30.0
   completion_calls: float = 16.0
   candidate_count: float = 120.0


def _normalize_text(text: str) -> str:
   return re.sub(r"\s+", " ", text.lower().strip())


def _relation_sequence(reasoning_path: str) -> List[str]:
   parts = [part.strip() for part in reasoning_path.split("->") if part.strip()]
   return [_normalize_text(part) for part in parts[1::2]]


def _token_overlap_score(prediction: str, target: str) -> float:
   pred_tokens = set(_normalize_text(prediction).split())
   gold_tokens = set(_normalize_text(target).split())
   if not pred_tokens or not gold_tokens:
      return 0.0
   intersection = len(pred_tokens & gold_tokens)
   precision = intersection / len(pred_tokens)
   recall = intersection / len(gold_tokens)
   if precision + recall == 0:
      return 0.0
   return 2 * precision * recall / (precision + recall)


def compute_path_support(predicted_paths: Sequence[str], ground_paths: Sequence) -> float:
   if not predicted_paths or not ground_paths:
      return 0.0

   best_score = 0.0
   for predicted_path in predicted_paths:
      predicted_relations = _relation_sequence(predicted_path)
      for ground_path in ground_paths:
         if isinstance(ground_path, list):
            ground_relations = [_normalize_text(item) for item in ground_path]
            if ground_relations:
               exact = float(predicted_relations == ground_relations)
               overlap = len(set(predicted_relations) & set(ground_relations)) / max(len(set(ground_relations)), 1)
               best_score = max(best_score, 0.5 * exact + 0.5 * overlap)
         else:
            best_score = max(best_score, _token_overlap_score(predicted_path, str(ground_path)))
   return best_score


def compute_efficiency_score(metrics: Dict[str, float], budgets: EfficiencyBudgets) -> float:
   runtime_score = 1.0 - min(metrics.get("runtime_seconds", budgets.runtime_seconds) / budgets.runtime_seconds, 1.0)
   completion_score = 1.0 - min(metrics.get("completion_calls", budgets.completion_calls) / budgets.completion_calls, 1.0)
   candidate_score = 1.0 - min(metrics.get("candidate_count", budgets.candidate_count) / budgets.candidate_count, 1.0)
   return max(0.0, (runtime_score + completion_score + candidate_score) / 3.0)


def prediction_lines(raw_prediction: str) -> List[str]:
   if not raw_prediction:
      return []
   return [line.strip() for line in raw_prediction.split("\n") if line.strip()]


def compute_answer_metrics(prediction: Iterable[str], answer: List[str]) -> Dict[str, float]:
   prediction = list(prediction)
   hit = float(eval_hit(" ".join(prediction), answer))
   f1, precision, recall = eval_f1(prediction, answer)
   return {
      "answer_hit": hit,
      "answer_f1": float(f1),
      "answer_precision": float(precision),
      "answer_recall": float(recall),
   }


def compute_reward(metrics: Dict[str, float], weights: RewardWeights) -> float:
   total_weight = weights.answer_hit + weights.answer_f1 + weights.path_support + weights.efficiency
   if total_weight <= 0:
      raise ValueError("Reward weights must sum to a positive value.")
   reward = (
      weights.answer_hit * metrics.get("answer_hit", 0.0)
      + weights.answer_f1 * metrics.get("answer_f1", 0.0)
      + weights.path_support * metrics.get("path_support", 0.0)
      + weights.efficiency * metrics.get("efficiency", 0.0)
   ) / total_weight
   return float(reward)

