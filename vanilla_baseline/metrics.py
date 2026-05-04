from __future__ import annotations

import re
import string
from typing import Iterable, List, Tuple


def normalize(text: str) -> str:
   text = str(text).lower()
   exclude = set(string.punctuation)
   text = "".join(char for char in text if char not in exclude)
   text = re.sub(r"\b(a|an|the)\b", " ", text)
   text = re.sub(r"\b(<pad>)\b", " ", text)
   return " ".join(text.split())


def match(prediction: str, answer: str) -> bool:
   return normalize(answer) in normalize(prediction)


def eval_acc(prediction: str, answers: List[str]) -> float:
   if not answers:
      return 0.0
   matched = 0.0
   for answer in answers:
      if match(prediction, answer):
         matched += 1
   return matched / len(answers)


def eval_hit(prediction: str, answers: List[str]) -> int:
   for answer in answers:
      if match(prediction, answer):
         return 1
   return 0


def eval_f1(predictions: Iterable[str], answers: List[str]) -> Tuple[float, float, float]:
   predictions = [prediction for prediction in predictions if str(prediction).strip()]
   if not predictions or not answers:
      return 0.0, 0.0, 0.0

   matched = 0
   prediction_str = " ".join(predictions)
   for answer in answers:
      if match(prediction_str, answer):
         matched += 1

   precision = matched / len(predictions)
   recall = matched / len(answers)
   if precision + recall == 0:
      return 0.0, precision, recall
   return 2 * precision * recall / (precision + recall), precision, recall


def compute_metrics(predictions: Iterable[str], answers: List[str]) -> dict:
   predictions = [str(prediction).strip() for prediction in predictions if str(prediction).strip()]
   prediction_str = " ".join(predictions)
   f1, precision, recall = eval_f1(predictions, answers)
   return {
      "hit": eval_hit(prediction_str, answers),
      "f1": f1,
      "precision": precision,
      "recall": recall,
      "accuracy": eval_acc(prediction_str, answers),
   }

