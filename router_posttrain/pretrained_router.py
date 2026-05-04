from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PretrainedRouter(ABC, nn.Module):
   def __init__(self):
      super().__init__()

   @abstractmethod
   def predict_logits(self, questions: Sequence[str], device: torch.device) -> torch.Tensor:
      raise NotImplementedError

   @property
   @abstractmethod
   def num_actions(self) -> int:
      raise NotImplementedError


class HFSequenceClassificationRouter(PretrainedRouter):
   def __init__(self, checkpoint: str, freeze: bool = True, max_length: int = 128):
      super().__init__()
      self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
      self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
      self.max_length = max_length
      if freeze:
         for parameter in self.model.parameters():
            parameter.requires_grad = False

   @property
   def num_actions(self) -> int:
      return int(self.model.config.num_labels)

   def predict_logits(self, questions: Sequence[str], device: torch.device) -> torch.Tensor:
      encoded = self.tokenizer(
         list(questions),
         padding=True,
         truncation=True,
         max_length=self.max_length,
         return_tensors="pt",
      )
      encoded = {key: value.to(device) for key, value in encoded.items()}
      self.model.to(device)
      outputs = self.model(**encoded)
      return outputs.logits

