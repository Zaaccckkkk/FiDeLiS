from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional

import torch


@dataclass
class RolloutRecord:
   sample_id: str
   question: str
   action_id: int
   reward: float
   base_logits: torch.Tensor
   scalar_features: torch.Tensor
   metrics: Dict[str, float]
   payload: Dict


@dataclass
class PreferenceRecord:
   sample_id: str
   question: str
   better_action_id: int
   worse_action_id: int
   base_logits: torch.Tensor
   scalar_features: torch.Tensor
   better_reward: float
   worse_reward: float


class ReplayBuffer:
   def __init__(self, capacity: int = 4096):
      self.capacity = capacity
      self.records: List[RolloutRecord] = []

   def add(self, record: RolloutRecord):
      self.records.append(record)
      if len(self.records) > self.capacity:
         self.records = self.records[-self.capacity :]

   def sample(self, batch_size: int) -> List[RolloutRecord]:
      return random.sample(self.records, min(batch_size, len(self.records)))

   def __len__(self) -> int:
      return len(self.records)


class PreferenceBuffer:
   def __init__(self, capacity: int = 4096):
      self.capacity = capacity
      self.records: List[PreferenceRecord] = []

   def add(self, record: PreferenceRecord):
      self.records.append(record)
      if len(self.records) > self.capacity:
         self.records = self.records[-self.capacity :]

   def sample(self, batch_size: int) -> List[PreferenceRecord]:
      return random.sample(self.records, min(batch_size, len(self.records)))

   def __len__(self) -> int:
      return len(self.records)

