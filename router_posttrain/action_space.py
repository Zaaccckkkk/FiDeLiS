from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class RouterPolicy:
   id: int
   name: str
   max_length: int
   top_k: int
   top_n: int
   alpha: float
   lookahead_hops: int = 0
   add_hop_information: bool = False
   no_immediate_backtracking: bool = True
   simple_path_only: bool = True
   verifier: Optional[str] = None

   def to_overrides(self) -> dict:
      overrides = {
         "max_length": self.max_length,
         "top_k": self.top_k,
         "top_n": self.top_n,
         "alpha": self.alpha,
         "lookahead_hops": self.lookahead_hops,
         "add_hop_information": self.add_hop_information,
         "no_immediate_backtracking": self.no_immediate_backtracking,
         "simple_path_only": self.simple_path_only,
      }
      if self.verifier:
         overrides["verifier"] = self.verifier
      return overrides

   def to_json(self) -> dict:
      return asdict(self)


def default_action_library() -> List[RouterPolicy]:
   return [
      RouterPolicy(0, "short_greedy", max_length=1, top_k=1, top_n=12, alpha=0.0),
      RouterPolicy(1, "short_balanced", max_length=2, top_k=2, top_n=20, alpha=0.0),
      RouterPolicy(2, "medium_balanced", max_length=3, top_k=3, top_n=30, alpha=0.2),
      RouterPolicy(3, "medium_lookahead", max_length=3, top_k=3, top_n=30, alpha=0.3, lookahead_hops=1, add_hop_information=True),
      RouterPolicy(4, "deep_balanced", max_length=4, top_k=3, top_n=36, alpha=0.2),
      RouterPolicy(5, "deep_lookahead", max_length=4, top_k=4, top_n=40, alpha=0.35, lookahead_hops=1, add_hop_information=True),
      RouterPolicy(6, "wide_explore", max_length=3, top_k=5, top_n=50, alpha=0.15),
      RouterPolicy(7, "deep_wide_explore", max_length=5, top_k=5, top_n=60, alpha=0.25, lookahead_hops=1, add_hop_information=True),
   ]


def load_action_library(path: Optional[str]) -> List[RouterPolicy]:
   if path is None:
      return default_action_library()

   config_path = Path(path)
   with config_path.open("r") as handle:
      raw = json.load(handle)

   policies: List[RouterPolicy] = []
   for item in raw:
      policies.append(RouterPolicy(**item))
   return policies


def action_map(policies: Iterable[RouterPolicy]) -> dict:
   return {policy.id: policy for policy in policies}

