from __future__ import annotations

import torch
import torch.nn as nn


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
   return nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, output_dim),
   )


class ValueRouterHead(nn.Module):
   def __init__(self, base_logit_dim: int, scalar_feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
      super().__init__()
      self.network = _make_mlp(base_logit_dim + scalar_feature_dim, hidden_dim, base_logit_dim, dropout)

   def forward(self, base_logits: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
      inputs = torch.cat([base_logits, scalar_features], dim=-1)
      return self.network(inputs)


class RankingRouterHead(nn.Module):
   def __init__(self, base_logit_dim: int, scalar_feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
      super().__init__()
      self.network = _make_mlp(base_logit_dim + scalar_feature_dim, hidden_dim, base_logit_dim, dropout)

   def forward(self, base_logits: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
      inputs = torch.cat([base_logits, scalar_features], dim=-1)
      return self.network(inputs)


class BanditPolicyHead(nn.Module):
   def __init__(self, base_logit_dim: int, scalar_feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
      super().__init__()
      self.network = _make_mlp(base_logit_dim + scalar_feature_dim, hidden_dim, base_logit_dim, dropout)

   def forward(self, base_logits: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
      inputs = torch.cat([base_logits, scalar_features], dim=-1)
      residual_logits = self.network(inputs)
      return base_logits + residual_logits

