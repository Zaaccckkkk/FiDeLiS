from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from router_posttrain.action_space import RouterPolicy
from router_posttrain.data import RouterSample, build_scalar_features
from router_posttrain.fidelis_env import FidelisBanditEnvironment
from router_posttrain.heads import BanditPolicyHead, RankingRouterHead, ValueRouterHead
from router_posttrain.replay import PreferenceBuffer, PreferenceRecord, ReplayBuffer, RolloutRecord


@dataclass
class TrainerConfig:
   output_dir: str
   num_epochs: int = 3
   episodes_per_epoch: int = 32
   batch_size: int = 16
   hidden_dim: int = 128
   learning_rate: float = 2e-4
   weight_decay: float = 0.01
   epsilon: float = 0.15
   entropy_coef: float = 0.01
   kl_coef: float = 0.05
   replay_capacity: int = 2048
   pair_rollouts_per_query: int = 2
   seed: int = 13


class BasePostTrainer:
   def __init__(
      self,
      prior_router,
      action_library: List[RouterPolicy],
      environment: FidelisBanditEnvironment,
      config: TrainerConfig,
      device: torch.device,
   ):
      self.prior_router = prior_router
      self.action_library = action_library
      self.environment = environment
      self.config = config
      self.device = device
      self.output_dir = Path(config.output_dir)
      self.output_dir.mkdir(parents=True, exist_ok=True)
      random.seed(config.seed)
      torch.manual_seed(config.seed)

   def _base_logits(self, sample: RouterSample) -> torch.Tensor:
      with torch.no_grad():
         logits = self.prior_router.predict_logits([sample.question], self.device)
      return logits.detach().cpu().squeeze(0)

   def _scalar_features(self, sample: RouterSample) -> torch.Tensor:
      return build_scalar_features(sample).cpu()

   def _write_jsonl(self, path: Path, payloads: List[Dict]):
      with path.open("a") as handle:
         for item in payloads:
            handle.write(json.dumps(item) + "\n")

   def evaluate(self, samples: List[RouterSample]) -> Dict[str, float]:
      raise NotImplementedError

   def train(self, train_samples: List[RouterSample], eval_samples: List[RouterSample]):
      raise NotImplementedError


class ValueBasedTrainer(BasePostTrainer):
   def __init__(self, prior_router, action_library, environment, config, device):
      super().__init__(prior_router, action_library, environment, config, device)
      scalar_dim = build_scalar_features(
         RouterSample(id="shape", question="shape probe", q_entity=[], a_entity=[], graph=[], ground_paths=[])
      ).numel()
      self.model = ValueRouterHead(len(action_library), scalar_dim, config.hidden_dim).to(device)
      self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
      self.buffer = ReplayBuffer(config.replay_capacity)

   def _select_action(self, base_logits: torch.Tensor, scalar_features: torch.Tensor, explore: bool) -> int:
      with torch.no_grad():
         q_values = self.model(base_logits.unsqueeze(0).to(self.device), scalar_features.unsqueeze(0).to(self.device)).squeeze(0)
      if explore and random.random() < self.config.epsilon:
         return random.randrange(len(self.action_library))
      return int(torch.argmax(q_values).item())

   def _update(self):
      if len(self.buffer) < self.config.batch_size:
         return None
      batch = self.buffer.sample(self.config.batch_size)
      base_logits = torch.stack([record.base_logits for record in batch]).to(self.device)
      scalar_features = torch.stack([record.scalar_features for record in batch]).to(self.device)
      actions = torch.tensor([record.action_id for record in batch], dtype=torch.long, device=self.device)
      rewards = torch.tensor([record.reward for record in batch], dtype=torch.float32, device=self.device)

      q_values = self.model(base_logits, scalar_features)
      predicted = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
      loss = F.mse_loss(predicted, rewards)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      return float(loss.item())

   def evaluate(self, samples: List[RouterSample]) -> Dict[str, float]:
      rewards = []
      answer_hits = []
      for sample in samples:
         base_logits = self._base_logits(sample)
         scalar_features = self._scalar_features(sample)
         action_id = self._select_action(base_logits, scalar_features, explore=False)
         result = self.environment.evaluate(sample, self.action_library[action_id])
         rewards.append(result.reward)
         answer_hits.append(result.metrics["answer_hit"])
      return {
         "avg_reward": sum(rewards) / max(len(rewards), 1),
         "avg_answer_hit": sum(answer_hits) / max(len(answer_hits), 1),
      }

   def train(self, train_samples: List[RouterSample], eval_samples: List[RouterSample]):
      log_path = self.output_dir / "value_train.jsonl"
      for epoch in range(self.config.num_epochs):
         epoch_logs: List[Dict] = []
         for step in range(self.config.episodes_per_epoch):
            sample = random.choice(train_samples)
            base_logits = self._base_logits(sample)
            scalar_features = self._scalar_features(sample)
            action_id = self._select_action(base_logits, scalar_features, explore=True)
            result = self.environment.evaluate(sample, self.action_library[action_id])
            self.buffer.add(
               RolloutRecord(
                  sample_id=sample.id,
                  question=sample.question,
                  action_id=action_id,
                  reward=result.reward,
                  base_logits=base_logits,
                  scalar_features=scalar_features,
                  metrics=result.metrics,
                  payload=result.prediction,
               )
            )
            loss = self._update()
            epoch_logs.append(
               {
                  "epoch": epoch,
                  "step": step,
                  "sample_id": sample.id,
                  "action_id": action_id,
                  "policy_name": self.action_library[action_id].name,
                  "reward": result.reward,
                  "loss": loss,
                  **result.metrics,
               }
            )
         self._write_jsonl(log_path, epoch_logs)
         torch.save(self.model.state_dict(), self.output_dir / f"value_router_epoch_{epoch}.pt")
         eval_metrics = self.evaluate(eval_samples)
         self._write_jsonl(self.output_dir / "value_eval.jsonl", [{"epoch": epoch, **eval_metrics}])


class RankingBasedTrainer(BasePostTrainer):
   def __init__(self, prior_router, action_library, environment, config, device):
      super().__init__(prior_router, action_library, environment, config, device)
      scalar_dim = build_scalar_features(
         RouterSample(id="shape", question="shape probe", q_entity=[], a_entity=[], graph=[], ground_paths=[])
      ).numel()
      self.model = RankingRouterHead(len(action_library), scalar_dim, config.hidden_dim).to(device)
      self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
      self.buffer = PreferenceBuffer(config.replay_capacity)

   def _rank_actions(self, base_logits: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
      with torch.no_grad():
         return self.model(base_logits.unsqueeze(0).to(self.device), scalar_features.unsqueeze(0).to(self.device)).squeeze(0).cpu()

   def _collect_preferences(self, sample: RouterSample) -> PreferenceRecord:
      base_logits = self._base_logits(sample)
      scalar_features = self._scalar_features(sample)
      utility = self._rank_actions(base_logits, scalar_features)
      first_action = int(torch.argmax(utility).item())
      candidates = [index for index in range(len(self.action_library)) if index != first_action]
      second_action = random.choice(candidates)

      first_result = self.environment.evaluate(sample, self.action_library[first_action])
      second_result = self.environment.evaluate(sample, self.action_library[second_action])
      if first_result.reward >= second_result.reward:
         better_action, worse_action = first_action, second_action
         better_reward, worse_reward = first_result.reward, second_result.reward
      else:
         better_action, worse_action = second_action, first_action
         better_reward, worse_reward = second_result.reward, first_result.reward

      return PreferenceRecord(
         sample_id=sample.id,
         question=sample.question,
         better_action_id=better_action,
         worse_action_id=worse_action,
         base_logits=base_logits,
         scalar_features=scalar_features,
         better_reward=better_reward,
         worse_reward=worse_reward,
      )

   def _update(self):
      if len(self.buffer) < self.config.batch_size:
         return None
      batch = self.buffer.sample(self.config.batch_size)
      base_logits = torch.stack([record.base_logits for record in batch]).to(self.device)
      scalar_features = torch.stack([record.scalar_features for record in batch]).to(self.device)
      better_actions = torch.tensor([record.better_action_id for record in batch], dtype=torch.long, device=self.device)
      worse_actions = torch.tensor([record.worse_action_id for record in batch], dtype=torch.long, device=self.device)

      utilities = self.model(base_logits, scalar_features)
      better_scores = utilities.gather(1, better_actions.unsqueeze(1)).squeeze(1)
      worse_scores = utilities.gather(1, worse_actions.unsqueeze(1)).squeeze(1)
      loss = -F.logsigmoid(better_scores - worse_scores).mean()

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      return float(loss.item())

   def evaluate(self, samples: List[RouterSample]) -> Dict[str, float]:
      rewards = []
      answer_hits = []
      for sample in samples:
         base_logits = self._base_logits(sample)
         scalar_features = self._scalar_features(sample)
         with torch.no_grad():
            utilities = self.model(base_logits.unsqueeze(0).to(self.device), scalar_features.unsqueeze(0).to(self.device)).squeeze(0)
         action_id = int(torch.argmax(utilities).item())
         result = self.environment.evaluate(sample, self.action_library[action_id])
         rewards.append(result.reward)
         answer_hits.append(result.metrics["answer_hit"])
      return {
         "avg_reward": sum(rewards) / max(len(rewards), 1),
         "avg_answer_hit": sum(answer_hits) / max(len(answer_hits), 1),
      }

   def train(self, train_samples: List[RouterSample], eval_samples: List[RouterSample]):
      log_path = self.output_dir / "ranking_train.jsonl"
      for epoch in range(self.config.num_epochs):
         epoch_logs: List[Dict] = []
         for step in range(self.config.episodes_per_epoch):
            sample = random.choice(train_samples)
            record = self._collect_preferences(sample)
            self.buffer.add(record)
            loss = self._update()
            epoch_logs.append(
               {
                  "epoch": epoch,
                  "step": step,
                  "sample_id": record.sample_id,
                  "better_action_id": record.better_action_id,
                  "worse_action_id": record.worse_action_id,
                  "better_reward": record.better_reward,
                  "worse_reward": record.worse_reward,
                  "loss": loss,
               }
            )
         self._write_jsonl(log_path, epoch_logs)
         torch.save(self.model.state_dict(), self.output_dir / f"ranking_router_epoch_{epoch}.pt")
         eval_metrics = self.evaluate(eval_samples)
         self._write_jsonl(self.output_dir / "ranking_eval.jsonl", [{"epoch": epoch, **eval_metrics}])


class BanditRLTrainer(BasePostTrainer):
   def __init__(self, prior_router, action_library, environment, config, device):
      super().__init__(prior_router, action_library, environment, config, device)
      scalar_dim = build_scalar_features(
         RouterSample(id="shape", question="shape probe", q_entity=[], a_entity=[], graph=[], ground_paths=[])
      ).numel()
      self.model = BanditPolicyHead(len(action_library), scalar_dim, config.hidden_dim).to(device)
      self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
      self.reward_baseline = 0.0

   def _sample_action(self, base_logits: torch.Tensor, scalar_features: torch.Tensor, explore: bool) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
      base_logits_device = base_logits.unsqueeze(0).to(self.device)
      scalar_features_device = scalar_features.unsqueeze(0).to(self.device)
      policy_logits = self.model(base_logits_device, scalar_features_device).squeeze(0)
      if explore:
         distribution = torch.distributions.Categorical(logits=policy_logits)
         action = distribution.sample()
      else:
         distribution = torch.distributions.Categorical(logits=policy_logits)
         action = torch.argmax(policy_logits)
      return int(action.item()), distribution.log_prob(action), distribution.entropy(), policy_logits

   def evaluate(self, samples: List[RouterSample]) -> Dict[str, float]:
      rewards = []
      answer_hits = []
      for sample in samples:
         base_logits = self._base_logits(sample)
         scalar_features = self._scalar_features(sample)
         action_id, _, _, _ = self._sample_action(base_logits, scalar_features, explore=False)
         result = self.environment.evaluate(sample, self.action_library[action_id])
         rewards.append(result.reward)
         answer_hits.append(result.metrics["answer_hit"])
      return {
         "avg_reward": sum(rewards) / max(len(rewards), 1),
         "avg_answer_hit": sum(answer_hits) / max(len(answer_hits), 1),
      }

   def train(self, train_samples: List[RouterSample], eval_samples: List[RouterSample]):
      log_path = self.output_dir / "bandit_train.jsonl"
      for epoch in range(self.config.num_epochs):
         epoch_logs: List[Dict] = []
         for step in range(self.config.episodes_per_epoch):
            sample = random.choice(train_samples)
            base_logits = self._base_logits(sample)
            scalar_features = self._scalar_features(sample)
            action_id, log_prob, entropy, policy_logits = self._sample_action(base_logits, scalar_features, explore=True)
            result = self.environment.evaluate(sample, self.action_library[action_id])

            reward = torch.tensor(result.reward, dtype=torch.float32, device=self.device)
            baseline = torch.tensor(self.reward_baseline, dtype=torch.float32, device=self.device)
            advantage = reward - baseline
            prior_distribution = torch.softmax(base_logits.to(self.device), dim=-1)
            current_distribution = torch.softmax(policy_logits, dim=-1)
            kl = F.kl_div(current_distribution.log(), prior_distribution, reduction="batchmean")
            loss = -(advantage.detach() * log_prob) - self.config.entropy_coef * entropy + self.config.kl_coef * kl

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * result.reward
            epoch_logs.append(
               {
                  "epoch": epoch,
                  "step": step,
                  "sample_id": sample.id,
                  "action_id": action_id,
                  "policy_name": self.action_library[action_id].name,
                  "reward": result.reward,
                  "loss": float(loss.item()),
                  "kl": float(kl.item()),
                  **result.metrics,
               }
            )

         self._write_jsonl(log_path, epoch_logs)
         torch.save(self.model.state_dict(), self.output_dir / f"bandit_router_epoch_{epoch}.pt")
         eval_metrics = self.evaluate(eval_samples)
         self._write_jsonl(self.output_dir / "bandit_eval.jsonl", [{"epoch": epoch, **eval_metrics}])

