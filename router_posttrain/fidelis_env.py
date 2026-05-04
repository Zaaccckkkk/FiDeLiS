from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import time
from typing import Dict

from src.llm_navigator import LLM_Navigator

from router_posttrain.action_space import RouterPolicy
from router_posttrain.data import RouterSample
from router_posttrain.reward import (
   EfficiencyBudgets,
   RewardWeights,
   compute_answer_metrics,
   compute_efficiency_score,
   compute_path_support,
   compute_reward,
   prediction_lines,
)


@dataclass
class EnvironmentResult:
   reward: float
   metrics: Dict[str, float]
   prediction: Dict


class FidelisBanditEnvironment:
   def __init__(
      self,
      args_template,
      reward_weights: RewardWeights,
      efficiency_budgets: EfficiencyBudgets,
   ):
      self.args_template = args_template
      self.reward_weights = reward_weights
      self.efficiency_budgets = efficiency_budgets

   def evaluate(self, sample: RouterSample, policy: RouterPolicy) -> EnvironmentResult:
      args = deepcopy(self.args_template)
      usage_tracker: Dict[str, float] = {}
      args.usage_tracker = usage_tracker

      navigator = LLM_Navigator(args)
      start_time = time.time()
      result, _ = navigator.beam_search(sample.__dict__, policy_overrides=policy.to_overrides())
      runtime_seconds = time.time() - start_time

      llm_prediction = prediction_lines(result.get("prediction_llm", ""))
      direct_prediction = prediction_lines(result.get("prediction_direct_answer", ""))
      llm_metrics = compute_answer_metrics(llm_prediction, sample.a_entity)
      direct_metrics = compute_answer_metrics(direct_prediction, sample.a_entity)

      metrics: Dict[str, float] = {
         "llm_answer_hit": llm_metrics["answer_hit"],
         "llm_answer_f1": llm_metrics["answer_f1"],
         "direct_answer_hit": direct_metrics["answer_hit"],
         "direct_answer_f1": direct_metrics["answer_f1"],
         "answer_hit": max(llm_metrics["answer_hit"], direct_metrics["answer_hit"]),
         "answer_f1": max(llm_metrics["answer_f1"], direct_metrics["answer_f1"]),
         "path_support": compute_path_support(result.get("reasoning_path", []), sample.ground_paths),
         "runtime_seconds": runtime_seconds,
         "completion_calls": float(usage_tracker.get("completion_calls", 0) + usage_tracker.get("batch_completion_calls", 0)),
         "embedding_calls": float(usage_tracker.get("embedding_calls", 0)),
         "candidate_count": float(result.get("search_metrics", {}).get("candidate_count", 0)),
         "termination_checks": float(result.get("search_metrics", {}).get("termination_checks", 0)),
      }
      metrics["efficiency"] = compute_efficiency_score(metrics, self.efficiency_budgets)
      reward = compute_reward(metrics, self.reward_weights)
      return EnvironmentResult(reward=reward, metrics=metrics, prediction=result)

