from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from router_posttrain.action_space import load_action_library
from router_posttrain.data import load_router_samples, build_scalar_features, RouterSample
from router_posttrain.fidelis_env import FidelisBanditEnvironment
from router_posttrain.heads import BanditPolicyHead, RankingRouterHead, ValueRouterHead
from router_posttrain.pretrained_router import HFSequenceClassificationRouter
from router_posttrain.reward import EfficiencyBudgets, RewardWeights
from router_posttrain.train import build_fidelis_args


def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--method", choices=["value", "ranking", "bandit"], required=True)
   parser.add_argument("--data_path", type=str, required=True)
   parser.add_argument("--prior_checkpoint", type=str, required=True)
   parser.add_argument("--head_checkpoint", type=str, required=True)
   parser.add_argument("--output_path", type=str, required=True)
   parser.add_argument("--action_config", type=str, default=None)
   parser.add_argument("--dataset_name", type=str, default=None)
   parser.add_argument("--sample_limit", type=int, default=-1)
   parser.add_argument("--hidden_dim", type=int, default=128)
   parser.add_argument("--d", type=str, default="RoG-webqsp")
   parser.add_argument("--save_cache", type=str, default="/data/rog/datasets")
   parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
   parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
   parser.add_argument("--top_n", type=int, default=30)
   parser.add_argument("--top_k", type=int, default=3)
   parser.add_argument("--max_length", type=int, default=3)
   parser.add_argument("--strategy", type=str, default="discrete_rating")
   parser.add_argument("--squeeze", type=bool, default=True)
   parser.add_argument("--verifier", type=str, default="deductive+planning")
   parser.add_argument("--add_hop_information", action="store_true")
   parser.add_argument("--alpha", type=float, default=0.3)
   parser.add_argument("--lookahead_hops", type=int, default=0)
   parser.add_argument("--no_immediate_backtracking", action="store_true")
   parser.add_argument("--simple_path_only", action="store_true")
   return parser.parse_args()


def build_head(method: str, num_actions: int, scalar_dim: int, hidden_dim: int):
   if method == "value":
      return ValueRouterHead(num_actions, scalar_dim, hidden_dim)
   if method == "ranking":
      return RankingRouterHead(num_actions, scalar_dim, hidden_dim)
   return BanditPolicyHead(num_actions, scalar_dim, hidden_dim)


def choose_action(method: str, head, base_logits: torch.Tensor, scalar_features: torch.Tensor, device: torch.device) -> int:
   with torch.no_grad():
      scores = head(base_logits.unsqueeze(0).to(device), scalar_features.unsqueeze(0).to(device)).squeeze(0)
   return int(torch.argmax(scores).item())


def main():
   args = parse_args()
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   samples = load_router_samples(args.data_path, dataset_name=args.dataset_name, limit=args.sample_limit)
   action_library = load_action_library(args.action_config)
   prior_router = HFSequenceClassificationRouter(args.prior_checkpoint, freeze=True)
   scalar_dim = build_scalar_features(RouterSample(id="shape", question="shape probe", q_entity=[], a_entity=[], graph=[], ground_paths=[])).numel()
   head = build_head(args.method, len(action_library), scalar_dim, args.hidden_dim).to(device)
   head.load_state_dict(torch.load(args.head_checkpoint, map_location=device))
   head.eval()

   reward_weights = RewardWeights()
   budgets = EfficiencyBudgets()
   environment = FidelisBanditEnvironment(build_fidelis_args(args), reward_weights, budgets)

   outputs = []
   for sample in samples:
      base_logits = prior_router.predict_logits([sample.question], device).detach().cpu().squeeze(0)
      scalar_features = build_scalar_features(sample).cpu()
      action_id = choose_action(args.method, head, base_logits, scalar_features, device)
      result = environment.evaluate(sample, action_library[action_id])
      outputs.append(
         {
            "sample_id": sample.id,
            "action_id": action_id,
            "policy_name": action_library[action_id].name,
            "reward": result.reward,
            "metrics": result.metrics,
            "prediction": result.prediction,
         }
      )

   output_path = Path(args.output_path)
   output_path.parent.mkdir(parents=True, exist_ok=True)
   with output_path.open("w") as handle:
      for item in outputs:
         handle.write(json.dumps(item) + "\n")


if __name__ == "__main__":
   main()
