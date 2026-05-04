from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch

from router_posttrain.action_space import load_action_library
from router_posttrain.data import load_router_samples
from router_posttrain.fidelis_env import FidelisBanditEnvironment
from router_posttrain.methods import BanditRLTrainer, RankingBasedTrainer, TrainerConfig, ValueBasedTrainer
from router_posttrain.pretrained_router import HFSequenceClassificationRouter
from router_posttrain.reward import EfficiencyBudgets, RewardWeights


def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--method", choices=["value", "ranking", "bandit"], required=True)
   parser.add_argument("--train_path", type=str, required=True)
   parser.add_argument("--eval_path", type=str, required=True)
   parser.add_argument("--dataset_name", type=str, default=None)
   parser.add_argument("--sample_limit", type=int, default=-1)
   parser.add_argument("--prior_checkpoint", type=str, required=True)
   parser.add_argument("--action_config", type=str, default=None)
   parser.add_argument("--output_dir", type=str, required=True)
   parser.add_argument("--num_epochs", type=int, default=3)
   parser.add_argument("--episodes_per_epoch", type=int, default=32)
   parser.add_argument("--batch_size", type=int, default=16)
   parser.add_argument("--hidden_dim", type=int, default=128)
   parser.add_argument("--learning_rate", type=float, default=2e-4)
   parser.add_argument("--weight_decay", type=float, default=0.01)
   parser.add_argument("--epsilon", type=float, default=0.15)
   parser.add_argument("--entropy_coef", type=float, default=0.01)
   parser.add_argument("--kl_coef", type=float, default=0.05)
   parser.add_argument("--seed", type=int, default=13)

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

   parser.add_argument("--reward_hit_weight", type=float, default=0.50)
   parser.add_argument("--reward_f1_weight", type=float, default=0.20)
   parser.add_argument("--reward_path_weight", type=float, default=0.15)
   parser.add_argument("--reward_eff_weight", type=float, default=0.15)
   parser.add_argument("--runtime_budget", type=float, default=30.0)
   parser.add_argument("--completion_budget", type=float, default=16.0)
   parser.add_argument("--candidate_budget", type=float, default=120.0)
   return parser.parse_args()


def build_fidelis_args(cli_args):
   namespace = argparse.Namespace()
   namespace.d = cli_args.d
   namespace.save_cache = cli_args.save_cache
   namespace.model_name = cli_args.model_name
   namespace.embedding_model = cli_args.embedding_model
   namespace.top_n = cli_args.top_n
   namespace.top_k = cli_args.top_k
   namespace.max_length = cli_args.max_length
   namespace.strategy = cli_args.strategy
   namespace.squeeze = cli_args.squeeze
   namespace.verifier = cli_args.verifier
   namespace.add_hop_information = cli_args.add_hop_information
   namespace.alpha = cli_args.alpha
   namespace.lookahead_hops = cli_args.lookahead_hops
   namespace.no_immediate_backtracking = cli_args.no_immediate_backtracking
   namespace.simple_path_only = cli_args.simple_path_only
   namespace.debug = False
   namespace.usage_tracker = None
   return namespace


def main():
   args = parse_args()
   random.seed(args.seed)
   torch.manual_seed(args.seed)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   train_samples = load_router_samples(args.train_path, dataset_name=args.dataset_name, limit=args.sample_limit)
   eval_samples = load_router_samples(args.eval_path, dataset_name=args.dataset_name, limit=args.sample_limit)

   reward_weights = RewardWeights(
      answer_hit=args.reward_hit_weight,
      answer_f1=args.reward_f1_weight,
      path_support=args.reward_path_weight,
      efficiency=args.reward_eff_weight,
   )
   efficiency_budgets = EfficiencyBudgets(
      runtime_seconds=args.runtime_budget,
      completion_calls=args.completion_budget,
      candidate_count=args.candidate_budget,
   )
   fidelis_args = build_fidelis_args(args)
   environment = FidelisBanditEnvironment(fidelis_args, reward_weights, efficiency_budgets)
   action_library = load_action_library(args.action_config)
   prior_router = HFSequenceClassificationRouter(args.prior_checkpoint, freeze=True)

   config = TrainerConfig(
      output_dir=args.output_dir,
      num_epochs=args.num_epochs,
      episodes_per_epoch=args.episodes_per_epoch,
      batch_size=args.batch_size,
      hidden_dim=args.hidden_dim,
      learning_rate=args.learning_rate,
      weight_decay=args.weight_decay,
      epsilon=args.epsilon,
      entropy_coef=args.entropy_coef,
      kl_coef=args.kl_coef,
      seed=args.seed,
   )

   if prior_router.num_actions != len(action_library):
      raise ValueError(
         f"Prior checkpoint exposes {prior_router.num_actions} labels but the action library has {len(action_library)} actions."
      )

   if args.method == "value":
      trainer = ValueBasedTrainer(prior_router, action_library, environment, config, device)
   elif args.method == "ranking":
      trainer = RankingBasedTrainer(prior_router, action_library, environment, config, device)
   else:
      trainer = BanditRLTrainer(prior_router, action_library, environment, config, device)

   trainer.train(train_samples, eval_samples)


if __name__ == "__main__":
   main()
