from __future__ import annotations

import argparse
import json

from .data import CollectionConfig, audit_dataset, collect
from .env import EnvConfig, bank_manifest
from .evaluate import closed_loop, evaluate_model
from .pipeline import run_pipeline
from .ppo import PPOConfig, load_expert, train_expert
from .train import ModelConfig, train_model
from .utils import device_for, runtime, seed_everything, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="mini-WAM: PPO states -> RGB -> policy/video/joint -> closed-loop SR")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    pipeline = sub.add_parser("pipeline", help="Complete experiment; all three modes")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--out", required=True)
    pipeline.add_argument("--resume", action="store_true")
    pipeline.add_argument("--require-perfect", action="store_true", help="Exit 2 after reporting if any test SR < 100%")
    expert = sub.add_parser("expert", help="Train state PPO")
    expert.add_argument("--depth", type=int, choices=(1, 2), default=1)
    expert.add_argument("--size", type=int, default=5)
    expert.add_argument("--updates", type=int, default=150)
    expert.add_argument("--seed", type=int, default=42)
    expert.add_argument("--out", required=True)
    expert.add_argument("--resume", action="store_true")
    collection = sub.add_parser("collect", help="Collect pixels from trained PPO plus counterfactual actions")
    collection.add_argument("--expert", required=True)
    collection.add_argument("--train-repeats", type=int, default=32)
    collection.add_argument("--eval-repeats", type=int, default=8)
    collection.add_argument("--exploration", type=float, default=0.25)
    collection.add_argument("--seed", type=int, default=43)
    collection.add_argument("--out", required=True)
    collection.add_argument("--resume", action="store_true")
    train = sub.add_parser("train", help="Train one objective")
    train.add_argument("--dataset", required=True)
    train.add_argument("--mode", choices=("policy", "video", "joint"), required=True)
    train.add_argument("--horizon", type=int, default=3)
    train.add_argument("--updates", type=int, default=1200)
    train.add_argument("--seed", type=int, default=3407)
    train.add_argument("--out", required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate", help="Pixel model closed-loop SR + offline forecasts")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--out", required=True)
    eval_expert = sub.add_parser("evaluate-expert", help="State PPO exhaustive SR")
    eval_expert.add_argument("--checkpoint", required=True)
    eval_expert.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    eval_expert.add_argument("--out", required=True)
    audit = sub.add_parser("audit", help="Checksums, transition replay, split leakage and action coverage")
    audit.add_argument("--dataset", required=True)
    sub.add_parser("doctor", help="Runtime and complete finite task-bank sizes")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    seed_everything(42, args.threads)
    device = device_for(args.device)
    if args.command == "pipeline":
        result = run_pipeline(args.config, args.out, device, args.resume)
        if args.require_perfect and not result["all_test_sr_100"]:
            return 2
    elif args.command == "expert":
        train_expert(EnvConfig(size=args.size, depth=args.depth), PPOConfig(updates=args.updates, seed=args.seed), args.out, device, args.resume)
    elif args.command == "collect":
        collect(args.expert, CollectionConfig(args.train_repeats, args.eval_repeats, args.exploration, args.seed), args.out, device, args.resume)
    elif args.command == "train":
        train_model(args.dataset, ModelConfig(mode=args.mode, horizon=args.horizon, updates=args.updates, seed=args.seed), args.out, device, args.resume)
    elif args.command == "evaluate":
        result = evaluate_model(args.checkpoint, args.dataset, args.split, args.out, device)
        print(json.dumps(result["closed_loop"], indent=2))
    elif args.command == "evaluate-expert":
        model, config = load_expert(args.checkpoint, device)
        result = closed_loop(model, config, args.split, device, state_policy=True)
        write_json(args.out, result)
        print(json.dumps(result, indent=2))
    elif args.command == "audit":
        print(json.dumps(audit_dataset(args.dataset), indent=2))
    elif args.command == "doctor":
        print(json.dumps({**runtime(device), "task_banks": {str(depth): {
            key: len(value) for key, value in bank_manifest(EnvConfig(depth=depth))["tasks"].items()}
            for depth in (1, 2)}}, indent=2))
    return 0
