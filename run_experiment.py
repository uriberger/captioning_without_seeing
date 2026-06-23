#!/usr/bin/env python3
import argparse
from config import Config
from pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description="Captioning without seeing — experiment runner")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")

    # Override any Config field from the command line
    parser.add_argument("--oracle_model", type=str)
    parser.add_argument("--blind_model", type=str)
    parser.add_argument("--oracle_thinking", type=lambda x: x.lower() == "true")
    parser.add_argument("--blind_thinking", type=lambda x: x.lower() == "true")
    parser.add_argument("--n_queries", type=int)
    parser.add_argument("--locale", type=str)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_answer_tokens", type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.from_json(args.config) if args.config else Config()

    # Apply CLI overrides
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    for k, v in overrides.items():
        setattr(cfg, k, v)

    print("Config:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")

    run(cfg)


if __name__ == "__main__":
    main()
