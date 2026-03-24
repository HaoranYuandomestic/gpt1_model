import argparse

import torch
import yaml

from dataset import build_dataloaders
from eval_perplexity import compute_token_level_perplexity
from tokenizer_utils import load_tokenizer
from utils import GPTConfig, build_gpt1_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt1.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, valid_loader = build_dataloaders(config)
    tokenizer = load_tokenizer(
        config["data"].get("tokenizer_path", "data/tokenizer_gpt2"),
        cache_dir=config["data"].get("cache_dir"),
    )

    model_cfg = config["model"].copy()
    if model_cfg.get("vocab_size") in (None, 0, "auto"):
        model_cfg["vocab_size"] = len(tokenizer)

    model = build_gpt1_model(GPTConfig(**model_cfg)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = compute_token_level_perplexity(model, valid_loader, device)
    print(f"avg_nll: {metrics['avg_nll']:.6f}")
    print(f"token_level_perplexity: {metrics['perplexity']:.6f}")
    print(f"total_tokens: {metrics['total_tokens']}")


if __name__ == "__main__":
    main()
