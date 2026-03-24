import argparse

import torch
import yaml

from tokenizer_utils import decode_ids, load_tokenizer
from utils import GPTConfig, build_gpt1_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt1.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model.eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(decode_ids(tokenizer, output_ids[0].tolist()))


if __name__ == "__main__":
    main()
