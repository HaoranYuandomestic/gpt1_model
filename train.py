import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from dataset import build_dataloaders
from eval_perplexity import compute_token_level_perplexity
from tokenizer_utils import load_tokenizer
from utils import GPTConfig, build_gpt1_model, count_parameters, shift_labels_for_causal_lm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt1.yaml")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - current_step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer,
    device: torch.device,
    scheduler=None,
    grad_clip: Optional[float] = 1.0,
    log_interval: int = 100,
):
    model.train()
    total_loss = 0.0
    total_steps = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = shift_labels_for_causal_lm(input_ids)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_steps += 1

        if total_steps % log_interval == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            tokens_seen = total_steps * input_ids.numel()
            tokens_per_sec = tokens_seen / elapsed
            pbar.set_postfix(loss=f"{loss.item():.4f}", tok_s=f"{tokens_per_sec:.0f}")

    return total_loss / max(1, total_steps)


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = shift_labels_for_causal_lm(input_ids)
        _, loss = model(input_ids, labels)
        total_loss += loss.item()
        total_steps += 1
    return total_loss / max(1, total_steps)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    config: Dict[str, Any],
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def maybe_resume(checkpoint_path, model, optimizer, scheduler, device):
    if checkpoint_path is None:
        return 0, float("inf")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0) + 1, ckpt.get("best_val_loss", float("inf"))


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["train"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = build_dataloaders(config)
    tokenizer = load_tokenizer(
        config["data"].get("tokenizer_path", "data/tokenizer_gpt2"),
        cache_dir=config["data"].get("cache_dir"),
    )

    model_cfg = config["model"].copy()
    if model_cfg.get("vocab_size") in (None, 0, "auto"):
        model_cfg["vocab_size"] = len(tokenizer)

    model = build_gpt1_model(GPTConfig(**model_cfg)).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
        betas=tuple(config["train"].get("betas", [0.9, 0.999])),
        eps=float(config["train"].get("eps", 1e-8)),
    )

    total_steps = len(train_loader) * int(config["train"]["epochs"])
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=int(config["train"].get("warmup_steps", 0)),
        total_steps=max(1, total_steps),
    )

    start_epoch, best_val_loss = maybe_resume(args.resume, model, optimizer, scheduler, device)

    ckpt_dir = config["train"].get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            grad_clip=config["train"].get("grad_clip", 1.0),
            log_interval=int(config["train"].get("log_interval", 100)),
        )
        val_loss = evaluate_loss(model, valid_loader, device)
        ppl = compute_token_level_perplexity(model, valid_loader, device)

        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_ppl={ppl['perplexity']:.4f}"
        )

        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_loss, config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_loss, config)


if __name__ == "__main__":
    main()
