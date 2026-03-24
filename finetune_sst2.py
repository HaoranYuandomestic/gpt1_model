import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from tokenizer_utils import load_tokenizer
from utils import build_gpt1_model


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LocalSST2Dataset(Dataset):
    """
    读取 jsonl:
    {"sentence": "...", "label": 0/1}
    """
    def __init__(self, path: str, require_label: bool = True):
        if not os.path.exists(path):
            raise FileNotFoundError(f"SST-2 file not found: {path}")

        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                self.samples.append(self._normalize(ex, require_label))

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {path}")

    def _normalize(self, ex: Dict, require_label: bool):
        sentence = ex["sentence"]
        label = ex.get("label", None)

        if require_label:
            if label is None:
                raise ValueError("Missing label in labeled dataset.")
            label = int(label)
            if label not in [0, 1]:
                raise ValueError(f"Invalid label: {label}")
        else:
            label = -1 if label is None else int(label)

        return {
            "sentence": sentence,
            "label": label,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GPTForSequenceClassification(nn.Module):
    def __init__(self, gpt_backbone: nn.Module, hidden_size: int, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gpt = gpt_backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.size()
        device = input_ids.device

        if T > self.gpt.config.block_size:
            raise ValueError(
                f"Input length {T} exceeds block_size={self.gpt.config.block_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.gpt.wte(input_ids)
        pos_emb = self.gpt.wpe(pos)
        x = self.gpt.drop(tok_emb + pos_emb)

        for block in self.gpt.h:
            x = block(x)

        x = self.gpt.ln_f(x)

        if attention_mask is None:
            pooled = x[:, -1, :]
        else:
            lengths = attention_mask.sum(dim=1) - 1
            pooled = x[torch.arange(B, device=device), lengths]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


@dataclass
class SST2Collator:
    tokenizer: object
    max_length: int

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        sentences = [x["sentence"] for x in batch]
        labels = [x["label"] for x in batch]

        texts = [f"Sentence: {s}\nSentiment:" for s in sentences]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "sentences": sentences,
        }

        if labels[0] != -1:
            out["labels"] = torch.tensor(labels, dtype=torch.long)

        return out


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return avg_loss, acc


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_acc": best_acc,
        },
        path,
    )


def maybe_resize_wpe_for_loading(gpt_model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    if "wpe.weight" not in state_dict:
        return state_dict

    old_wpe = state_dict["wpe.weight"]
    new_wpe = gpt_model.state_dict()["wpe.weight"]

    if old_wpe.shape == new_wpe.shape:
        return state_dict

    old_len, dim1 = old_wpe.shape
    new_len, dim2 = new_wpe.shape
    if dim1 != dim2:
        raise ValueError(f"wpe hidden dim mismatch: {dim1} vs {dim2}")

    resized = new_wpe.clone()
    copy_len = min(old_len, new_len)
    resized[:copy_len] = old_wpe[:copy_len]

    if new_len > old_len:
        resized[old_len:] = old_wpe[-1:].repeat(new_len - old_len, 1)

    state_dict["wpe.weight"] = resized
    print(f"[Info] resized wpe.weight from {tuple(old_wpe.shape)} to {tuple(resized.shape)}")
    return state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer_gpt2")
    parser.add_argument("--train_file", type=str, default="data/sst2/train.jsonl")
    parser.add_argument("--valid_file", type=str, default="data/sst2/validation.jsonl")

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.002)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints/sst2")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_path)

    vocab_size = tokenizer.vocab_size
    if tokenizer.pad_token_id is not None:
        vocab_size = max(vocab_size, tokenizer.pad_token_id + 1)

    gpt = build_gpt1_model(vocab_size=vocab_size, block_size=args.block_size)

    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_dict = maybe_resize_wpe_for_loading(gpt, state_dict)
    gpt.load_state_dict(state_dict, strict=False)

    model = GPTForSequenceClassification(
        gpt_backbone=gpt,
        hidden_size=gpt.config.n_embd,
        num_labels=2,
        dropout=0.1,
    ).to(device)

    train_ds = LocalSST2Dataset(args.train_file, require_label=True)
    valid_ds = LocalSST2Dataset(args.valid_file, require_label=True)

    collator = SST2Collator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    no_decay = ["bias", "ln_", "ln_f.weight", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            logits = outputs["logits"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=-1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_count += labels.size(0)

            pbar.set_postfix(
                loss=f"{running_loss / running_count:.4f}",
                acc=f"{running_correct / running_count:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_loss / running_count
        train_acc = running_correct / running_count

        val_loss, val_acc = evaluate(model, valid_loader, device)

        print(
            f"[Epoch {epoch+1}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            best_acc=best_val_acc,
            path=os.path.join(args.save_dir, "last.pt"),
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                best_acc=best_val_acc,
                path=os.path.join(args.save_dir, "best.pt"),
            )
            print(f"New best model saved. val_acc={best_val_acc:.4f}")

    print("\nLoading best checkpoint for final validation...")
    best_ckpt = torch.load(os.path.join(args.save_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_loss, val_acc = evaluate(model, valid_loader, device)
    print(f"[Validation] loss={val_loss:.4f} acc={val_acc:.4f}")


if __name__ == "__main__":
    main()