import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tokenizer_utils import load_tokenizer
from utils import build_gpt1_model


LABEL_ID2NAME = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_ID2NAME.items()}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LocalSNLIDataset(Dataset):
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"SNLI file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            raise ValueError(f"Empty SNLI file: {path}")

        if text[0] == "[":
            raw_samples = json.loads(text)
        else:
            raw_samples = [json.loads(line) for line in text.splitlines() if line.strip()]

        self.samples = []
        for ex in raw_samples:
            norm = self._normalize(ex)
            if norm is not None:
                self.samples.append(norm)

        if not self.samples:
            raise ValueError(f"No valid SNLI samples found in {path}")

    def _normalize(self, ex: Dict):
        premise = ex.get("premise") or ex.get("sentence1")
        hypothesis = ex.get("hypothesis") or ex.get("sentence2")
        label = ex.get("label")

        if premise is None or hypothesis is None or label is None:
            return None

        if isinstance(label, str):
            label = label.strip().lower()
            if label not in LABEL_NAME_TO_ID:
                return None
            label = LABEL_NAME_TO_ID[label]
        else:
            label = int(label)
            if label not in [0, 1, 2]:
                return None

        return {"premise": str(premise), "hypothesis": str(hypothesis), "label": label}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GPTForSequenceClassification(nn.Module):
    def __init__(self, gpt_backbone: nn.Module, hidden_size: int, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.gpt = gpt_backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.size()
        device = input_ids.device

        if T > self.gpt.config.block_size:
            raise ValueError(f"Input length {T} exceeds block_size={self.gpt.config.block_size}")

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
class SNLICollator:
    tokenizer: object
    max_length: int

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        premises = [x["premise"] for x in batch]
        hypotheses = [x["hypothesis"] for x in batch]
        labels = [x["label"] for x in batch]

        texts = [f"Premise: {p}\nHypothesis: {h}\nLabel:" for p, h in zip(premises, hypotheses)]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "premises": premises,
            "hypotheses": hypotheses,
        }


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


@torch.no_grad()
def evaluate_and_collect_examples(model, dataloader, device, num_examples=10):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    examples = []

    for batch in tqdm(dataloader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        if len(examples) < num_examples:
            probs = torch.softmax(logits, dim=-1)
            for i in range(labels.size(0)):
                if len(examples) >= num_examples:
                    break
                examples.append(
                    {
                        "premise": batch["premises"][i],
                        "hypothesis": batch["hypotheses"][i],
                        "gold": labels[i].item(),
                        "pred": preds[i].item(),
                        "probs": probs[i].detach().cpu().tolist(),
                    }
                )

    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return avg_loss, acc, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--snli_ckpt", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer_gpt2")
    parser.add_argument("--test_file", type=str, default="data/snli/test.jsonl")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_examples", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size
    if tokenizer.pad_token_id is not None:
        vocab_size = max(vocab_size, tokenizer.pad_token_id + 1)

    gpt = build_gpt1_model(vocab_size=vocab_size, block_size=args.block_size)
    pre_ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    pre_state = pre_ckpt["model_state_dict"] if "model_state_dict" in pre_ckpt else pre_ckpt
    pre_state = maybe_resize_wpe_for_loading(gpt, pre_state)
    gpt.load_state_dict(pre_state, strict=False)

    model = GPTForSequenceClassification(
        gpt_backbone=gpt,
        hidden_size=gpt.config.n_embd,
        num_labels=3,
        dropout=0.1,
    ).to(device)

    snli_ckpt = torch.load(args.snli_ckpt, map_location=device)
    model.load_state_dict(snli_ckpt["model_state_dict"])

    eval_ds = LocalSNLIDataset(args.test_file)
    collator = SNLICollator(tokenizer=tokenizer, max_length=args.max_length)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    loss, acc, examples = evaluate_and_collect_examples(model=model, dataloader=eval_loader, device=device, num_examples=args.num_examples)

    print(f"[SNLI test] loss={loss:.4f} acc={acc:.4f}")
    print("\n===== Sample Predictions =====")
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Premise: {ex['premise']}")
        print(f"Hypothesis: {ex['hypothesis']}")
        print(f"Gold: {LABEL_ID2NAME[ex['gold']]}")
        print(f"Pred: {LABEL_ID2NAME[ex['pred']]}")
        print(
            "Probabilities: "
            f"entailment={ex['probs'][0]:.4f}, "
            f"neutral={ex['probs'][1]:.4f}, "
            f"contradiction={ex['probs'][2]:.4f}"
        )


if __name__ == "__main__":
    main()
