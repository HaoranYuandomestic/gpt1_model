import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizer_utils import load_tokenizer
from utils import build_gpt1_model


LABEL_ID2NAME = {
    0: "negative",
    1: "positive",
}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LocalSST2Dataset(Dataset):
    """
    读取 jsonl:
    {"sentence": "...", "label": 0/1}
    或
    {"sentence": "...", "label": null}
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
def evaluate(model, dataloader, device, num_examples=10, has_label=True):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    examples = []

    for batch in tqdm(dataloader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if has_label:
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
            )
            labels = None

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        if has_label:
            loss = outputs["loss"]
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

        if len(examples) < num_examples:
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                if len(examples) >= num_examples:
                    break
                ex = {
                    "sentence": batch["sentences"][i],
                    "pred": preds[i].item(),
                    "probs": probs[i].detach().cpu().tolist(),
                }
                if has_label:
                    ex["gold"] = labels[i].item()
                examples.append(ex)

    if has_label:
        avg_loss = total_loss / max(1, total_count)
        acc = total_correct / max(1, total_count)
    else:
        avg_loss = None
        acc = None

    return avg_loss, acc, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--sst2_ckpt", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer_gpt2")
    parser.add_argument("--test_file", type=str, default="data/sst2/validation.jsonl")

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--has_label", action="store_true")
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
        num_labels=2,
        dropout=0.1,
    ).to(device)

    sst2_ckpt = torch.load(args.sst2_ckpt, map_location="cpu")
    sst2_state = sst2_ckpt["model_state_dict"] if "model_state_dict" in sst2_ckpt else sst2_ckpt
    model.load_state_dict(sst2_state, strict=False)

    test_ds = LocalSST2Dataset(args.test_file, require_label=args.has_label)
    collator = SST2Collator(tokenizer=tokenizer, max_length=args.max_length)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    test_loss, test_acc, examples = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        num_examples=args.num_examples,
        has_label=args.has_label,
    )

    if args.has_label:
        print(f"\n[SST-2 Test] loss={test_loss:.4f} acc={test_acc:.4f}")
    else:
        print("\n[SST-2 Predict] unlabeled split, so only predictions are shown.")

    print("\n===== Sample Predictions =====")
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Sentence: {ex['sentence']}")
        if "gold" in ex:
            print(f"Gold: {LABEL_ID2NAME[ex['gold']]}")
        print(f"Pred: {LABEL_ID2NAME[ex['pred']]}")
        print(
            "Probabilities: "
            f"negative={ex['probs'][0]:.4f}, "
            f"positive={ex['probs'][1]:.4f}"
        )


if __name__ == "__main__":
    main()