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


ID2LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LocalRACEDataset(Dataset):
    """
    读取 jsonl:
    {"article": "...", "question": "...", "options": ["...", "...", "...", "..."], "answer": "A"}
    """
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"RACE file not found: {path}")

        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                self.samples.append(self._normalize(ex))

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {path}")

    def _normalize(self, ex: Dict):
        article = ex["article"]
        question = ex["question"]
        options = ex["options"]
        answer = ex["answer"]

        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("Each sample must contain exactly 4 options.")

        if isinstance(answer, str):
            answer = answer.strip().upper()
            if answer not in ["A", "B", "C", "D"]:
                raise ValueError(f"Invalid answer: {answer}")
            label = ord(answer) - ord("A")
        elif isinstance(answer, int):
            label = answer
        else:
            raise ValueError("answer must be A/B/C/D or int")

        return {
            "article": article,
            "question": question,
            "options": options,
            "label": label,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GPTForMultipleChoice(nn.Module):
    def __init__(self, gpt_backbone: nn.Module, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.gpt = gpt_backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids:      (B, C, T)
        attention_mask: (B, C, T)
        labels:         (B,)
        """
        B, C, T = input_ids.size()
        device = input_ids.device

        if T > self.gpt.config.block_size:
            raise ValueError(
                f"Input length {T} exceeds block_size={self.gpt.config.block_size}"
            )

        flat_input_ids = input_ids.view(B * C, T)
        flat_attention_mask = attention_mask.view(B * C, T) if attention_mask is not None else None

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.gpt.wte(flat_input_ids)
        pos_emb = self.gpt.wpe(pos)
        x = self.gpt.drop(tok_emb + pos_emb)

        for block in self.gpt.h:
            x = block(x)

        x = self.gpt.ln_f(x)

        if flat_attention_mask is None:
            pooled = x[:, -1, :]
        else:
            lengths = flat_attention_mask.sum(dim=1) - 1
            pooled = x[torch.arange(B * C, device=device), lengths]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).view(B, C)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


@dataclass
class RACECollator:
    tokenizer: object
    max_length: int
    num_choices: int = 4

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = []
        labels = []
        articles = []
        questions = []
        options_list = []

        for ex in batch:
            article = ex["article"]
            question = ex["question"]
            options = ex["options"]
            label = ex["label"]

            for op in options:
                text = (
                    f"Article: {article}\n"
                    f"Question: {question}\n"
                    f"Option: {op}\n"
                    f"Answer:"
                )
                texts.append(text)

            labels.append(label)
            articles.append(article)
            questions.append(question)
            options_list.append(options)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        B = len(batch)
        T = enc["input_ids"].size(1)

        return {
            "input_ids": enc["input_ids"].view(B, self.num_choices, T),
            "attention_mask": enc["attention_mask"].view(B, self.num_choices, T),
            "labels": torch.tensor(labels, dtype=torch.long),
            "articles": articles,
            "questions": questions,
            "options_list": options_list,
        }


def maybe_resize_wpe_for_loading(gpt_model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """
    兼容 block_size 不一致时的 wpe.weight 加载。
    例如 checkpoint 是 128，新模型设成 256。
    """
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
def evaluate(model, dataloader, device, num_examples=10):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    examples = []

    for batch in tqdm(dataloader, desc="Testing"):
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
        probs = torch.softmax(logits, dim=-1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        if len(examples) < num_examples:
            for i in range(labels.size(0)):
                if len(examples) >= num_examples:
                    break
                examples.append(
                    {
                        "article": batch["articles"][i],
                        "question": batch["questions"][i],
                        "options": batch["options_list"][i],
                        "gold": labels[i].item(),
                        "pred": preds[i].item(),
                        "probs": probs[i].detach().cpu().tolist(),
                    }
                )

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="预训练 GPT checkpoint")
    parser.add_argument("--race_ckpt", type=str, required=True, help="RACE 微调后的 best.pt")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer_gpt2")
    parser.add_argument("--test_file", type=str, default="data/race/test.jsonl")

    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
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

    model = GPTForMultipleChoice(
        gpt_backbone=gpt,
        hidden_size=gpt.config.n_embd,
        dropout=0.1,
    ).to(device)

    race_ckpt = torch.load(args.race_ckpt, map_location="cpu")
    race_state = race_ckpt["model_state_dict"] if "model_state_dict" in race_ckpt else race_ckpt
    model.load_state_dict(race_state, strict=False)

    test_ds = LocalRACEDataset(args.test_file)
    collator = RACECollator(tokenizer=tokenizer, max_length=args.max_length)

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
    )

    print(f"\n[RACE Test] loss={test_loss:.4f} acc={test_acc:.4f}")

    print("\n===== Sample Predictions =====")
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {ex['question']}")
        for j, op in enumerate(ex["options"]):
            print(f"{ID2LABEL[j]}. {op}")
        print(f"Gold: {ID2LABEL[ex['gold']]}")
        print(f"Pred: {ID2LABEL[ex['pred']]}")
        print(
            "Probabilities: "
            f"A={ex['probs'][0]:.4f}, "
            f"B={ex['probs'][1]:.4f}, "
            f"C={ex['probs'][2]:.4f}, "
            f"D={ex['probs'][3]:.4f}"
        )

        article_preview = ex["article"][:300].replace("\n", " ")
        print(f"Article preview: {article_preview}...")


if __name__ == "__main__":
    main()