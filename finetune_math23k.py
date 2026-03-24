import os
import re
import ast
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
from datasets import Dataset

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from tokenizer_utils import load_tokenizer
from utils import GPTConfig, build_gpt1_model, count_parameters


def set_seed(seed: int):
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


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_equation(eq):
    eq = str(eq).strip()
    eq = eq.replace(" ", "")
    eq = eq.replace("（", "(").replace("）", ")")
    eq = eq.replace("×", "*").replace("÷", "/")
    eq = eq.replace("%", "/100")

    for prefix in ["x=", "X=", "y=", "Y="]:
        if eq.startswith(prefix):
            eq = eq[len(prefix):]

    return eq

def normalize_answer(ans: Any) -> str:
    ans = normalize_text(ans)
    ans = ans.strip("。.;；,，")
    return ans


def maybe_to_float(x: Any) -> Optional[float]:
    try:
        return float(str(x))
    except Exception:
        return None


ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}
ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def safe_eval_expr(expr: str) -> float:
    """
    只允许数字和 +-*/()**
    """
    expr = normalize_equation(expr)
    if not expr:
        raise ValueError("empty expression")

    # 只保留安全字符
    if re.search(r"[^0-9\.\+\-\*\/\(\)]", expr):
        raise ValueError(f"unsafe chars in expr: {expr}")

    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("invalid constant")
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.BinOp):
            op_type = type(n.op)
            if op_type not in ALLOWED_BINOPS:
                raise ValueError(f"unsupported op: {op_type}")
            return ALLOWED_BINOPS[op_type](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            if op_type not in ALLOWED_UNARYOPS:
                raise ValueError(f"unsupported unary op: {op_type}")
            return ALLOWED_UNARYOPS[op_type](_eval(n.operand))
        raise ValueError(f"unsupported node: {type(n)}")

    return float(_eval(node))


def answers_equal(pred: Any, gold: Any, tol: float = 1e-6) -> bool:
    pred_f = maybe_to_float(pred)
    gold_f = maybe_to_float(gold)

    if pred_f is not None and gold_f is not None:
        return abs(pred_f - gold_f) <= tol

    return normalize_answer(pred) == normalize_answer(gold)


def build_prompt(question: str) -> str:
    return f"题目：{question}\n建模："


def build_target(equation: str) -> str:
    return equation


def find_field(example: Dict, candidates: List[str], required: bool = False) -> str:
    for k in candidates:
        if k in example and example[k] is not None:
            return str(example[k])
    if required:
        raise KeyError(f"Cannot find required field in example. candidates={candidates}, keys={list(example.keys())}")
    return ""


def standardize_example(example: Dict) -> Dict[str, str]:
    question = find_field(
        example,
        ["original_text", "segmented_text", "question", "text", "Problem", "problem"],
        required=True,
    )
    equation = find_field(
        example,
        ["equation", "target", "expr", "formula"],
        required=True,
    )
    answer = find_field(
        example,
        ["ans", "answer", "result"],
        required=False,
    )

    question = normalize_text(question)
    equation = normalize_equation(equation)
    answer = normalize_answer(answer)

    return {
        "question": question,
        "equation": equation,
        "answer": answer,
    }


@dataclass
class Math23KCollator:
    tokenizer: Any
    max_length: int
    add_eos: bool = True

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        eos_token = self.tokenizer.eos_token or ""
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must have pad_token_id or eos_token_id")

        for ex in batch:
            question = ex["question"]
            equation = ex["equation"]

            prompt = build_prompt(question)
            target = build_target(equation)
            full_text = prompt + target + (eos_token if self.add_eos and eos_token else "")

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

            full_ids = full_ids[: self.max_length]
            labels = full_ids.copy()

            prompt_len = min(len(prompt_ids), len(full_ids))
            for i in range(prompt_len):
                labels[i] = -100

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        max_len = max(len(x) for x in input_ids_list)

        for ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            attention_mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len

            input_ids_list[input_ids_list.index(ids[:len(ids)-pad_len] if pad_len > 0 else ids)] = ids
            labels_list[labels_list.index(labels[:len(labels)-pad_len] if pad_len > 0 else labels)] = labels
            attention_mask_list.append(attention_mask)

        # 上面 index 写法不稳，重新构建
        padded_inputs, padded_labels = [], []
        for ids, labels in zip(input_ids_list, labels_list):
            if len(ids) < max_len:
                ids = ids + [pad_id] * (max_len - len(ids))
            if len(labels) < max_len:
                labels = labels + [-100] * (max_len - len(labels))
            padded_inputs.append(ids)
            padded_labels.append(labels)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        }


class StableMath23KCollator:
    def __init__(self, tokenizer, max_length: int, add_eos: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos = add_eos

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        eos_token = self.tokenizer.eos_token or ""
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must have pad_token_id or eos_token_id")

        input_ids_list = []
        labels_list = []
        masks_list = []

        for ex in batch:
            prompt = build_prompt(ex["question"])
            target = build_target(ex["equation"])
            full_text = prompt + target + (eos_token if self.add_eos and eos_token else "")

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"][: self.max_length]

            labels = full_ids.copy()
            prompt_len = min(len(prompt_ids), len(full_ids))
            for i in range(prompt_len):
                labels[i] = -100

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        max_len = max(len(x) for x in input_ids_list)

        padded_inputs, padded_labels, padded_masks = [], [], []
        for ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_inputs.append(ids + [pad_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            padded_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }


def masked_lm_loss(model, input_ids: torch.Tensor, labels: torch.Tensor):
    logits, _ = model(input_ids)
    vocab_size = logits.size(-1)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return logits, loss


import os
import json
from datasets import Dataset

def read_json_or_jsonl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    # 兼容 UTF-8 BOM
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"空文件: {path}")

    # 情况1：标准 JSON 数组 / JSON 对象
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # 如果整个文件是一个 dict，就尝试取里面可能的列表字段
                for key in ["data", "items", "questions"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
        except json.JSONDecodeError:
            pass

    # 情况2：按 JSONL 逐行读取
    data = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: {path}, 第 {lineno} 行, 内容={line[:200]}") from e

    if not data:
        raise ValueError(f"无法解析文件: {path}")

    return data

def read_math23k_objects(path: str):
    """
    读取这种格式：
    { ... }
    { ... }
    { ... }

    即多个 JSON 对象连续排列，而不是标准 JSON 数组。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"空文件: {path}")

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    data = []

    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break

        obj, end = decoder.raw_decode(text, idx)
        data.append(obj)
        idx = end

    if not data:
        raise ValueError(f"无法解析 Math23K 文件: {path}")

    return data


def load_math23k_dataset(data_dir: str, val_ratio: float = 0.1, seed: int = 42):
    train_path = os.path.join(data_dir, "math23k_train.json")
    test_path = os.path.join(data_dir, "math23k_test.json")

    train_data = read_math23k_objects(train_path)
    test_data = read_math23k_objects(test_path)

    train_full = Dataset.from_list(train_data)
    test_ds = Dataset.from_list(test_data)

    split = train_full.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    valid_ds = split["test"]

    return train_ds, valid_ds, test_ds

def preprocess_dataset_split(ds):
    def _map_fn(ex):
        std = standardize_example(ex)
        return {
            "question": std["question"],
            "equation": std["equation"],
            "answer": std["answer"],
        }
    ds = ds.map(_map_fn)
    return ds


@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    for batch in tqdm(dataloader, desc="Validating", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = masked_lm_loss(model, input_ids, labels)
        total_loss += loss.item()
        total_steps += 1
    return total_loss / max(1, total_steps)


def decode_generated_equation(
    tokenizer,
    generated_ids: torch.Tensor,
    prompt_len: int,
    eos_token: Optional[str] = None,
) -> str:
    gen_ids = generated_ids[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = text.split("\n")[0]
    if eos_token and eos_token in text:
        text = text.split(eos_token)[0]
    text = normalize_equation(text)
    return text


@torch.no_grad()
def predict_one(
    model,
    tokenizer,
    question: str,
    device,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    model.eval()
    prompt = build_prompt(question)
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    out = model.generate(
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    pred_eq = decode_generated_equation(
        tokenizer=tokenizer,
        generated_ids=out[0].tolist(),
        prompt_len=input_ids.size(1),
        eos_token=tokenizer.eos_token,
    )
    return pred_eq


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    dataset,
    device,
    output_path: Optional[str] = None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    limit: Optional[int] = None,
):
    model.eval()

    total = 0
    eq_exact = 0
    ans_acc = 0
    valid_rate = 0
    rows = []

    iterator = dataset
    if limit is not None:
        iterator = dataset.select(range(min(limit, len(dataset))))

    for ex in tqdm(iterator, desc="Generating"):
        question = ex["question"]
        gold_eq = normalize_equation(ex["equation"])
        gold_ans = normalize_answer(ex.get("answer", ""))

        pred_eq = predict_one(
            model=model,
            tokenizer=tokenizer,
            question=question,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        pred_ans = ""
        pred_valid = False

        try:
            val = safe_eval_expr(pred_eq)
            pred_valid = True
            pred_ans = str(val)
        except Exception:
            pred_valid = False
            pred_ans = ""

        gold_eval_ans = gold_ans
        if not gold_eval_ans:
            try:
                gold_eval_ans = str(safe_eval_expr(gold_eq))
            except Exception:
                gold_eval_ans = ""

        eq_match = (pred_eq == gold_eq)
        ans_match = answers_equal(pred_ans, gold_eval_ans)

        total += 1
        eq_exact += int(eq_match)
        ans_acc += int(ans_match)
        valid_rate += int(pred_valid)

        rows.append({
            "question": question,
            "gold_equation": gold_eq,
            "pred_equation": pred_eq,
            "gold_answer": gold_eval_ans,
            "pred_answer": pred_ans,
            "pred_valid": pred_valid,
            "equation_exact_match": eq_match,
            "answer_match": ans_match,
        })

    metrics = {
        "total": total,
        "equation_exact_match": eq_exact / max(1, total),
        "answer_accuracy": ans_acc / max(1, total),
        "valid_expression_rate": valid_rate / max(1, total),
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics, rows


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, path, args_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "args": args_dict,
        },
        path,
    )


def load_checkpoint_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return ckpt


def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0, log_interval=100):
    model.train()
    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = masked_lm_loss(model, input_ids, labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_steps += 1

        if total_steps % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / max(1, total_steps)


def parse_args():
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument("--data_dir", type=str, default="./data/math23k")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer_gpt2")
    parser.add_argument("--save_dir", type=str, default="checkpoints/math23k")
    parser.add_argument("--prediction_file", type=str, default="outputs/math23k/predictions.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)

    # 预训练参数
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="预训练 GPT checkpoint 路径")
    parser.add_argument("--resume", type=str, default=None)

    # 模型
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_inner", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.002)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # 生成
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--eval_limit", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.prediction_file), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    tokenizer = load_tokenizer(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("tokenizer 没有 pad_token，也没有 eos_token，无法继续")

    vocab_size = len(tokenizer)

    # 数据
    train_ds, valid_ds, test_ds = load_math23k_dataset(
		args.data_dir,
		val_ratio=args.val_ratio,
		seed=args.split_seed,
	)
    train_ds = preprocess_dataset_split(train_ds)
    valid_ds = preprocess_dataset_split(valid_ds)
    test_ds = preprocess_dataset_split(test_ds)

    collator = StableMath23KCollator(tokenizer=tokenizer, max_length=args.max_length)

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

    # 模型
    model_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_inner=args.n_inner,
        embd_pdrop=args.dropout,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
    )
    model = build_gpt1_model(model_cfg).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # 载入预训练权重
    print(f"Loading pretrained checkpoint from: {args.pretrained_ckpt}")
    load_checkpoint_into_model(model, args.pretrained_ckpt, device)

    # resume
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = create_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=max(1, total_steps))

    start_epoch = 0
    best_valid_eq = -1.0

    if args.resume is not None:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_valid_eq = ckpt.get("best_metric", -1.0)

    # 训练
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_clip=args.grad_clip,
            log_interval=100,
        )

        valid_loss = evaluate_loss(model, valid_loader, device)
        valid_metrics, _ = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            dataset=valid_ds,
            device=device,
            output_path=None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            limit=args.eval_limit,
        )

        print(
            f"[Epoch {epoch+1}] "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_eq_acc={valid_metrics['equation_exact_match']:.4f} "
            f"valid_ans_acc={valid_metrics['answer_accuracy']:.4f} "
            f"valid_valid_rate={valid_metrics['valid_expression_rate']:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_valid_eq,
            path=os.path.join(args.save_dir, "last.pt"),
            args_dict=vars(args),
        )

        if valid_metrics["equation_exact_match"] > best_valid_eq:
            best_valid_eq = valid_metrics["equation_exact_match"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_valid_eq,
                path=os.path.join(args.save_dir, "best.pt"),
                args_dict=vars(args),
            )
            print(f"New best model saved. valid_eq_acc={best_valid_eq:.4f}")

    # 最终测试
    best_path = os.path.join(args.save_dir, "best.pt")
    print(f"\nLoading best checkpoint for final test: {best_path}")
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=False)

    test_metrics, rows = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        dataset=test_ds,
        device=device,
        output_path=args.prediction_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        limit=args.eval_limit,
    )

    print("\n[Test Metrics]")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"\nPrediction file saved to: {args.prediction_file}")

    print("\n[Examples: gold vs pred]")
    for row in rows[:10]:
        print("=" * 80)
        print(f"题目: {row['question']}")
        print(f"金标方程: {row['gold_equation']}")
        print(f"预测方程: {row['pred_equation']}")
        print(f"金标答案: {row['gold_answer']}")
        print(f"预测答案: {row['pred_answer']}")
        print(f"方程完全匹配: {row['equation_exact_match']}")
        print(f"答案匹配: {row['answer_match']}")
        print(f"表达式是否合法: {row['pred_valid']}")


if __name__ == "__main__":
    main()