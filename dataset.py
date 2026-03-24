import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer_utils import load_tokenizer


class LMDataset(Dataset):
    def __init__(self, blocks: List[List[int]]):
        self.blocks = [torch.tensor(x, dtype=torch.long) for x in blocks]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.blocks[idx]}


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _extract_text_from_json_obj(obj, text_column: str) -> Optional[str]:
    if isinstance(obj, dict):
        if text_column in obj and obj[text_column] is not None:
            return clean_text(str(obj[text_column]))
        for key in ["text", "content", "sentence", "article"]:
            if key in obj and obj[key] is not None:
                return clean_text(str(obj[key]))
    elif isinstance(obj, str):
        return clean_text(obj)
    return None


def load_texts_from_local_dir(
    data_dir: str,
    max_texts: Optional[int] = None,
    text_column: str = "text",
) -> List[str]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Local data dir not found: {data_dir}")

    texts: List[str] = []

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            text = clean_text(file_path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                texts.append(text)
        elif suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = _extract_text_from_json_obj(obj, text_column)
                    if text:
                        texts.append(text)
                    if max_texts is not None and len(texts) >= max_texts:
                        break
        elif suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                for row in obj:
                    text = _extract_text_from_json_obj(row, text_column)
                    if text:
                        texts.append(text)
                    if max_texts is not None and len(texts) >= max_texts:
                        break
            else:
                text = _extract_text_from_json_obj(obj, text_column)
                if text:
                    texts.append(text)

        if max_texts is not None and len(texts) >= max_texts:
            break

    if not texts:
        raise FileNotFoundError(
            f"No usable local texts found in {data_dir}. Supported files: .txt/.json/.jsonl"
        )
    return texts[:max_texts] if max_texts is not None else texts


def tokenize_and_chunk_texts(
    texts: List[str],
    tokenizer_name_or_path: str,
    block_size: int,
    cache_dir: Optional[str] = None,
    add_eos_between_texts: bool = True,
) -> List[List[int]]:
    tokenizer = load_tokenizer(tokenizer_name_or_path, cache_dir=cache_dir)
    all_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        all_ids.extend(ids)
        if add_eos_between_texts and eos_id is not None:
            all_ids.append(eos_id)

    num_full_blocks = len(all_ids) // block_size
    usable_ids = all_ids[: num_full_blocks * block_size]
    blocks = [usable_ids[i : i + block_size] for i in range(0, len(usable_ids), block_size)]
    if not blocks:
        raise ValueError("No blocks created. Increase max_texts or reduce block_size.")
    return blocks


def split_blocks(
    blocks: List[List[int]],
    valid_ratio: float = 0.01,
    seed: int = 42,
) -> Tuple[List[List[int]], List[List[int]]]:
    random.Random(seed).shuffle(blocks)
    n_valid = max(1, int(len(blocks) * valid_ratio))
    valid_blocks = blocks[:n_valid]
    train_blocks = blocks[n_valid:]
    if not train_blocks:
        raise ValueError("Training split is empty. Reduce valid_ratio or use more data.")
    return train_blocks, valid_blocks


def build_dataloaders(config: dict):
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    source = data_cfg.get("source", "local")
    if source != "local":
        raise ValueError(
            "This project is configured for offline/local data loading only. "
            "Please set data.source=local and point data.local_text_dir to ./data/..."
        )

    texts = load_texts_from_local_dir(
        data_dir=data_cfg["local_text_dir"],
        max_texts=data_cfg.get("max_texts"),
        text_column=data_cfg.get("text_column", "text"),
    )

    blocks = tokenize_and_chunk_texts(
        texts=texts,
        tokenizer_name_or_path=data_cfg.get("tokenizer_path", "data/tokenizer_gpt2"),
        block_size=model_cfg["block_size"],
        cache_dir=data_cfg.get("cache_dir"),
        add_eos_between_texts=data_cfg.get("add_eos_between_texts", True),
    )

    train_blocks, valid_blocks = split_blocks(
        blocks=blocks,
        valid_ratio=data_cfg.get("valid_ratio", 0.01),
        seed=train_cfg.get("seed", 42),
    )

    train_dataset = LMDataset(train_blocks)
    valid_dataset = LMDataset(valid_blocks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory", False),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=train_cfg["eval_batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory", False),
    )
    return train_loader, valid_loader
