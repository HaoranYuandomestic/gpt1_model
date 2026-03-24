from pathlib import Path
from typing import Iterable, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_LOCAL_TOKENIZER_PATH = "data/tokenizer_gpt2"


def load_tokenizer(
    name_or_path: str = DEFAULT_LOCAL_TOKENIZER_PATH,
    cache_dir: Optional[str] = None,
) -> PreTrainedTokenizerBase:
    """
    优先离线加载本地 tokenizer。
    默认路径：data/tokenizer_gpt2
    """
    if not Path(name_or_path).exists():
        raise FileNotFoundError(
            f"Tokenizer path not found: {name_or_path}\n"
            f"请确认你的 data 目录下存在 tokenizer_gpt2。"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_tokenizer(tokenizer: PreTrainedTokenizerBase, save_dir: str) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)


def encode_text(tokenizer: PreTrainedTokenizerBase, text: str):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def decode_ids(tokenizer: PreTrainedTokenizerBase, ids):
    return tokenizer.decode(ids, skip_special_tokens=True)


def iter_text_files(path: str) -> Iterable[str]:
    for file_path in sorted(Path(path).glob("*.txt")):
        yield file_path.read_text(encoding="utf-8", errors="ignore")