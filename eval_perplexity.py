import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import shift_labels_for_causal_lm


@torch.no_grad()
def compute_token_level_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    ignore_index: int = -100,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for step, batch in enumerate(dataloader):
        if max_batches is not None and step >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = shift_labels_for_causal_lm(input_ids, ignore_index=ignore_index)
        logits, _ = model(input_ids)

        vocab_size = logits.size(-1)
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        token_losses = F.cross_entropy(
            logits_flat,
            labels_flat,
            reduction="none",
            ignore_index=ignore_index,
        )

        valid_mask = labels_flat != ignore_index
        valid_losses = token_losses[valid_mask]
        total_nll += valid_losses.sum().item()
        total_tokens += valid_mask.sum().item()

    if total_tokens == 0:
        raise ValueError("No valid tokens found for perplexity computation.")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return {
        "avg_nll": avg_nll,
        "perplexity": ppl,
        "total_tokens": total_tokens,
    }
