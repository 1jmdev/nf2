from __future__ import annotations

from itertools import islice

import torch
from datasets import load_dataset


def load_text_samples(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str | None = "sample-10BT",
    split: str = "train",
    text_column: str = "text",
    target_bytes: int = 50_000_000,
    streaming: bool = True,
) -> list[str]:
    """Load roughly target_bytes of text from a Hugging Face dataset."""

    kwargs = {"split": split, "streaming": streaming}
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)
    texts: list[str] = []
    total = 0
    for row in ds:
        text = row.get(text_column)
        if not text:
            continue
        texts.append(text)
        total += len(text.encode("utf-8", errors="ignore"))
        if total >= target_bytes:
            break
    if not texts:
        raise RuntimeError(f"No text loaded from {dataset_name}; check dataset_config/text_column")
    return texts


def token_batches(
    tokenizer,
    texts: list[str],
    sequence_length: int = 1024,
    batch_size: int = 1,
    max_batches: int | None = None,
    device: str | torch.device = "cpu",
):
    """Yield contiguous token batches without building one giant token sequence."""

    tokens_per_batch = sequence_length * batch_size
    buffer: list[int] = []
    yielded = 0
    for text in texts:
        buffer.extend(tokenizer(text, add_special_tokens=False).input_ids)
        while len(buffer) >= tokens_per_batch:
            chunk = buffer[:tokens_per_batch]
            del buffer[:tokens_per_batch]
            input_ids = torch.tensor(chunk, dtype=torch.long).view(batch_size, sequence_length).to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            yield {"input_ids": input_ids, "attention_mask": attention_mask}
            yielded += 1
            if max_batches is not None and yielded >= max_batches:
                return

    if buffer and max_batches is None:
        needed = tokens_per_batch - len(buffer)
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        buffer.extend([pad_id] * needed)
        input_ids = torch.tensor(buffer, dtype=torch.long).view(batch_size, sequence_length).to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        yield {"input_ids": input_ids, "attention_mask": attention_mask}


def preview_dataset_rows(dataset_name: str, dataset_config: str | None = None, split: str = "train", limit: int = 3) -> list[dict]:
    kwargs = {"split": split, "streaming": True}
    ds = load_dataset(dataset_name, dataset_config, **kwargs) if dataset_config else load_dataset(dataset_name, **kwargs)
    return list(islice(ds, limit))
