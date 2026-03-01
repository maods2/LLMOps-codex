from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from services.data.tokenization.simple_tokenizer import SimpleTokenizer


class TokenizedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[str], tokenizer: SimpleTokenizer, seq_len: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.tokenizer.encode(self.samples[idx], self.seq_len)
        x = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}


def cache_dataset(samples: list[str], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(samples), encoding="utf-8")


def load_cached_dataset(cache_path: Path) -> list[str]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return [str(item) for item in payload]
