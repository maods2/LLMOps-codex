from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def stream_jsonl(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def split_indices(size: int, split: tuple[float, float, float]) -> tuple[range, range, range]:
    train_end = int(size * split[0])
    val_end = train_end + int(size * split[1])
    return range(0, train_end), range(train_end, val_end), range(val_end, size)
