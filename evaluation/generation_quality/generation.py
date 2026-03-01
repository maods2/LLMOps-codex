from __future__ import annotations


def token_accuracy(preds: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    hits = sum(int(a == b) for a, b in zip(preds, labels))
    return hits / len(labels)
