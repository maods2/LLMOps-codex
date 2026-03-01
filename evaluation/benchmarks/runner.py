from __future__ import annotations

from evaluation.generation_quality.generation import token_accuracy
from evaluation.perplexity.metrics import perplexity


def run_eval(avg_loss: float, predictions: list[int], labels: list[int]) -> dict[str, float]:
    return {
        "perplexity": perplexity(avg_loss),
        "token_accuracy": token_accuracy(predictions, labels),
    }
