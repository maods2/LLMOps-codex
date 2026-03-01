from __future__ import annotations

from services.training.core_model.config import ModelConfig, TrainConfig
from services.training.pretrain.train import run_pretrain


def test_pretrain_runs_with_partial_accumulation() -> None:
    model = run_pretrain(
        ["tiny corpus"] * 2,
        ModelConfig(vocab_size=64, n_layers=1, hidden_size=32, n_heads=4, context_length=8),
        TrainConfig(max_steps=1, gradient_accumulation_steps=8),
    )
    assert model.estimate_params() > 0
