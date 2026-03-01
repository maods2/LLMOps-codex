from __future__ import annotations

import torch

from services.training.core_model.config import ModelConfig
from services.training.core_model.model import DecoderOnlyTransformer


def test_model_forward() -> None:
    model = DecoderOnlyTransformer(ModelConfig(vocab_size=64, n_layers=2, hidden_size=64, n_heads=8))
    ids = torch.randint(0, 64, (2, 16))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 16, 64)
    assert out.loss is not None
