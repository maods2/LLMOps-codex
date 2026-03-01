from __future__ import annotations

import torch

from services.training.sft.train import build_sft_labels


def test_masked_loss_labels() -> None:
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[False, True, True]])
    labels = build_sft_labels(ids, mask)
    assert labels[0, 0].item() == -100
    assert labels[0, 1].item() == 2
