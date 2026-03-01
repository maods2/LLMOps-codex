from __future__ import annotations

import torch

from services.training.dpo.train import dpo_loss


def test_dpo_step() -> None:
    loss = dpo_loss(
        torch.tensor([0.2, 0.3]),
        torch.tensor([0.1, 0.05]),
        torch.tensor([0.15, 0.2]),
        torch.tensor([0.1, 0.1]),
    )
    assert float(loss) > 0
