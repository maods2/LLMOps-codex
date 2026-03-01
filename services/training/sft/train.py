from __future__ import annotations

import torch


def build_sft_labels(input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    labels[~response_mask] = -100
    return labels
