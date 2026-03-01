from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PreferenceBatch:
    prompt: torch.Tensor
    chosen: torch.Tensor
    rejected: torch.Tensor


def dpo_loss(policy_chosen_logp: torch.Tensor, policy_rejected_logp: torch.Tensor, ref_chosen_logp: torch.Tensor, ref_rejected_logp: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratios - ref_logratios)
    return -torch.nn.functional.logsigmoid(logits).mean()
