from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from services.training.core_model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


def _rope(freqs: Tensor, x: Tensor) -> Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = freqs.cos(), freqs.sin()
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.hidden_size // cfg.n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        idx = torch.arange(t, device=x.device, dtype=x.dtype)
        freqs = torch.outer(idx, torch.pow(torch.tensor(10000.0, device=x.device, dtype=x.dtype), -torch.arange(0, self.head_dim, 2, device=x.device, dtype=x.dtype) / self.head_dim))
        q, k = _rope(freqs[None, None, :, :], q), _rope(freqs[None, None, :, :], k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal, float("-inf"))
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        probs = torch.softmax(attn, dim=-1)
        out = probs @ v
        out = out.transpose(1, 2).reshape(b, t, c)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_size * 4)
        self.fc1 = nn.Linear(cfg.hidden_size, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.hidden_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn = SelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: Tensor, labels: Tensor | None = None, attention_mask: Tensor | None = None) -> ModelOutput:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x, attention_mask)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )
        return ModelOutput(logits=logits, loss=loss)

    def estimate_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
