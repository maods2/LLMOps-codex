from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    vocab_size: int = Field(default=32000)
    n_layers: int = Field(default=12)
    hidden_size: int = Field(default=1024)
    n_heads: int = Field(default=8)
    context_length: int = Field(default=512)
    dropout: float = Field(default=0.0)


class TrainConfig(BaseModel):
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 50
    max_steps: int = 200
    precision: str = "fp16"
    grad_checkpoint: bool = True
    checkpoint_dir: str = "./artifacts/checkpoints"
