from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from observability.logging.logger import get_logger
from observability.metrics.tracker import TrainingMetrics
from services.training.core_model.config import TrainConfig
from services.training.core_model.model import DecoderOnlyTransformer

LOGGER = get_logger(__name__)


@dataclass
class TrainState:
    step: int = 0
    best_loss: float = float("inf")


class Trainer:
    def __init__(self, model: DecoderOnlyTransformer, cfg: TrainConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.max_steps, eta_min=cfg.min_lr)
        self.metrics = TrainingMetrics()
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        self.model.train()
        ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        out = self.model(ids, labels=labels)
        if out.loss is None:
            raise RuntimeError("Model did not return loss")
        loss = out.loss / self.cfg.gradient_accumulation_steps
        loss.backward()
        return float(loss.item())

    def optimizer_step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    def save_checkpoint(self, step: int) -> Path:
        path = Path(self.cfg.checkpoint_dir) / f"step-{step}.pt"
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)
        LOGGER.info("checkpoint_saved", extra={"path": str(path)})
        return path
