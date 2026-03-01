from __future__ import annotations

from prometheus_client import Gauge


class TrainingMetrics:
    def __init__(self) -> None:
        self.loss = Gauge("train_loss", "training loss")
        self.perplexity = Gauge("train_perplexity", "training perplexity")
        self.learning_rate = Gauge("learning_rate", "optimizer learning rate")
        self.gpu_memory_mb = Gauge("gpu_memory_mb", "GPU memory usage MB")

    def update(self, loss: float, lr: float, gpu_memory_mb: float) -> None:
        self.loss.set(loss)
        self.perplexity.set(float(pow(2.718281828, loss)))
        self.learning_rate.set(lr)
        self.gpu_memory_mb.set(gpu_memory_mb)
