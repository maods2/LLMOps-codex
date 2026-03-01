from __future__ import annotations

from typing import Protocol

try:
    from prometheus_client import Gauge
except ImportError:  # pragma: no cover - exercised only in minimal environments
    class Gauge:  # type: ignore[no-redef]
        def __init__(self, _name: str, _doc: str) -> None:
            self.value = 0.0

        def set(self, value: float) -> None:
            self.value = value


class _GaugeLike(Protocol):
    def set(self, value: float) -> None:
        ...


_TRAIN_LOSS_GAUGE = Gauge("train_loss", "training loss")
_TRAIN_PERPLEXITY_GAUGE = Gauge("train_perplexity", "training perplexity")
_LEARNING_RATE_GAUGE = Gauge("learning_rate", "optimizer learning rate")
_GPU_MEMORY_GAUGE = Gauge("gpu_memory_mb", "GPU memory usage MB")


class TrainingMetrics:
    def __init__(self) -> None:
        # Reuse process-wide gauge instances to avoid duplicate metric registration
        # when multiple Trainer objects are created during tests.
        self.loss: _GaugeLike = _TRAIN_LOSS_GAUGE
        self.perplexity: _GaugeLike = _TRAIN_PERPLEXITY_GAUGE
        self.learning_rate: _GaugeLike = _LEARNING_RATE_GAUGE
        self.gpu_memory_mb: _GaugeLike = _GPU_MEMORY_GAUGE

    def update(self, loss: float, lr: float, gpu_memory_mb: float) -> None:
        self.loss.set(loss)
        self.perplexity.set(float(pow(2.718281828, loss)))
        self.learning_rate.set(lr)
        self.gpu_memory_mb.set(gpu_memory_mb)
