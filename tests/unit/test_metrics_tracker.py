from __future__ import annotations

from observability.metrics.tracker import TrainingMetrics


def test_training_metrics_can_be_initialized_multiple_times() -> None:
    first = TrainingMetrics()
    second = TrainingMetrics()
    assert first.loss is second.loss
    first.update(loss=1.0, lr=1e-3, gpu_memory_mb=100.0)
