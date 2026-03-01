from __future__ import annotations

from collections.abc import Mapping


def log_metrics(metrics: Mapping[str, float]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
