from __future__ import annotations

from types import SimpleNamespace

from services.training.lora.train import merge_adapter


class DummyModel(SimpleNamespace):
    def merge_and_unload(self) -> str:
        return "merged"


def test_model_merge() -> None:
    model = DummyModel()
    assert merge_adapter(model) == "merged"
