from __future__ import annotations

from typing import Any


class InferenceEngine:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def generate(self, prompt: str, max_tokens: int = 32) -> str:
        return f"{prompt} ::generated::{max_tokens}"


def build_vllm_engine(model_name: str) -> Any:
    try:
        from vllm import LLM
    except ImportError:
        return InferenceEngine(model_name)
    return LLM(model=model_name)
