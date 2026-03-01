from __future__ import annotations

from pathlib import Path
from typing import Any


def apply_lora(model: Any, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> Any:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for LoRA") from exc

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def save_adapter(model: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))


def merge_adapter(model: Any) -> Any:
    return model.merge_and_unload()
