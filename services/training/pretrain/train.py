from __future__ import annotations


import torch
from torch.utils.data import DataLoader

from services.data.preprocessing.pipeline import TokenizedDataset
from services.data.tokenization.simple_tokenizer import SimpleTokenizer
from services.training.core_model.config import ModelConfig, TrainConfig
from services.training.core_model.model import DecoderOnlyTransformer
from services.training.core_model.trainer import Trainer


def run_pretrain(samples: list[str], model_cfg: ModelConfig, train_cfg: TrainConfig) -> DecoderOnlyTransformer:
    tokenizer = SimpleTokenizer()
    tokenizer.fit(samples)
    dataset = TokenizedDataset(samples, tokenizer, model_cfg.context_length)
    loader = DataLoader(dataset, batch_size=train_cfg.micro_batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(model_cfg)
    trainer = Trainer(model, train_cfg, device)
    step = 0
    accum_steps = 0
    for batch in loader:
        loss = trainer.train_step(batch)
        accum_steps += 1
        if accum_steps % train_cfg.gradient_accumulation_steps == 0:
            trainer.optimizer_step()
        step += 1
        if step % 50 == 0:
            trainer.save_checkpoint(step)
        if step >= train_cfg.max_steps:
            break
        _ = loss

    if accum_steps % train_cfg.gradient_accumulation_steps != 0:
        trainer.optimizer_step()
    return model


if __name__ == "__main__":
    tiny = ["hello world", "small dataset for local pretraining"] * 16
    run_pretrain(tiny, ModelConfig(vocab_size=512), TrainConfig(max_steps=4))
