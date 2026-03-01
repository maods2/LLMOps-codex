from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimpleTokenizer:
    vocab: dict[str, int] = field(default_factory=lambda: {"<pad>": 0, "<unk>": 1})

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            for tok in text.split():
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in text.split()]
        ids = ids[:max_length]
        return ids + [self.pad_id] * (max_length - len(ids))
