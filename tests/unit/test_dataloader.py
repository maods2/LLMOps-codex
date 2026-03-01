from __future__ import annotations

from services.data.preprocessing.pipeline import TokenizedDataset
from services.data.tokenization.simple_tokenizer import SimpleTokenizer


def test_dataset_item() -> None:
    tok = SimpleTokenizer()
    tok.fit(["a b c"])
    ds = TokenizedDataset(["a b"], tok, seq_len=4)
    item = ds[0]
    assert item["input_ids"].shape[0] == 4
    assert item["labels"].shape[0] == 4
