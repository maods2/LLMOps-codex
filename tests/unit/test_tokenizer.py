from __future__ import annotations

from services.data.tokenization.simple_tokenizer import SimpleTokenizer


def test_tokenizer_encode_pad() -> None:
    tok = SimpleTokenizer()
    tok.fit(["hello world"])
    ids = tok.encode("hello unknown", max_length=4)
    assert len(ids) == 4
    assert ids[-1] == tok.pad_id
