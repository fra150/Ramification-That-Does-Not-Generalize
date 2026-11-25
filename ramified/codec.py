"""Exact document codec.

The tokenizer is trained on words, so whitespace is not in its terminal
alphabet. The original text codec dropped spaces and used EOW as a separator,
which is exact for word lists but not for documents with punctuation:
"hello, world" decoded as "hello, world" only by accident, and
"hello . world" lost the space after the period.

The chunk codec fixes this without adding tokens: split the normalised text
into space-separated chunks, encode each chunk with EOW, decode EOW as a
space. Token count is unchanged; document round-trip becomes exact on
whitespace-normalised text.
"""

from __future__ import annotations

import re

from .bpe import EOW


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_chunks(text: str, tok) -> list[int]:
    text = normalize(text)
    ids: list[int] = []
    for chunk in text.split(" "):
        ids.extend(tok.encode(chunk, eow=True))
    return ids


def detokenize_chunks(ids, tok) -> str:
    return tok.decode(ids).replace(EOW, " ").strip()


def roundtrip_ok(text: str, tok) -> bool:
    return detokenize_chunks(tokenize_chunks(text, tok), tok) == normalize(text)
