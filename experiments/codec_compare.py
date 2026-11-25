"""Compare codecs and training splits on the same denominator.

The v0.2 chunk codec changed two things at once: the encoding unit (chunks
instead of regex words) and the bpc denominator (normalised instead of raw
text). This script separates them.

Configurations:
  A  train=words   encode=words    (v0.1)
  B  train=words   encode=chunks   (words merges, exact document codec)
  C  train=chunks  encode=chunks   (v0.2 as first tested)

All bpc figures use add-alpha smoothing (alpha=0.1) and the normalised test
length, so B and C are directly comparable and A is shown on both denominators
to expose the v0.1 deflation.

Usage:
    python -m experiments.codec_compare --corpus code --vocab 512
    python -m experiments.codec_compare --corpus synthetic --vocab 512
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

from ramified.bpe import tokenize_normalized, train_tokenizer
from ramified.codec import (
    detokenize_chunks,
    normalize,
    tokenize_chunks,
)
from ramified.corpora import load_code_corpus
from ramified.metrics import mean_boundary_f1
from ramified.synthetic import generate_synthetic


def bpc(train_ids, test_ids, vocab_size, char_len, alpha=0.1):
    counts = Counter(train_ids)
    total = sum(counts.values())
    if not total or not char_len:
        return float("nan")
    denom = total + alpha * vocab_size
    bits = sum(-math.log2((counts.get(t, 0) + alpha) / denom) for t in test_ids)
    return bits / char_len


def eval_words(tok, train_text, test_text, syn=None):
    train_ids = tokenize_normalized(train_text, tok)
    test_ids = tokenize_normalized(test_text, tok)
    out = {
        "train_tokens": len(train_ids),
        "test_tokens": len(test_ids),
        "tok_per_raw_char": len(test_ids) / max(1, len(test_text)),
        "tok_per_norm_char": len(test_ids) / max(1, len(normalize(test_text))),
        "bpc_raw_denom": bpc(train_ids, test_ids, tok.vocab_size, len(test_text)),
        "bpc_norm_denom": bpc(train_ids, test_ids, tok.vocab_size, len(normalize(test_text))),
        "decode_doc": None,
    }
    if syn is not None:
        p, r, f1 = mean_boundary_f1(tok, syn.test_words)
        out.update({"boundary_f1": f1, "boundary_precision": p, "boundary_recall": r})
    return out


def eval_chunks(tok, train_text, test_text, syn=None):
    train_ids = tokenize_chunks(train_text, tok)
    test_ids = tokenize_chunks(test_text, tok)
    out = {
        "train_tokens": len(train_ids),
        "test_tokens": len(test_ids),
        "tok_per_raw_char": len(test_ids) / max(1, len(test_text)),
        "tok_per_norm_char": len(test_ids) / max(1, len(normalize(test_text))),
        "bpc_raw_denom": bpc(train_ids, test_ids, tok.vocab_size, len(test_text)),
        "bpc_norm_denom": bpc(train_ids, test_ids, tok.vocab_size, len(normalize(test_text))),
        "decode_doc": detokenize_chunks(test_ids, tok) == normalize(test_text),
    }
    if syn is not None:
        p, r, f1 = mean_boundary_f1(tok, syn.test_words)
        out.update({"boundary_f1": f1, "boundary_precision": p, "boundary_recall": r})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["code", "synthetic"], default="code")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    syn = None
    if args.corpus == "synthetic":
        syn = generate_synthetic(seed=args.seed)
        train_text, test_text = syn.train_text, syn.test_text
    else:
        code = load_code_corpus()
        split = int(len(code) * 0.9)
        nl = code.find("\n", split)
        if nl > 0:
            split = nl
        train_text, test_text = code[:split], code[split:]

    raw_len = len(test_text)
    norm_len = len(normalize(test_text))
    n_chunks = len(normalize(test_text).split(" "))
    n_regex = len(re.findall(r"\w+|[^\w\s]", normalize(test_text)))
    print(
        f"corpus={args.corpus} vocab={args.vocab} raw={raw_len} norm={norm_len} "
        f"reduction={1 - norm_len / raw_len:.1%} chunks={n_chunks} regex_units={n_regex}"
    )

    tok_words = train_tokenizer([train_text], vocab_size=args.vocab, n_levels=1, split="words")
    tok_chunks = train_tokenizer([train_text], vocab_size=args.vocab, n_levels=1, split="chunks")

    rows = [
        ("A words/words", eval_words(tok_words, train_text, test_text, syn)),
        ("B words/chunks", eval_chunks(tok_words, train_text, test_text, syn)),
        ("C chunks/chunks", eval_chunks(tok_chunks, train_text, test_text, syn)),
    ]

    header = (
        f"{'config':<16} {'test_tok':>9} {'tok/raw':>8} {'tok/norm':>9} "
        f"{'bpc_raw':>8} {'bpc_norm':>9} {'dD':>4} {'bF1':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, m in rows:
        bf1 = m.get("boundary_f1")
        bf1_s = "-" if bf1 is None else f"{bf1:.3f}"
        dD = m["decode_doc"]
        dD_s = "-" if dD is None else ("1" if dD else "0")
        print(
            f"{name:<16} {m['test_tokens']:>9} {m['tok_per_raw_char']:>8.3f} "
            f"{m['tok_per_norm_char']:>9.3f} {m['bpc_raw_denom']:>8.3f} "
            f"{m['bpc_norm_denom']:>9.3f} {dD_s:>4} {bf1_s:>6}"
        )


if __name__ == "__main__":
    main()
