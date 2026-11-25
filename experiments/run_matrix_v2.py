"""v0.2 experiment matrix.

Changes versus v0.1:
- chunk codec: exact document round-trip, no extra tokens
- add-alpha smoothing (alpha=0.1) instead of add-1
- fixed level-1 budget option (hier2_b256)
- reports both word-level and document-level decode

Usage:
    python -m experiments.run_matrix_v2 --corpora synthetic \
        --methods flat --vocab-sizes 256 --seeds 42 --out-dir results/v2_smoke
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from collections import Counter

from ramified.bpe import train_tokenizer
from ramified.codec import detokenize_chunks, normalize, roundtrip_ok, tokenize_chunks
from ramified.corpora import load_code_corpus
from ramified.metrics import decode_exact, mean_boundary_f1
from ramified.synthetic import generate_synthetic

METHODS = {
    "flat": {"n_levels": 1, "weights": None, "fixed_l1": None},
    "hier2": {"n_levels": 2, "weights": [0.6, 0.4], "fixed_l1": None},
    "hier2_b256": {"n_levels": 2, "weights": None, "fixed_l1": 256},
    "hier3": {"n_levels": 3, "weights": [0.5, 0.3, 0.2], "fixed_l1": None},
}


def alphabet_size(text: str) -> int:
    words = re.findall(r"\w+|[^\w\s]", text)
    return len(set("".join(words)))


def bpc_alpha(train_ids, test_ids, vocab_size, test_char_len, alpha=0.1):
    if test_char_len == 0:
        return float("nan")
    counts = Counter(train_ids)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    denom = total + alpha * vocab_size
    bits = 0.0
    for t in test_ids:
        bits -= math.log2((counts.get(t, 0) + alpha) / denom)
    return bits / test_char_len


def build_weights(spec, vocab_size, train_text):
    n_levels = spec["n_levels"]
    if n_levels == 1:
        return 1, None
    if spec["fixed_l1"] is not None:
        total = max(1, vocab_size - alphabet_size(train_text) - 2)
        budget = min(spec["fixed_l1"], total)
        w1 = budget / total
        return n_levels, [w1, 1.0 - w1]
    return n_levels, spec["weights"]


def eval_synthetic(tok, syn):
    test_ids = tokenize_chunks(syn.test_text, tok)
    train_ids = tokenize_chunks(syn.train_text, tok)
    p, r, f1 = mean_boundary_f1(tok, syn.test_words)
    norm_len = len(normalize(syn.test_text))
    return {
        "tokens_per_char": len(test_ids) / max(1, norm_len),
        "bits_per_char": bpc_alpha(train_ids, test_ids, tok.vocab_size, norm_len),
        "boundary_precision": p,
        "boundary_recall": r,
        "boundary_f1": f1,
        "decode_word": decode_exact(tok, syn.test_words),
        "decode_doc": 1.0 if roundtrip_ok(syn.test_text, tok) else 0.0,
    }


def eval_code(tok, train_text, test_text):
    train_ids = tokenize_chunks(train_text, tok)
    test_ids = tokenize_chunks(test_text, tok)
    norm_len = len(normalize(test_text))
    words = re.findall(r"\w+", test_text)[:1000]
    return {
        "tokens_per_char": len(test_ids) / max(1, norm_len),
        "bits_per_char": bpc_alpha(train_ids, test_ids, tok.vocab_size, norm_len),
        "boundary_precision": None,
        "boundary_recall": None,
        "boundary_f1": None,
        "decode_word": decode_exact(tok, [type("W", (), {"word": w}) for w in words]),
        "decode_doc": 1.0 if roundtrip_ok(test_text, tok) else 0.0,
    }


def fmt(row):
    bf1 = row["boundary_f1"]
    bf1_s = "-" if bf1 is None else f"{bf1:.3f}"
    return (
        f"{row['corpus']:<9} {row['method']:<11} v={row['vocab_target']:<5} "
        f"tok/ch={row['tokens_per_char']:.3f} bpc={row['bits_per_char']:.3f} "
        f"bF1={bf1_s} dW={row['decode_word']:.3f} dD={row['decode_doc']:.0f} "
        f"t={row['train_seconds']:.1f}s"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="synthetic")
    ap.add_argument("--methods", default="flat,hier2,hier2_b256,hier3")
    ap.add_argument("--vocab-sizes", default="256,512,1024,2048,4096")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out-dir", default="results/v2")
    args = ap.parse_args()

    corpora = [x.strip() for x in args.corpora.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    vocab_sizes = [int(x) for x in args.vocab_sizes.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    for seed in seeds:
        syn = generate_synthetic(seed=seed) if "synthetic" in corpora else None
        code = load_code_corpus() if "code" in corpora else None

        for method in methods:
            spec = METHODS[method]
            for vocab in vocab_sizes:
                for corpus_name in corpora:
                    if corpus_name == "synthetic":
                        train_text, test_text = syn.train_text, syn.test_text
                    else:
                        split = int(len(code) * 0.9)
                        nl = code.find("\n", split)
                        if nl > 0:
                            split = nl
                        train_text, test_text = code[:split], code[split:]

                    n_levels, weights = build_weights(spec, vocab, train_text)
                    t0 = time.perf_counter()
                    tok = train_tokenizer(
                        [train_text],
                        vocab_size=vocab,
                        n_levels=n_levels,
                        level_weights=weights,
                        split="chunks",
                    )
                    train_s = time.perf_counter() - t0

                    if corpus_name == "synthetic":
                        metrics = eval_synthetic(tok, syn)
                    else:
                        metrics = eval_code(tok, train_text, test_text)

                    row = {
                        "corpus": corpus_name,
                        "method": method,
                        "n_levels": len(tok.levels),
                        "merges_l1": len(tok.levels[0]) if len(tok.levels) > 0 else 0,
                        "merges_l2": len(tok.levels[1]) if len(tok.levels) > 1 else 0,
                        "merges_l3": len(tok.levels[2]) if len(tok.levels) > 2 else 0,
                        "vocab_target": vocab,
                        "vocab_actual": tok.vocab_size,
                        "seed": seed,
                        "train_seconds": round(train_s, 2),
                        **metrics,
                    }
                    rows.append(row)
                    print(fmt(row), flush=True)

    path = os.path.join(args.out_dir, "results.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
