"""Run the ramified-tokenization experiment matrix.

Example (smoke test):
    python -m experiments.run_matrix --corpora synthetic --vocab-sizes 256,512,1024

Full matrix:
    python -m experiments.run_matrix --corpora synthetic,code \
        --methods flat,hier2,hier3 --vocab-sizes 256,512,1024,2048,4096 \
        --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time

from ramified.bpe import EOW, train_tokenizer, tokenize_normalized
from ramified.corpora import load_code_corpus
from ramified.metrics import bits_per_char, decode_exact, mean_boundary_f1
from ramified.synthetic import generate_synthetic

METHODS = {
    "flat": (1, None),
    "hier2": (2, [0.6, 0.4]),
    "hier3": (3, [0.5, 0.3, 0.2]),
}


def eval_synthetic(tok, corpus):
    train_ids = tokenize_normalized(corpus.train_text, tok)
    test_ids = tokenize_normalized(corpus.test_text, tok)
    bpc = bits_per_char(train_ids, test_ids, tok.vocab_size, len(corpus.test_text))
    p, r, f1 = mean_boundary_f1(tok, corpus.test_words)
    return {
        "tokens_per_char": len(test_ids) / max(1, len(corpus.test_text)),
        "bits_per_char": bpc,
        "boundary_precision": p,
        "boundary_recall": r,
        "boundary_f1": f1,
        "decode_exact": decode_exact(tok, corpus.test_words),
    }


def eval_code(tok, train_text, test_text):
    train_ids = tokenize_normalized(train_text, tok)
    test_ids = tokenize_normalized(test_text, tok)
    bpc = bits_per_char(train_ids, test_ids, tok.vocab_size, len(test_text))
    words = re.findall(r"\w+", test_text)[:1000]
    ok = 0
    for w in words:
        if tok.decode(tok.encode(w, eow=True)).replace(EOW, "") == w:
            ok += 1
    return {
        "tokens_per_char": len(test_ids) / max(1, len(test_text)),
        "bits_per_char": bpc,
        "boundary_precision": None,
        "boundary_recall": None,
        "boundary_f1": None,
        "decode_exact": ok / max(1, len(words)),
    }


def fmt(row):
    bf1 = row["boundary_f1"]
    bf1_s = "-" if bf1 is None else f"{bf1:.3f}"
    return (
        f"{row['corpus']:<9} {row['method']:<6} v={row['vocab_target']:<5} "
        f"tok/ch={row['tokens_per_char']:.3f} bpc={row['bits_per_char']:.3f} "
        f"bF1={bf1_s} dec={row['decode_exact']:.3f} t={row['train_seconds']:.1f}s"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="synthetic")
    ap.add_argument("--methods", default="flat,hier2,hier3")
    ap.add_argument("--vocab-sizes", default="256,512,1024")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out-dir", default="results")
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
            n_levels, weights = METHODS[method]
            for vocab in vocab_sizes:
                for corpus_name in corpora:
                    if corpus_name == "synthetic":
                        train_text = syn.train_text
                        test_text = syn.test_text
                    else:
                        split = int(len(code) * 0.9)
                        nl = code.find("\n", split)
                        if nl > 0:
                            split = nl
                        train_text, test_text = code[:split], code[split:]

                    t0 = time.perf_counter()
                    tok = train_tokenizer(
                        [train_text],
                        vocab_size=vocab,
                        n_levels=n_levels,
                        level_weights=weights,
                    )
                    train_s = time.perf_counter() - t0

                    if corpus_name == "synthetic":
                        metrics = eval_synthetic(tok, syn)
                    else:
                        metrics = eval_code(tok, train_text, test_text)

                    row = {
                        "corpus": corpus_name,
                        "method": method,
                        "n_levels": n_levels,
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
