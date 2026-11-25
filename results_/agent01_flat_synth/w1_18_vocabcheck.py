"""W1-18 helper: are the v1024/v2048 'null' level assignments identical?"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")
from ramified.bpe import train_tokenizer
from ramified.levels import assign_levels
from ramified.synthetic import generate_synthetic
from results.redteam.w1_18_bpc_audit import provenance_weighted


def ids(tok):
    return {m.new for m in tok.levels[0]}


def jac(a, b):
    return len(a & b) / len(a | b)


for vocab in (512, 1024, 2048):
    syn = generate_synthetic(seed=42)
    flat = train_tokenizer([syn.train_text], vocab_size=vocab, n_levels=1, split="chunks")
    stats = provenance_weighted(flat, syn.train_word_objects, syn.train_text)
    freq = assign_levels(flat, stats, mode="frequency", level1_fraction=0.6, seed=42000 + vocab)
    morph = assign_levels(flat, stats, mode="morph", level1_fraction=0.6, seed=42000 + vocab)
    f1, m1 = ids(freq), ids(morph)
    test = syn.test_text.split()
    tok_f = [t for c in test for t in freq.encode(c, eow=True)]
    tok_m = [t for c in test for t in morph.encode(c, eow=True)]
    print(
        f"v={vocab} n_rules={len(flat.levels[0])} n_l1={len(f1)} "
        f"jaccard_l1={jac(f1, m1):.4f} rules_level_diff={len(f1 ^ m1)} "
        f"test_tokenization_equal={tok_f == tok_m} "
        f"n_test_tokens_f={len(tok_f)} n_test_tokens_m={len(tok_m)}"
    )
