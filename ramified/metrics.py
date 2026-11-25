"""Evaluation metrics for the ramified tokenizer experiments."""

from __future__ import annotations

import math
from collections import Counter

from .bpe import EOW


def bits_per_char(train_ids, test_ids, vocab_size, test_char_len):
    """Cross-entropy of a unigram model on the token stream, in bits per char."""
    if test_char_len == 0:
        return float("nan")
    counts = Counter(train_ids)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    denom = total + vocab_size
    bits = 0.0
    for t in test_ids:
        p = (counts.get(t, 0) + 1) / denom
        bits -= math.log2(p)
    return bits / test_char_len


def boundary_scores(pred, true):
    pred = set(pred)
    true = set(true)
    if not pred and not true:
        return 1.0, 1.0, 1.0
    tp = len(pred & true)
    # zero_division=0: an empty prediction is not perfect precision.
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def mean_boundary_f1(tokenizer, words):
    """Macro-averaged boundary P/R/F1 (one score per word)."""
    ps, rs, fs = [], [], []
    for w in words:
        p, r, f = boundary_scores(tokenizer.word_boundaries(w.word), w.boundaries)
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return (
        sum(ps) / len(ps) if ps else float("nan"),
        sum(rs) / len(rs) if rs else float("nan"),
        sum(fs) / len(fs) if fs else float("nan"),
    )


def boundary_report(tokenizer, words):
    """Macro and micro boundary scores, plus the pooled TP/FP/FN counts.

    Macro averages per-word scores; micro pools all boundary decisions. The
    two conventions can differ by more than the small method effects, so both
    are reported.
    """
    ps, rs, fs = [], [], []
    tp = fp = fn = 0
    for w in words:
        pred = set(tokenizer.word_boundaries(w.word))
        true = set(w.boundaries)
        p, r, f = boundary_scores(pred, true)
        ps.append(p)
        rs.append(r)
        fs.append(f)
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)

    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    )
    return {
        "boundary_macro_precision": sum(ps) / len(ps) if ps else float("nan"),
        "boundary_macro_recall": sum(rs) / len(rs) if rs else float("nan"),
        "boundary_macro_f1": sum(fs) / len(fs) if fs else float("nan"),
        "boundary_micro_precision": micro_p,
        "boundary_micro_recall": micro_r,
        "boundary_micro_f1": micro_f1,
        "boundary_tp": tp,
        "boundary_fp": fp,
        "boundary_fn": fn,
    }


def decode_exact(tokenizer, words):
    ok = 0
    for w in words:
        decoded = tokenizer.decode(tokenizer.encode(w.word, eow=True)).replace(EOW, "")
        if decoded == w.word:
            ok += 1
    return ok / len(words) if words else float("nan")
