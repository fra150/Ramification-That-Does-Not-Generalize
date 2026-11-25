"""W1-18: audit of ``bits_per_char`` in the v512 level-assignment result.

Recomputes the v512, 10-seed morph/frequency comparison under several
estimators and decomposes the difference. Does not modify source files.

Outputs
-------
results/redteam/w1_18_bpc_audit.csv     one row per (seed, mode)
results/redteam/w1_18_bpc_paired.csv    one row per (estimator, pair)
stdout                                  decomposition + bootstrap detail

Usage
-----
    python results/redteam/w1_18_bpc_audit.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict

import numpy as np

from ramified.bpe import train_tokenizer
from ramified.codec import normalize
from ramified.levels import _walk, assign_levels, collect_provenance
from ramified.metrics import bits_per_char, mean_boundary_f1
from ramified.synthetic import generate_synthetic

MODES = ["flat", "frequency", "morph", "random_closed"]
PAIRS = [("morph", "frequency"), ("random_closed", "frequency"),
         ("morph", "random_closed")]


def provenance_weighted(tok, word_objects, text):
    """Same counts as collect_provenance(..., text=text), with unique-word
    caching. The source function walks every occurrence; _walk is linear, so
    weighting each unique word by its text frequency gives identical counts."""
    freq = Counter(text.split())
    by_word = {w.word: w for w in word_objects}
    stats = defaultdict(lambda: (0, 0))
    for word_str, count in freq.items():
        word = by_word.get(word_str)
        if word is None:
            continue
        local = defaultdict(lambda: (0, 0))
        pos = 0
        for node in tok.encode_nodes(word.word, eow=True):
            pos = _walk(node, pos, tok, word.boundaries, local)
        for key, (intra, cross) in local.items():
            si, sc = stats[key]
            stats[key] = (si + intra * count, sc + cross * count)
    return dict(stats)


def encode_train(train_chunks, tok):
    ids = []
    for chunk, count in Counter(train_chunks).items():
        ids.extend(tok.encode(chunk, eow=True) * count)
    return ids


def encode_test(chunks, tok):
    per_word = [tok.encode(chunk, eow=True) for chunk in chunks]
    return [t for ids in per_word for t in ids], per_word


def nll_ids(train_ids, test_ids, vocab_size, alpha):
    counts = Counter(train_ids)
    n = sum(counts.values())
    denom = n + alpha * vocab_size
    return sum(-math.log2((counts.get(t, 0) + alpha) / denom) for t in test_ids), counts, n


def nll_witten_bell(train_ids, test_ids, vocab_size):
    counts = Counter(train_ids)
    n = sum(counts.values())
    types = len(counts)
    denom = n + types
    unseen_alphabet = max(1, vocab_size - types)
    bits = 0.0
    for t in test_ids:
        if t in counts:
            bits -= math.log2(counts[t] / denom)
        else:
            bits -= math.log2((types / denom) / unseen_alphabet)
    return bits


def empirical_entropy(ids):
    counts = Counter(ids)
    n = sum(counts.values())
    return sum(-(c / n) * math.log2(c / n) for c in counts.values())


def describe(deltas):
    n = len(deltas)
    mean = statistics.mean(deltas)
    std = statistics.stdev(deltas) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    crit = {10: 2.262, 9: 2.306}.get(n, 1.96)
    return {
        "n": n, "mean": mean, "std": std, "se": se,
        "t": mean / se if se else float("inf"),
        "ci_low": mean - crit * se, "ci_high": mean + crit * se,
        "wins": sum(1 for d in deltas if d > 1e-12),
        "losses": sum(1 for d in deltas if d < -1e-12),
        "ties": sum(1 for d in deltas if abs(d) <= 1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--flat-csv", default="results/level_assignment_v512_10seeds.csv")
    ap.add_argument("--out", default="results/redteam/w1_18_bpc_audit.csv")
    ap.add_argument("--paired-out", default="results/redteam/w1_18_bpc_paired.csv")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    vocab = args.vocab

    official = {}
    with open(args.flat_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["vocab"]) == vocab:
                official[(int(row["seed"]), row["mode"])] = float(row["bits_per_char"])

    rows = []
    cross_rows = []
    boot_rows = []
    repro = []

    for seed in seeds:
        syn = generate_synthetic(seed=seed)
        train_chunks = syn.train_text.split()
        flat = train_tokenizer(
            [syn.train_text], vocab_size=vocab, n_levels=1, split="chunks"
        )
        stats = provenance_weighted(flat, syn.train_word_objects, syn.train_text)
        if seed == seeds[0]:
            official_stats = collect_provenance(
                flat, syn.train_word_objects, text=syn.train_text
            )
            same = official_stats == stats
            print(f"provenance_match_official={same} "
                  f"rules={len(stats)} (seed {seed})", flush=True)
            assert same, "fast provenance differs from ramified.levels.collect_provenance"
        toks = {"flat": flat}
        for mode in ("frequency", "morph", "random_closed"):
            toks[mode] = assign_levels(
                flat, stats, mode=mode, level1_fraction=0.6,
                seed=seed * 1000 + vocab,
            )

        norm_len = len(normalize(syn.test_text))
        norm_train_len = len(normalize(syn.train_text))
        chunks = syn.test_text.split()
        assert " ".join(chunks) == syn.test_text
        assert sum(len(c) for c in chunks) + len(chunks) - 1 == norm_len
        n_flat = None
        train_ids = {}
        test_ids = {}
        test_per_word = {}
        for mode in MODES:
            tok = toks[mode]
            train_ids[mode] = encode_train(train_chunks, tok)
            test_ids[mode], test_per_word[mode] = encode_test(chunks, tok)
            if seed == seeds[0]:
                from ramified.codec import tokenize_chunks
                assert test_ids[mode] == tokenize_chunks(syn.test_text, tok)
            if mode == "flat":
                n_flat = len(train_ids[mode])

        for mode in MODES:
            tok = toks[mode]
            tr, te = train_ids[mode], test_ids[mode]
            counts = Counter(tr)
            n_train = sum(counts.values())
            types_train = len(counts)
            types_test = len(set(te))
            unseen_types = len(set(te) - set(counts))
            unseen_tokens = sum(1 for t in te if t not in counts)
            n_test = len(te)
            tpc = n_test / norm_len

            nll01, _, _ = nll_ids(tr, te, vocab, 0.1)
            nll1, _, _ = nll_ids(tr, te, vocab, 1.0)
            nll001, _, _ = nll_ids(tr, te, vocab, 0.01)
            nll0001, _, _ = nll_ids(tr, te, vocab, 0.001)
            nll1e6, _, _ = nll_ids(tr, te, vocab, 1e-6)
            nll_wb = nll_witten_bell(tr, te, vocab)
            nll_fixdenom = sum(
                -math.log2((counts.get(t, 0) + 0.1) / (n_flat + 0.1 * vocab))
                for t in te
            )
            nll_seen = sum(
                -math.log2((counts[t] + 0.1) / (n_train + 0.1 * vocab))
                for t in te if t in counts
            )
            pen_unseen = math.log2((n_train + 0.1 * vocab) / 0.1)
            nll_unseen = unseen_tokens * pen_unseen
            h_test = empirical_entropy(te)
            h_train = empirical_entropy(tr)
            bpc_official = nll01 / norm_len
            _, _, f1 = mean_boundary_f1(tok, syn.test_words)

            if (seed, mode) in official:
                repro.append(abs(bpc_official - official[(seed, mode)]))

            rows.append({
                "seed": seed, "vocab": vocab, "mode": mode,
                "vocab_size": tok.vocab_size,
                "n_train_tokens": n_train,
                "n_train_types": types_train,
                "n_test_tokens": n_test,
                "n_test_types": types_test,
                "unseen_tokens": unseen_tokens,
                "unseen_rate": unseen_tokens / n_test,
                "unseen_types": unseen_types,
                "tokens_per_char": tpc,
                "norm_len": norm_len,
                "bpc_official_a0.1": bpc_official,
                "bpc_repro_delta": (
                    bpc_official - official[(seed, mode)]
                    if (seed, mode) in official else ""
                ),
                "bpc_add1_metrics_py": nll1 / norm_len,
                "bpc_add0.01": nll001 / norm_len,
                "bpc_add0.001": nll0001 / norm_len,
                "bpc_add1e-6": nll1e6 / norm_len,
                "bpc_witten_bell": nll_wb / norm_len,
                "bpc_fixed_denom_flatN": nll_fixdenom / norm_len,
                "bpc_seen_only": nll_seen / norm_len,
                "bpc_unseen_component": nll_unseen / norm_len,
                "unseen_share_of_nll": nll_unseen / nll01 if nll01 else 0.0,
                "penalty_per_unseen_bit": pen_unseen,
                "mean_surprisal_seen_bit": (
                    (nll01 - nll_unseen) / (n_test - unseen_tokens)
                    if n_test > unseen_tokens else float("nan")
                ),
                "h_train_per_token_bit": h_train,
                "h_test_per_token_bit": h_test,
                "bpc_train_entropy": tpc * h_train,
                "bpc_test_entropy": tpc * h_test,
                "bpc_uniform9": tpc * math.log2(vocab),
                "bpc_train_stream_insample": (
                    nll_ids(tr, tr, vocab, 0.1)[0] / norm_train_len
                ),
                "boundary_f1": f1,
                "bpc_lib_add1_plus_uniform": (
                    bits_per_char(tr, te, tok.vocab_size, norm_len)
                ),
            })

        # cross-model: score stream A with unigram counts from train stream B
        for a in ("frequency", "morph"):
            for b in ("frequency", "morph"):
                counts = Counter(train_ids[b])
                n = sum(counts.values())
                denom = n + 0.1 * vocab
                nll = sum(
                    -math.log2((counts.get(t, 0) + 0.1) / denom)
                    for t in test_ids[a]
                )
                cross_rows.append({
                    "seed": seed, "stream": a, "model": b,
                    "bpc": nll / norm_len,
                })

        # word bootstrap of the official estimator, morph vs frequency
        nll_words = {}
        for mode in ("frequency", "morph"):
            counts = Counter(train_ids[mode])
            n = sum(counts.values())
            denom = n + 0.1 * vocab
            vals = []
            for ids in test_per_word[mode]:
                vals.append(sum(
                    -math.log2((counts.get(t, 0) + 0.1) / denom) for t in ids
                ))
            nll_words[mode] = np.asarray(vals, dtype=np.float64)
        lens = np.asarray([len(w) + 1 for w in chunks], dtype=np.float64)
        lens[-1] -= 1.0
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(chunks), size=(args.boot, len(chunks)))
        diff = nll_words["morph"][idx].sum(axis=1) - nll_words["frequency"][idx].sum(axis=1)
        d_fixed = diff / norm_len
        d_recomp = diff / lens[idx].sum(axis=1)
        boot_rows.append({
            "seed": seed,
            "delta_point": (nll_words["morph"].sum() - nll_words["frequency"].sum()) / norm_len,
            "boot_mean_fixedL": float(d_fixed.mean()),
            "boot_lo_fixedL": float(np.percentile(d_fixed, 2.5)),
            "boot_hi_fixedL": float(np.percentile(d_fixed, 97.5)),
            "boot_p_ge0_fixedL": float((d_fixed >= 0).mean()),
            "boot_mean_recompL": float(d_recomp.mean()),
            "boot_lo_recompL": float(np.percentile(d_recomp, 2.5)),
            "boot_hi_recompL": float(np.percentile(d_recomp, 97.5)),
        })

        print(
            f"seed={seed} L={norm_len} "
            + " ".join(
                f"{m}[tpc={next(r['tokens_per_char'] for r in rows if r['seed'] == seed and r['mode'] == m):.4f},"
                f"un={next(r['unseen_tokens'] for r in rows if r['seed'] == seed and r['mode'] == m)},"
                f"bpc={next(r['bpc_official_a0.1'] for r in rows if r['seed'] == seed and r['mode'] == m):.4f}]"
                for m in ("frequency", "morph")
            ),
            flush=True,
        )

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    estimators = [
        "bpc_official_a0.1", "bpc_add1_metrics_py", "bpc_add0.01",
        "bpc_add0.001", "bpc_add1e-6", "bpc_witten_bell",
        "bpc_fixed_denom_flatN", "bpc_seen_only", "bpc_train_entropy",
        "bpc_test_entropy", "bpc_uniform9", "bpc_train_stream_insample",
        "tokens_per_char", "unseen_tokens", "boundary_f1",
    ]
    paired_rows = []
    for a, b in PAIRS:
        for est in estimators:
            vals_a = [r[est] for r in rows if r["mode"] == a]
            vals_b = [r[est] for r in rows if r["mode"] == b]
            ok = [(x, y) for x, y in zip(vals_a, vals_b)
                  if isinstance(x, (int, float)) and isinstance(y, (int, float))]
            if len(ok) != len(seeds):
                continue
            d = [x - y for x, y in ok]
            st = describe(d)
            paired_rows.append({
                "estimator": est, "mode_a": a, "mode_b": b, **st,
            })

    with open(args.paired_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    print("\n== reproduction of official bpc ==")
    print(f"max |recomputed - CSV| = {max(repro):.3e} over {len(repro)} rows")

    print("\n== paired differences, morph - frequency, v512, 10 seeds ==")
    for r in paired_rows:
        if r["mode_a"] == "morph" and r["mode_b"] == "frequency":
            print(
                f"{r['estimator']:<24} mean={r['mean']:+.5f} "
                f"t={r['t']:+7.2f} CI95=[{r['ci_low']:+.5f},{r['ci_high']:+.5f}] "
                f"W/L/T={r['wins']}/{r['losses']}/{r['ties']}"
            )

    print("\n== per-seed decomposition (official alpha=0.1) ==")
    print("seed mode        tpc    bpc    unseen/U  unseen_share  bpc_seen  bpc_unseen  H_test")
    for r in rows:
        if r["mode"] not in ("frequency", "morph"):
            continue
        print(
            f"{r['seed']} {r['mode']:<11} {r['tokens_per_char']:.4f} "
            f"{r['bpc_official_a0.1']:.4f} {r['unseen_tokens']:>4}/{r['n_test_tokens']:<4} "
            f"{r['unseen_share_of_nll']:>10.3f} {r['bpc_seen_only']:.4f} "
            f"{r['bpc_unseen_component']:.4f} {r['h_test_per_token_bit']:.4f}"
        )

    print("\n== word-bootstrap, official bpc, morph - frequency ==")
    for r in boot_rows:
        print(
            f"seed={r['seed']} point={r['delta_point']:+.5f} "
            f"fixedL CI95=[{r['boot_lo_fixedL']:+.5f},{r['boot_hi_fixedL']:+.5f}] "
            f"P(d>=0)={r['boot_p_ge0_fixedL']:.3f} "
            f"recompL CI95=[{r['boot_lo_recompL']:+.5f},{r['boot_hi_recompL']:+.5f}]"
        )

    print("\n== cross-model bpc (stream x model), means over seeds ==")
    for a in ("frequency", "morph"):
        for b in ("frequency", "morph"):
            vals = [r["bpc"] for r in cross_rows if r["stream"] == a and r["model"] == b]
            print(f"stream={a:<9} model={b:<9} mean_bpc={statistics.mean(vals):.5f}")

    print("\n== correlation of delta bpc with delta F1 / delta tpc ==")
    idx = {(r["seed"], r["mode"]): r for r in rows}
    d_bpc, d_f1, d_tpc, d_un = [], [], [], []
    for s in seeds:
        m, f = idx[(s, "morph")], idx[(s, "frequency")]
        d_bpc.append(m["bpc_official_a0.1"] - f["bpc_official_a0.1"])
        d_f1.append(m["boundary_f1"] - f["boundary_f1"])
        d_tpc.append(m["tokens_per_char"] - f["tokens_per_char"])
        d_un.append(m["unseen_tokens"] - f["unseen_tokens"])
    print(f"mean d_bpc={statistics.mean(d_bpc):+.5f} mean d_F1={statistics.mean(d_f1):+.5f} "
          f"mean d_tpc={statistics.mean(d_tpc):+.5f} mean d_unseen={statistics.mean(d_un):+.2f}")
    print(f"corr(d_bpc,d_F1)={np.corrcoef(d_bpc, d_f1)[0,1]:+.3f} "
          f"corr(d_bpc,d_tpc)={np.corrcoef(d_bpc, d_tpc)[0,1]:+.3f} "
          f"corr(d_bpc,d_unseen)={np.corrcoef(d_bpc, d_un)[0,1]:+.3f}")
    x = np.asarray(d_f1); y = np.asarray(d_bpc)
    beta, alpha0 = np.polyfit(x, y, 1)
    resid = y - (alpha0 + beta * x)
    ss_res, ss_tot = (resid ** 2).sum(), ((y - y.mean()) ** 2).sum()
    print(f"OLS d_bpc = {alpha0:+.5f} + {beta:+.3f}*d_F1 ; R2={1 - ss_res/ss_tot:.3f}")
    print(f"residual mean={resid.mean():+.5f} sd={resid.std(ddof=1):.5f}")

    print("\n== vocab-size accounting (why bpc is not comparable across vocab) ==")
    print("mean n_train_tokens, used types, alpha*V, alpha*V/N for v512 run:")
    for m in MODES:
        vals = [r for r in rows if r["mode"] == m]
        n = statistics.mean(r["n_train_tokens"] for r in vals)
        t = statistics.mean(r["n_train_types"] for r in vals)
        print(f"{m:<11} N={n:9.1f} used_types={t:6.1f} alpha*V={0.1*vocab:6.1f} "
              f"frac={0.1*vocab/n:.2e}")


if __name__ == "__main__":
    main()
