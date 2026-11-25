"""W1-15 red-team check: is the v512 morph-level advantage an artifact of
frequency-weighted provenance estimated on the same words that trained BPE?

Compares level assignment under provenance estimated by:
  - frequency-weighted training text (the original design)
  - unique training words (type weighting)
  - two disjoint random halves of the unique training words (held-out words)
  - 10-fold cross-fitting by root: assignment for a test word never uses any
    training word sharing its root (held-out-root / leave-one-root-out)

Holds the flat BPE rules and vocabulary fixed, exactly as
experiments/level_assignment.py does. Evaluates boundary F1 and bpc at v512
on seeds 42-46. Does not modify source files.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict

from ramified.bpe import train_tokenizer
from ramified.codec import normalize
from ramified.levels import assign_levels, collect_provenance, level_summary
from ramified.metrics import boundary_scores
from ramified.synthetic import generate_synthetic

T95 = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228}


def f1_words(tok, words):
    """Mean boundary F1 over word types, with the number of words."""
    fs = []
    for w in words:
        p, r, f = boundary_scores(tok.word_boundaries(w.word), w.boundaries)
        fs.append(f)
    return (sum(fs) / len(fs) if fs else float("nan")), len(fs)


def token_counts(tok, counts):
    out = Counter()
    for chunk, c in counts.items():
        for t in tok.encode(chunk, eow=True):
            out[t] += c
    return out


def bpc_counts(tok, train_counts, test_counts, char_len, alpha=0.1,
               train_token_counts=None):
    """Exact bpc_alpha, computed from unique-chunk counts. Tokenization is
    per whitespace chunk, so summing counts over unique chunks is identical
    to tokenizing the full 120k-token text, but ~13x cheaper."""
    train_token_counts = (
        train_token_counts if train_token_counts is not None
        else token_counts(tok, train_counts)
    )
    total = sum(train_token_counts.values())
    denom = total + alpha * tok.vocab_size
    bits = 0.0
    for chunk, c in test_counts.items():
        for t in tok.encode(chunk, eow=True):
            bits -= c * math.log2((train_token_counts.get(t, 0) + alpha) / denom)
    return bits / char_len


def level1_set(tok):
    return {m.new for m in tok.levels[0]}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def freq_token(flat, n1, seed):
    """Frequency-prefix tokenizer with exactly n1 level-1 rules. Used as a
    size-matched control when dependency closure shrinks a morph level 1."""
    n_rules = len(flat.levels[0])
    frac = min(1.0, max(0.0, (n1 + 0.1) / n_rules))
    return assign_levels(flat, {}, mode="frequency", level1_fraction=frac,
                         seed=seed)


def agg(word_counts, indices, weights=None):
    """Additive per-word provenance counts. Identical to collect_provenance
    on the corresponding word multiset, but computed once per unique word."""
    out = defaultdict(lambda: [0, 0])
    for i in indices:
        wgt = 1 if weights is None else weights[i]
        if not wgt:
            continue
        for key, (a, b) in word_counts[i].items():
            out[key][0] += a * wgt
            out[key][1] += b * wgt
    return {k: (v[0], v[1]) for k, v in out.items()}


def row(seed, variant, mode, eval_set, tok, words, stats, bpc=None, jac=None):
    f1, n = f1_words(tok, words)
    out = {
        "seed": seed,
        "vocab": 512,
        "variant": variant,
        "mode": mode,
        "eval_set": eval_set,
        "eval_n_words": n,
        "n_l1": len(tok.levels[0]) if tok.levels else 0,
        "n_l2": len(tok.levels[1]) if len(tok.levels) > 1 else 0,
        "boundary_f1": f1,
        "bits_per_char": "" if bpc is None else bpc,
        "jaccard_vs_freq_all_morph": "" if jac is None else jac,
    }
    for item in level_summary(tok, stats):
        prefix = f"l{item['level'] + 1}"
        out[f"{prefix}_intra"] = item["intra"]
        out[f"{prefix}_cross"] = item["cross"]
        out[f"{prefix}_unseen"] = item["unseen"]
    return out


def describe(deltas):
    n = len(deltas)
    mean = statistics.mean(deltas)
    std = statistics.stdev(deltas) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("inf")
    crit = T95.get(n, 1.96)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "t": t,
        "ci_low": mean - crit * se,
        "ci_high": mean + crit * se,
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--fraction", type=float, default=0.6)
    ap.add_argument("--root-folds", type=int, default=10)
    ap.add_argument("--out-dir", default="results/redteam")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    vocab = args.vocab
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    fold_rows = []
    sim_rows = []
    check_rows = []

    for seed in seeds:
        t0 = time.perf_counter()
        syn = generate_synthetic(seed=seed)
        flat = train_tokenizer(
            [syn.train_text], vocab_size=vocab, n_levels=1, split="chunks"
        )
        train_seconds = time.perf_counter() - t0

        words = list(syn.train_word_objects)
        n_words = len(words)
        train_counts = Counter(syn.train_text.split())
        test_counts = Counter(syn.test_text.split())
        test_char_len = len(normalize(syn.test_text))
        weights = [train_counts.get(w.word, 0) for w in words]
        covered = sum(1 for x in weights if x > 0)

        # Per-unique-word provenance, then weighted aggregation. Cached once.
        # This is exactly additive: the derivation of a word type is the same
        # on every occurrence.
        wc = [collect_provenance(flat, [w]) for w in words]
        all_idx = list(range(n_words))
        stats_unique = agg(wc, all_idx)
        stats_freq = agg(wc, all_idx, weights)

        # Verify the cache against the library function before trusting it.
        # The frequency-weighted reference re-encodes all 120k token
        # occurrences, so run it once; the unique check runs every seed.
        ref_unique = collect_provenance(flat, words)
        keys = set(stats_unique) | set(ref_unique)
        uniq_mismatch = max(
            (abs(stats_unique.get(k, (0, 0))[0] - ref_unique.get(k, (0, 0))[0])
             + abs(stats_unique.get(k, (0, 0))[1] - ref_unique.get(k, (0, 0))[1]))
            for k in keys
        )
        if seed == seeds[0]:
            ref_freq = collect_provenance(flat, words, text=syn.train_text)
            keys = set(stats_freq) | set(ref_freq)
            freq_mismatch = max(
                (abs(stats_freq.get(k, (0, 0))[0] - ref_freq.get(k, (0, 0))[0])
                 + abs(stats_freq.get(k, (0, 0))[1] - ref_freq.get(k, (0, 0))[1]))
                for k in keys
            )
        else:
            freq_mismatch = ""
        check_rows.append({
            "seed": seed,
            "n_train_word_types": n_words,
            "n_types_in_text": covered,
            "n_text_tokens": len(syn.train_text.split()),
            "vocab": vocab,
            "train_seconds": round(train_seconds, 2),
            "unique_stats_max_abs_mismatch": uniq_mismatch,
            "freq_stats_max_abs_mismatch": freq_mismatch,
        })
        print(
            f"seed={seed} train={train_seconds:.1f}s types={n_words} "
            f"types_seen={covered} cache_mismatch={uniq_mismatch},{freq_mismatch}",
            flush=True,
        )

        rng = random.Random(seed * 7919 + 15)
        perm = list(range(n_words))
        rng.shuffle(perm)
        half = n_words // 2
        half_a, half_b = perm[:half], perm[half:]

        roots = sorted({w.root for w in words})
        rng.shuffle(roots)
        root_folds = [roots[k::args.root_folds] for k in range(args.root_folds)]
        root_to_idx = defaultdict(list)
        for i, w in enumerate(words):
            root_to_idx[w.root].append(i)

        stats_variants = {
            "freq_all": stats_freq,
            "unique_all": stats_unique,
            "freq_halfA": agg(wc, half_a, weights),
            "freq_halfB": agg(wc, half_b, weights),
            "unique_halfA": agg(wc, half_a),
            "unique_halfB": agg(wc, half_b),
        }
        # fold k: provenance from every training word whose root is not in fold k
        fold_stats = []
        for k, fold_roots in enumerate(root_folds):
            excluded = set()
            for r in fold_roots:
                excluded.update(root_to_idx[r])
            keep = [i for i in all_idx if i not in excluded]
            fold_stats.append(agg(wc, keep, weights))

        # Tokenizers.
        toks = {}
        for name, st in stats_variants.items():
            toks[name] = assign_levels(
                flat, st, mode="morph", level1_fraction=args.fraction,
                seed=seed * 1000 + vocab,
            )
        freq_tok = assign_levels(
            flat, stats_freq, mode="frequency", level1_fraction=args.fraction,
            seed=seed * 1000 + vocab,
        )
        rand_tok = assign_levels(
            flat, stats_freq, mode="random_closed", level1_fraction=args.fraction,
            seed=seed * 1000 + vocab,
        )
        cvt = [
            assign_levels(
                flat, st, mode="morph", level1_fraction=args.fraction,
                seed=seed * 1000 + vocab,
            )
            for st in fold_stats
        ]
        # Size-matched frequency controls: closure can shrink a morph level 1,
        # so compare against a rank prefix of the *same* size.
        freqsize = {
            name: freq_token(flat, len(tok.levels[0]), seed * 1000 + vocab)
            for name, tok in toks.items()
        }
        cvt_size = [
            freq_token(flat, len(tok.levels[0]), seed * 1000 + vocab)
            for tok in cvt
        ]

        # Baselines on full held-out test words.
        base_f1, _ = f1_words(freq_tok, syn.test_words)
        freq_train_tokens = token_counts(freq_tok, train_counts)
        base_bpc = bpc_counts(freq_tok, train_counts, test_counts, test_char_len,
                              train_token_counts=freq_train_tokens)
        all_rows.append(row(seed, "rank", "flat", "test_all", flat,
                            syn.test_words, stats_freq, base_bpc))
        all_rows.append(row(seed, "rank", "frequency", "test_all", freq_tok,
                            syn.test_words, stats_freq, base_bpc))
        all_rows.append(row(seed, "rank", "random_closed", "test_all", rand_tok,
                            syn.test_words, stats_freq,
                            bpc_counts(rand_tok, train_counts, test_counts,
                                       test_char_len)))

        # Held-out training-word halves, baseline F1 on those word sets.
        all_rows.append(row(seed, "rank", "frequency", "train_halfA", freq_tok,
                            [words[i] for i in half_a], stats_freq))
        all_rows.append(row(seed, "rank", "frequency", "train_halfB", freq_tok,
                            [words[i] for i in half_b], stats_freq))

        # All-test evaluation for every provenance variant.
        for name, tok in toks.items():
            st = stats_variants[name]
            fs_tok = freqsize[name]
            all_rows.append(row(
                seed, name, "morph", "test_all", tok, syn.test_words, st,
                bpc_counts(tok, train_counts, test_counts, test_char_len),
                jaccard(level1_set(tok), level1_set(toks["freq_all"])),
            ))
            all_rows.append(row(
                seed, name, "frequency_size", "test_all", fs_tok,
                syn.test_words, st,
                bpc_counts(fs_tok, train_counts, test_counts, test_char_len),
            ))
            # Held-out training-word evaluation: A -> B and B -> A.
            if name == "freq_halfA" or name == "unique_halfA":
                all_rows.append(row(seed, name, "morph", "train_halfB", tok,
                                    [words[i] for i in half_b], st))
                all_rows.append(row(seed, name, "morph", "train_halfA", tok,
                                    [words[i] for i in half_a], st))
                all_rows.append(row(seed, name, "frequency_size", "train_halfB",
                                    fs_tok, [words[i] for i in half_b], st))
                all_rows.append(row(seed, name, "frequency_size", "train_halfA",
                                    fs_tok, [words[i] for i in half_a], st))
            if name == "freq_halfB" or name == "unique_halfB":
                all_rows.append(row(seed, name, "morph", "train_halfA", tok,
                                    [words[i] for i in half_a], st))
                all_rows.append(row(seed, name, "morph", "train_halfB", tok,
                                    [words[i] for i in half_b], st))
                all_rows.append(row(seed, name, "frequency_size", "train_halfA",
                                    fs_tok, [words[i] for i in half_a], st))
                all_rows.append(row(seed, name, "frequency_size", "train_halfB",
                                    fs_tok, [words[i] for i in half_b], st))

        # Root cross-fitting.
        test_by_root = defaultdict(list)
        test_word_root = {}
        for w in syn.test_words:
            test_by_root[w.root].append(w)
            test_word_root[w.word] = w.root
        tokens_by_root = defaultdict(list)
        for t in syn.test_text.split():
            r = test_word_root.get(t)
            if r is not None:
                tokens_by_root[r].append(t)

        cv_f1_sum = 0.0
        cv_n = 0
        cv_bits = 0.0
        cv_chars = 0
        base_cv_f1_sum = 0.0
        cv_size_f1_sum = 0.0
        cv_size_bits = 0.0
        for k, fold_roots in enumerate(root_folds):
            fold_words = []
            for r in fold_roots:
                fold_words.extend(test_by_root.get(r, []))
            if not fold_words:
                continue
            morph_f1, n = f1_words(cvt[k], fold_words)
            freq_f1, _ = f1_words(freq_tok, fold_words)
            size_f1, _ = f1_words(cvt_size[k], fold_words)
            cv_f1_sum += morph_f1 * n
            base_cv_f1_sum += freq_f1 * n
            cv_size_f1_sum += size_f1 * n
            cv_n += n

            fold_text = " ".join(t for r in fold_roots for t in tokens_by_root.get(r, []))
            if fold_text:
                norm_len = len(normalize(fold_text))
                fold_test_counts = Counter(fold_text.split())
                morph_bpc = bpc_counts(
                    cvt[k], train_counts, fold_test_counts, norm_len
                )
                freq_bpc = bpc_counts(
                    freq_tok, train_counts, fold_test_counts, norm_len,
                    train_token_counts=freq_train_tokens,
                )
                size_bpc = bpc_counts(
                    cvt_size[k], train_counts, fold_test_counts, norm_len
                )
                cv_bits += morph_bpc * norm_len
                cv_size_bits += size_bpc * norm_len
                cv_chars += norm_len
            else:
                morph_bpc = freq_bpc = size_bpc = float("nan")
                norm_len = 0

            fold_rows.append({
                "seed": seed,
                "fold": k,
                "n_fold_roots": len(fold_roots),
                "n_holdout_words": n,
                "morph_f1": morph_f1,
                "frequency_f1": freq_f1,
                "frequency_size_f1": size_f1,
                "delta_f1": morph_f1 - freq_f1,
                "delta_f1_size_matched": morph_f1 - size_f1,
                "n_l1": len(cvt[k].levels[0]),
                "morph_bpc": morph_bpc if norm_len else "",
                "frequency_bpc": freq_bpc if norm_len else "",
                "frequency_size_bpc": size_bpc if norm_len else "",
                "delta_bpc": (morph_bpc - freq_bpc) if norm_len else "",
                "delta_bpc_size_matched": (morph_bpc - size_bpc) if norm_len else "",
            })

        cv_f1 = cv_f1_sum / cv_n if cv_n else float("nan")
        cv_bpc = cv_bits / cv_chars if cv_chars else None
        cv_size_f1 = cv_size_f1_sum / cv_n if cv_n else float("nan")
        cv_size_bpc = cv_size_bits / cv_chars if cv_chars else None
        all_rows.append(row(
            seed, "root_cv", "morph", "test_all", cvt[0], syn.test_words,
            fold_stats[0], cv_bpc,
        ))
        # row() recomputes F1 with cvt[0]; overwrite the aggregate fields.
        all_rows[-1]["boundary_f1"] = cv_f1
        all_rows[-1]["eval_n_words"] = cv_n
        all_rows[-1]["n_l1"] = int(round(statistics.mean(len(t.levels[0]) for t in cvt)))
        all_rows[-1]["n_l2"] = int(round(statistics.mean(len(t.levels[1]) for t in cvt)))
        all_rows[-1]["l1_intra"] = ""
        all_rows[-1]["l1_cross"] = ""
        all_rows[-1]["l1_unseen"] = ""
        all_rows[-1]["l2_intra"] = ""
        all_rows[-1]["l2_cross"] = ""
        all_rows[-1]["l2_unseen"] = ""
        all_rows.append(row(
            seed, "root_cv", "frequency_size", "test_all", cvt_size[0],
            syn.test_words, fold_stats[0], cv_size_bpc,
        ))
        all_rows[-1]["boundary_f1"] = cv_size_f1
        all_rows[-1]["eval_n_words"] = cv_n
        all_rows[-1]["n_l1"] = int(round(statistics.mean(len(t.levels[0]) for t in cvt_size)))
        all_rows[-1]["n_l2"] = int(round(statistics.mean(len(t.levels[1]) for t in cvt_size)))
        all_rows[-1]["l1_intra"] = ""
        all_rows[-1]["l1_cross"] = ""
        all_rows[-1]["l1_unseen"] = ""
        all_rows[-1]["l2_intra"] = ""
        all_rows[-1]["l2_cross"] = ""
        all_rows[-1]["l2_unseen"] = ""
        print(
            f"seed={seed} freq_all F1={base_f1:.4f} "
            f"unique_all F1={f1_words(toks['unique_all'], syn.test_words)[0]:.4f} "
            f"halfA F1={f1_words(toks['freq_halfA'], syn.test_words)[0]:.4f} "
            f"halfB F1={f1_words(toks['freq_halfB'], syn.test_words)[0]:.4f} "
            f"rootCV F1={cv_f1:.4f} (freq {base_cv_f1_sum / cv_n:.4f})",
            flush=True,
        )

        # Assignment similarity diagnostics.
        pairs = [
            ("unique_all", "freq_all"),
            ("freq_halfA", "freq_halfB"),
            ("unique_halfA", "unique_halfB"),
            ("freq_halfA", "freq_all"),
            ("freq_halfB", "freq_all"),
            ("unique_halfA", "freq_all"),
            ("unique_halfB", "freq_all"),
        ]
        for a, b in pairs:
            sa, sb = level1_set(toks[a]), level1_set(toks[b])
            sim_rows.append({
                "seed": seed,
                "pair": f"{a}_vs_{b}",
                "jaccard_l1": jaccard(sa, sb),
                "n_l1_a": len(sa),
                "n_l1_b": len(sb),
                "n_symmetric_difference": len(sa ^ sb),
            })

        # Direct reproduction check against the published 10-seed CSV.
        ref_path = "results/level_assignment_v512_10seeds.csv"
        if os.path.exists(ref_path):
            with open(ref_path, encoding="utf-8") as fh:
                ref = {
                    (int(r["seed"]), r["mode"]): r
                    for r in csv.DictReader(fh)
                    if int(r["seed"]) == seed and int(r["vocab"]) == vocab
                }
            for mode, variant in (("flat", "rank"), ("frequency", "rank"),
                                  ("morph", "freq_all"), ("random_closed", "rank")):
                rr = ref.get((seed, mode))
                ours = next(
                    (r for r in all_rows
                     if r["seed"] == seed and r["variant"] == variant
                     and r["mode"] == mode and r["eval_set"] == "test_all"),
                    None,
                )
                if rr is None or ours is None:
                    continue
                check_rows.append({
                    "seed": seed,
                    "reproduction_mode": mode,
                    "reference_f1": float(rr["boundary_f1"]),
                    "ours_f1": ours["boundary_f1"],
                    "abs_diff_f1": abs(float(rr["boundary_f1"]) - ours["boundary_f1"]),
                    "reference_bpc": float(rr["bits_per_char"]),
                    "ours_bpc": ours["bits_per_char"],
                    "abs_diff_bpc": abs(float(rr["bits_per_char"]) - ours["bits_per_char"]),
                })

        # Flush partial results so an interrupted long run still leaves data.
        write_csv(os.path.join(args.out_dir, "w1_15_levels_weighting_runs.csv"), all_rows)
        write_csv(os.path.join(args.out_dir, "w1_15_levels_weighting_root_folds.csv"), fold_rows)
        write_csv(os.path.join(args.out_dir, "w1_15_levels_weighting_similarity.csv"), sim_rows)

    write_csv(os.path.join(args.out_dir, "w1_15_levels_weighting_checks.csv"), check_rows)
    analyze(all_rows, sim_rows, args.out_dir)
    print(f"\nwrote {args.out_dir}/w1_15_levels_weighting_*.csv")


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def find(rows, seed, variant, mode, eval_set):
    for r in rows:
        if (r["seed"] == seed and r["variant"] == variant
                and r["mode"] == mode and r["eval_set"] == eval_set):
            return r
    return None


def series(rows, variant, mode, eval_set, metric, base_variant="rank",
           base_mode="frequency", base_eval=None):
    base_eval = base_eval or eval_set
    out = {}
    for r in rows:
        if (r["variant"] == variant and r["mode"] == mode
                and r["eval_set"] == eval_set):
            b = find(rows, r["seed"], base_variant, base_mode, base_eval)
            if b is None:
                continue
            out[r["seed"]] = float(r[metric]) - float(b[metric])
    return [out[s] for s in sorted(out)]


def analyze(rows, sim_rows, out_dir):
    seeds = sorted({r["seed"] for r in rows})
    stats_out = []

    def add(label, deltas, metric):
        d = describe(deltas)
        stats_out.append({"comparison": label, "metric": metric, **d})
        print(
            f"  {label:<42} {metric:<13} "
            f"mean={d['mean']:+.4f} t={d['t']:+.2f} "
            f"CI95=[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] "
            f"W/L/T={d['wins']}/{d['losses']}/{d['ties']}"
        )

    def add_vs(label, variant, eval_set, metric):
        add(f"{label} [vs frequency]", series(
            rows, variant, "morph", eval_set, metric), metric)
        add(f"{label} [vs size-matched freq]", series(
            rows, variant, "morph", eval_set, metric,
            base_variant=variant, base_mode="frequency_size"), metric)

    print("\nPaired deltas across 5 seeds (morph - control)")
    for metric in ("boundary_f1", "bits_per_char"):
        add_vs("freq_all (published design)", "freq_all", "test_all", metric)
        add_vs("unique_all", "unique_all", "test_all", metric)
        add("random_closed control [vs frequency]", series(
            rows, "rank", "random_closed", "test_all", metric), metric)

        # Disjoint random halves: average the two half-estimates per seed.
        for tag, bv_a, bv_b, bm in (
            ("vs frequency", "rank", "rank", "frequency"),
            ("vs size-matched freq", "freq_halfA", "freq_halfB", "frequency_size"),
        ):
            a = series(rows, "freq_halfA", "morph", "test_all", metric,
                       base_variant=bv_a, base_mode=bm)
            b = series(rows, "freq_halfB", "morph", "test_all", metric,
                       base_variant=bv_b, base_mode=bm)
            add(f"freq_half mean(A,B) [{tag}]",
                [(x + y) / 2 for x, y in zip(a, b)], metric)
            add(f"freq_halfA [{tag}]", a, metric)
            add(f"freq_halfB [{tag}]", b, metric)

        for tag, bv_a, bv_b, bm in (
            ("vs frequency", "rank", "rank", "frequency"),
            ("vs size-matched freq", "unique_halfA", "unique_halfB", "frequency_size"),
        ):
            au = series(rows, "unique_halfA", "morph", "test_all", metric,
                        base_variant=bv_a, base_mode=bm)
            bu = series(rows, "unique_halfB", "morph", "test_all", metric,
                        base_variant=bv_b, base_mode=bm)
            add(f"unique_half mean(A,B) [{tag}]",
                [(x + y) / 2 for x, y in zip(au, bu)], metric)

        add_vs("root_cv (10-fold by root)", "root_cv", "test_all", metric)

    # Held-out training words: stats from A, evaluate on B, and vice versa.
    print("\nHeld-out training-word evaluation (provenance and eval words disjoint)")
    for metric in ("boundary_f1",):
        for tag, bv_a, bv_b, bm in (
            ("vs frequency", "rank", "rank", "frequency"),
            ("vs size-matched freq", "freq_halfA", "freq_halfB", "frequency_size"),
        ):
            ah = series(rows, "freq_halfA", "morph", "train_halfB", metric,
                        base_variant=bv_a, base_mode=bm)
            bh = series(rows, "freq_halfB", "morph", "train_halfA", metric,
                        base_variant=bv_b, base_mode=bm)
            add(f"freq_half A->B [{tag}]", ah, metric)
            add(f"freq_half B->A [{tag}]", bh, metric)
            add(f"freq_half mean, held-out words [{tag}]",
                [(x + y) / 2 for x, y in zip(ah, bh)], metric)
            ahu = series(rows, "unique_halfA", "morph", "train_halfB", metric,
                         base_variant="unique_halfA" if bm == "frequency_size" else bv_a,
                         base_mode=bm)
            bhu = series(rows, "unique_halfB", "morph", "train_halfA", metric,
                         base_variant="unique_halfB" if bm == "frequency_size" else bv_b,
                         base_mode=bm)
            add(f"unique_half mean, held-out words [{tag}]",
                [(x + y) / 2 for x, y in zip(ahu, bhu)], metric)

    # Direct effect of weighting, holding mode = morph.
    print("\nDirect weighting effect (unique - frequency-weighted provenance)")
    for metric in ("boundary_f1", "bits_per_char"):
        d = describe(series(
            rows, "unique_all", "morph", "test_all", metric,
            base_variant="freq_all", base_mode="morph", base_eval="test_all"))
        stats_out.append({"comparison": "unique_all vs freq_all assignment",
                          "metric": metric, **d})
        print(
            f"  unique_all - freq_all {metric:<13} "
            f"mean={d['mean']:+.4f} t={d['t']:+.2f} "
            f"CI95=[{d['ci_low']:+.4f},{d['ci_high']:+.4f}] "
            f"W/L/T={d['wins']}/{d['losses']}/{d['ties']}"
        )

    print("\nLevel-1 assignment similarity (Jaccard, mean over seeds)")
    for pair in sorted({r["pair"] for r in sim_rows}):
        vals = [float(r["jaccard_l1"]) for r in sim_rows if r["pair"] == pair]
        print(f"  {pair:<34} {statistics.mean(vals):.3f}")

    write_csv(os.path.join(out_dir, "w1_15_levels_weighting_stats.csv"), stats_out)


if __name__ == "__main__":
    main()
