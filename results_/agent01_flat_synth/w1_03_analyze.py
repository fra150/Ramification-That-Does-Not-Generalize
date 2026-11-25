"""Render W1-03 tables from the red-team CSVs (markdown to stdout)."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

COND_ORDER = ["baseline", "uniform", "noperiod", "uniform_noperiod"]
MODE_ORDER = ["flat", "frequency", "morph", "random_closed", "morph_type"]


def fmt(x, digits=4):
    if pd.isna(x):
        return "n/a"
    return f"{x:+.{digits}f}" if x < 0 else f"{x:.{digits}f}"


def fmt_signed(x, digits=4):
    if pd.isna(x):
        return "n/a"
    return f"{x:+.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/redteam")
    args = ap.parse_args()

    runs = pd.read_csv(os.path.join(args.dir, "w1_03_runs.csv"))
    stats = pd.read_csv(os.path.join(args.dir, "w1_03_stats.csv"))
    corpus = pd.read_csv(os.path.join(args.dir, "w1_03_corpus_stats.csv"))
    runs["condition"] = pd.Categorical(runs["condition"], COND_ORDER, ordered=True)
    runs["mode"] = pd.Categorical(runs["mode"], MODE_ORDER, ordered=True)
    seeds = sorted(runs["seed"].unique())

    print(f"# W1-03 tables ({len(seeds)} seeds: {seeds})\n")

    print("## Corpus statistics (mean over seeds)\n")
    agg = (
        corpus.groupby("condition", observed=True)
        .agg(
            word_tokens=("train_word_tokens", "mean"),
            periods=("train_periods", "mean"),
            types=("train_types", "mean"),
            coverage=("type_coverage", "mean"),
            top1=("top1_share", "mean"),
            top10=("top10_share", "mean"),
            entropy=("entropy_bits_per_word", "mean"),
            alphabet=("alphabet_size", "mean"),
        )
        .reindex(COND_ORDER)
    )
    print(
        "| condition | word tokens | periods | types | coverage | top-1 share | top-10 share | entropy (bits/word) | alphabet |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cond, row in agg.iterrows():
        print(
            f"| {cond} | {row.word_tokens:.0f} | {row.periods:.0f} | "
            f"{row.types:.0f} | {row.coverage:.3f} | {row.top1:.4f} | "
            f"{row.top10:.4f} | {row.entropy:.2f} | {row.alphabet:.0f} |"
        )

    print("\n## Mean metrics by condition and mode (mean over seeds)\n")
    for metric, digits in (("boundary_f1", 4), ("bits_per_char", 4)):
        print(f"### {metric}\n")
        piv_mean = runs.pivot_table(
            index="condition", columns="mode", values=metric, aggfunc="mean", observed=True
        ).reindex(index=COND_ORDER, columns=MODE_ORDER)
        piv_sd = runs.pivot_table(
            index="condition", columns="mode", values=metric, aggfunc="std", observed=True
        ).reindex(index=COND_ORDER, columns=MODE_ORDER)
        header = "| condition | " + " | ".join(MODE_ORDER) + " |"
        print(header)
        print("|---|" + "---:|" * len(MODE_ORDER))
        for cond in COND_ORDER:
            cells = []
            for mode in MODE_ORDER:
                m = piv_mean.loc[cond, mode]
                s = piv_sd.loc[cond, mode]
                cells.append("n/a" if pd.isna(m) else f"{m:.{digits}f} ({s:.{digits}f})")
            print(f"| {cond} | " + " | ".join(cells) + " |")
        print()

    print("## Flat (frequency-level) metric by seed and condition\n")
    flat = runs[runs["mode"] == "flat"]
    for metric in ("boundary_f1", "bits_per_char"):
        print(f"### {metric}\n")
        piv = flat.pivot(index="seed", columns="condition", values=metric)
        print("| seed | " + " | ".join(COND_ORDER) + " |")
        print("|---:|" + "---:|" * len(COND_ORDER))
        for seed, row in piv.iterrows():
            print(f"| {seed} | " + " | ".join(f"{row[c]:.4f}" for c in COND_ORDER) + " |")
        print()

    print("## Per-seed morph - frequency deltas\n")
    for metric in ("boundary_f1", "bits_per_char"):
        print(f"### {metric}\n")
        piv = runs[runs["mode"].isin(["morph", "frequency"])].pivot_table(
            index="seed", columns=["condition", "mode"], values=metric, observed=True
        )
        print("| seed | " + " | ".join(COND_ORDER) + " |")
        print("|---:|" + "---:|" * len(COND_ORDER))
        for seed, row in piv.iterrows():
            cells = []
            for cond in COND_ORDER:
                cells.append(f"{row[(cond, 'morph')] - row[(cond, 'frequency')]:+.4f}")
            print(f"| {seed} | " + " | ".join(cells) + " |")
        print()

    print("## Paired deltas (5 seeds; CI is 95% t)\n")
    for metric in ("boundary_f1", "bits_per_char"):
        print(f"### {metric}\n")
        sub = stats[
            (stats["kind"] == "paired_mode") & (stats["metric"] == metric)
        ]
        print("| condition | comparison | mean delta | 95% CI | t | W/L/T |")
        print("|---|---|---:|---|---:|---:|")
        for cond in COND_ORDER:
            for _, row in sub[sub["condition"] == cond].iterrows():
                comp = f"{row['mode_a']} - {row['mode_b']}"
                print(
                    f"| {cond} | {comp} | {fmt_signed(row['mean'])} | "
                    f"[{fmt_signed(row['ci_low'])}, {fmt_signed(row['ci_high'])}] | "
                    f"{row['t']:.2f} | {int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])} |"
                )
        print()

    print("## Did the effect shift between corpora? (paired by seed)\n")
    for metric in ("boundary_f1", "bits_per_char"):
        print(f"### {metric}\n")
        sub = stats[
            (stats["kind"] == "effect_shift") & (stats["metric"] == metric)
        ]
        print("| comparison of morph-frequency effect | mean shift | 95% CI | t | p |")
        print("|---|---:|---|---:|---:|")
        for _, row in sub.iterrows():
            print(
                f"| {row['condition']} | {fmt_signed(row['mean'])} | "
                f"[{fmt_signed(row['ci_low'])}, {fmt_signed(row['ci_high'])}] | "
                f"{row['t']:.2f} | {row['p']:.4f} |"
            )
        print()

    print("## Level-1 composition and tokenization change (mean over seeds)\n")
    sub = runs[runs["mode"].isin(["frequency", "morph", "random_closed"])]
    g = (
        sub.groupby(["condition", "mode"], observed=True)
        .agg(
            n_l1=("n_l1", "mean"),
            l1_intra=("l1_intra", "mean"),
            l1_cross=("l1_cross", "mean"),
            l2_intra=("l2_intra", "mean"),
            l2_cross=("l2_cross", "mean"),
            diff=("frac_test_words_diff_vs_frequency", "mean"),
        )
        .reindex(
            pd.MultiIndex.from_product(
                [COND_ORDER, ["frequency", "morph", "random_closed"]],
                names=["condition", "mode"],
            )
        )
    )
    print(
        "| condition | mode | L1 rules | L1 intra | L1 cross | L1 intra share | L2 intra | L2 cross | test words changed vs frequency |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (cond, mode), row in g.iterrows():
        share = row["l1_intra"] / row["n_l1"] if row["n_l1"] else float("nan")
        print(
            f"| {cond} | {mode} | {row['n_l1']:.0f} | {row['l1_intra']:.0f} | "
            f"{row['l1_cross']:.0f} | {share:.3f} | {row['l2_intra']:.0f} | "
            f"{row['l2_cross']:.0f} | {row['diff']:.3f} |"
        )


if __name__ == "__main__":
    main()
