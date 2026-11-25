"""Figure for the Fase 1f paper: boundary F1 by cell and paired deltas.

Reads the four 2x2 CSVs, computes mean and 95% CI over seeds, and writes
``paper/figures/f1f_2x2.png``. No statistics are invented: the same paired
t critical value used in the diary (2.262 for 10 seeds) is used here.
"""

from __future__ import annotations

import csv
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS = [
    ("variable\n× triple", "results/f1f_variable_triple.csv"),
    ("variable\n× pair", "results/f1f_variable_pair.csv"),
    ("fixed\n× triple", "results/f1f_fixed_triple.csv"),
    ("fixed\n× pair", "results/f1f_fixed_pair.csv"),
]
MODES = ["frequency", "morph", "const2", "near2"]
MODE_LABELS = {
    "frequency": "frequency / flat",
    "morph": "morph",
    "const2": "morph + const2",
    "near2": "morph + near2",
}
MODE_KEYS = {
    "frequency": "frequency",
    "morph": "morph",
    "const2": "morph_const2",
    "near2": "morph_near2",
}
COLORS = {
    "frequency": "#6c757d",
    "morph": "#0b6e4f",
    "const2": "#c0392b",
    "near2": "#e08e0b",
}
T95 = 2.262


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def mean_ci(values):
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, 0.0
    sd = statistics.stdev(values)
    return mean, T95 * sd / math.sqrt(n)


def main():
    data = {}
    for cell, path in CELLS:
        rows = load(path)
        by_mode = {}
        for mode in MODES:
            key = MODE_KEYS[mode]
            vals = [float(r["boundary_f1"]) for r in rows if r["mode"] == key]
            by_mode[mode] = vals
        data[cell] = by_mode

    os.makedirs("paper/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

    # Panel A: mean F1 by mode.
    ax = axes[0]
    width = 0.2
    xs = range(len(CELLS))
    for j, mode in enumerate(MODES):
        means, errs = [], []
        for cell, _ in CELLS:
            m, e = mean_ci(data[cell][mode])
            means.append(m)
            errs.append(e)
        positions = [x + (j - 1.5) * width for x in xs]
        ax.bar(
            positions,
            means,
            width=width,
            label=MODE_LABELS[mode],
            color=COLORS[mode],
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([cell for cell, _ in CELLS])
    ax.set_ylim(0.845, 0.902)
    ax.set_ylabel("boundary F1 (macro)")
    ax.set_title("A. Boundary F1 by cell (mean, 95% CI, 10 seeds)", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
    )

    # Panel B: paired deltas against the same-seed frequency baseline.
    ax = axes[1]
    comparisons = [("morph", "morph − frequency"), ("near2", "morph − near2"), ("const2", "morph − const2")]
    for j, (mode, label) in enumerate(comparisons):
        deltas = []
        for cell, _ in CELLS:
            base = data[cell]["frequency"]
            vals = data[cell][mode]
            deltas.append([a - b for a, b in zip(vals, base)])
        means, errs = zip(*[mean_ci(d) for d in deltas])
        positions = [x + (j - 1) * 0.14 for x in xs]
        ax.errorbar(
            positions,
            means,
            yerr=errs,
            fmt="o",
            markersize=6,
            capsize=4,
            linewidth=1.4,
            label=label,
            color=COLORS[mode],
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([cell for cell, _ in CELLS])
    ax.set_ylabel("Δ boundary F1 vs frequency")
    ax.set_title("B. Paired differences (mean, 95% CI, 10 seeds)", fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    out = "paper/figures/f1f_2x2.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
