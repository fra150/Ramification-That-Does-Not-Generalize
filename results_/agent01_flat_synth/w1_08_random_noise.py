"""W1-08 side check: how noisy is the random_closed control by itself?

Holds the flat tokenizer fixed (id tie-break) and draws 20 random_closed
assignments per seed. This isolates random-assignment noise from tie-break
noise: no retraining, only the level assignment seed changes.
"""

from __future__ import annotations

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from ramified.bpe import train_tokenizer  # noqa: E402
from ramified.levels import assign_levels, collect_provenance  # noqa: E402
from ramified.synthetic import generate_synthetic  # noqa: E402
from experiments.level_assignment import evaluate  # noqa: E402
from w1_08_tiebreak import install_variant  # noqa: E402

VOCAB = 512
FRACTION = 0.6
N_DRAWS = 20


def main():
    for seed in (42, 43):
        syn = generate_synthetic(seed=seed)
        install_variant("id")
        flat = train_tokenizer(
            [syn.train_text], vocab_size=VOCAB, n_levels=1, split="chunks"
        )
        stats = collect_provenance(flat, syn.train_word_objects, text=syn.train_text)
        base_f1 = evaluate(flat, syn)["boundary_f1"]
        morph = assign_levels(
            flat, stats, mode="morph", level1_fraction=FRACTION,
            seed=seed * 1000 + VOCAB,
        )
        morph_m = evaluate(morph, syn)
        rows = []
        for k in range(N_DRAWS):
            tok = assign_levels(
                flat,
                stats,
                mode="random_closed",
                level1_fraction=FRACTION,
                seed=seed * 1000 + VOCAB + 7919 * (k + 1),
            )
            m = evaluate(tok, syn)
            rows.append(m)
            print(
                f"seed={seed} draw={k:02d} F1={m['boundary_f1']:.6f} "
                f"bpc={m['bits_per_char']:.6f} L1={len(tok.levels[0])}",
                flush=True,
            )
        f1 = [r["boundary_f1"] for r in rows]
        bpc = [r["bits_per_char"] for r in rows]
        d_f1 = [morph_m["boundary_f1"] - x for x in f1]
        d_bpc = [morph_m["bits_per_char"] - x for x in bpc]
        print(
            f"\nseed={seed}: flat F1={base_f1:.6f}, morph F1={morph_m['boundary_f1']:.6f}"
        )
        print(
            f"  random_closed F1: mean={statistics.mean(f1):.6f} "
            f"sd={statistics.stdev(f1):.6f} range={max(f1)-min(f1):.6f}"
        )
        print(
            f"  random_closed bpc: mean={statistics.mean(bpc):.6f} "
            f"sd={statistics.stdev(bpc):.6f} range={max(bpc)-min(bpc):.6f}"
        )
        print(
            f"  morph - random F1: mean={statistics.mean(d_f1):+.6f} "
            f"sd={statistics.stdev(d_f1):.6f} "
            f"wins={sum(1 for d in d_f1 if d > 0)}/{N_DRAWS}"
        )
        print(
            f"  morph - random bpc: mean={statistics.mean(d_bpc):+.6f} "
            f"sd={statistics.stdev(d_bpc):.6f} "
            f"wins={sum(1 for d in d_bpc if d < 0)}/{N_DRAWS}"
        )


if __name__ == "__main__":
    main()
