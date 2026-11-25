"""Small unit checks that the tie-break variants do what they claim.

No source edits: imports the redteam copy and the original train_bpe and
compares them on tied toy sequences.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from ramified import bpe  # noqa: E402
from w1_08_tiebreak import train_bpe_variant  # noqa: E402


def rules(merges):
    return [(m.left, m.right, m.count) for m in merges]


def main():
    # (1,2) and (3,4) both have count 1: the tie is real.
    seqs = [[1, 2], [3, 4]]
    freqs = [1, 1]
    expansions = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}

    original, _ = bpe.train_bpe(
        [list(s) for s in seqs], freqs, 1, 5, 0, dict(expansions)
    )
    ident, _ = train_bpe_variant(
        seqs, freqs, 1, 5, 0, dict(expansions), variant="id"
    )
    rev, _ = train_bpe_variant(
        seqs, freqs, 1, 5, 0, dict(expansions), variant="rev_id"
    )
    print("source train_bpe   :", rules(original))
    print("copy variant id    :", rules(ident))
    print("copy variant rev_id:", rules(rev))
    assert rules(original) == rules(ident), "id copy must equal the source"
    assert rules(rev) != rules(ident), "rev_id must flip the tie"

    # Expansion-length tie-break: (1,2) expands to 2 chars, (3,5) to 6 chars.
    seqs2 = [[1, 2], [3, 5]]
    freqs2 = [1, 1]
    expansions2 = {1: "a", 2: "b", 3: "cccc", 5: "dd"}
    short, _ = train_bpe_variant(
        seqs2, freqs2, 1, 6, 0, dict(expansions2), variant="exp_short"
    )
    long, _ = train_bpe_variant(
        seqs2, freqs2, 1, 6, 0, dict(expansions2), variant="exp_long"
    )
    print("exp_short first merge:", rules(short))
    print("exp_long  first merge:", rules(long))
    assert short[0].left == 1 and short[0].right == 2
    assert long[0].left == 3 and long[0].right == 5

    print("OK: variants behave as specified and id reproduces the source")


if __name__ == "__main__":
    main()
