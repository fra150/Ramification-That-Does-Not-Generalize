"""Independent verification of results and of the text codec.

This script does not modify the pipeline. It checks what the agents' CSVs
actually claim, and it quantifies a known limitation: the current
tokenize_normalized/detokenize pair drops whitespace, so word-level decode is
exact but document-level decode is not. The proposed exact codec is tested
side by side.

Usage:
    python -m experiments.verify --csv results/agent04_flat_code/results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re

from ramified.bpe import EOW, detokenize, tokenize_normalized, train_tokenizer
from ramified.synthetic import generate_synthetic


def check_csv(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    issues = []
    for i, r in enumerate(rows, start=2):
        if int(r["vocab_actual"]) != int(r["vocab_target"]):
            issues.append(f"line {i}: vocab_actual {r['vocab_actual']} != target {r['vocab_target']}")
        for col in ("tokens_per_char", "bits_per_char", "decode_exact"):
            v = float(r[col])
            if math.isnan(v) or math.isinf(v):
                issues.append(f"line {i}: {col} is not finite ({v})")
            if col == "decode_exact" and not (0.0 <= v <= 1.0):
                issues.append(f"line {i}: decode_exact out of [0,1] ({v})")
        for col in ("boundary_precision", "boundary_recall", "boundary_f1"):
            if r[col] not in ("", None):
                v = float(r[col])
                if not (0.0 <= v <= 1.0):
                    issues.append(f"line {i}: {col} out of [0,1] ({v})")
    return rows, issues


def ensure_space_terminal(tok):
    """The trainer builds the alphabet from non-whitespace words, so ' ' may
    be missing. Add it as a leaf for verification only (no merges change)."""
    if " " not in tok.char_to_id:
        new_id = max(tok.expansions) + 1
        tok.char_to_id[" "] = new_id
        tok.expansions[new_id] = " "
        tok.vocab_size = len(tok.expansions)


def tokenize_exact(text, tok):
    """Exact codec: whitespace is encoded as a real space character."""
    ensure_space_terminal(tok)
    text = re.sub(r"\s+", " ", text).strip()
    ids = []
    for m in re.finditer(r"\w+|[^\w\s]|\s+", text):
        s = m.group()
        if s.isspace():
            ids.append(tok.char_to_id[" "])
        elif re.fullmatch(r"\w+", s):
            ids.extend(tok.encode(s, eow=True))
        else:
            for c in s:
                ids.append(tok.char_to_id.get(c, tok.unk_id))
    return ids


def detokenize_exact(ids, tok):
    return tok.decode(ids).replace(EOW, "")


def tokenize_chunks(text, tok):
    """Chunk codec: split on spaces, EOW per chunk, no explicit space token."""
    text = re.sub(r"\s+", " ", text).strip()
    ids = []
    for chunk in text.split(" "):
        ids.extend(tok.encode(chunk, eow=True))
    return ids


def detokenize_chunks(ids, tok):
    return tok.decode(ids).replace(EOW, " ").strip()


def roundtrip_report(tok, text, label):
    norm = re.sub(r"\s+", " ", text).strip()

    ids_old = tokenize_normalized(norm, tok)
    out_old = detokenize(ids_old, tok)

    ids_new = tokenize_exact(norm, tok)
    out_new = detokenize_exact(ids_new, tok)

    ids_chunk = tokenize_chunks(norm, tok)
    out_chunk = detokenize_chunks(ids_chunk, tok)

    print(f"\n[{label}] chars={len(norm)}")
    print(f"  current codec exact: {out_old == norm}")
    if out_old != norm:
        for i, (a, b) in enumerate(zip(norm, out_old)):
            if a != b:
                print(f"    first diff at {i}: expected {norm[max(0,i-20):i+20]!r}")
                print(f"                      got      {out_old[max(0,i-20):i+20]!r}")
                break
        else:
            print(f"    length differs: expected {len(norm)}, got {len(out_old)}")
    print(f"  exact codec exact:   {out_new == norm}")
    print(f"  chunk codec exact:   {out_chunk == norm}")
    if out_chunk != norm:
        for i in range(min(len(norm), len(out_chunk))):
            if norm[i] != out_chunk[i]:
                print(f"    chunk first diff at {i}: expected {norm[max(0,i-20):i+20]!r}")
                print(f"                            got      {out_chunk[max(0,i-20):i+20]!r}")
                break
        else:
            print(f"    chunk length differs: expected {len(norm)}, got {len(out_chunk)}")
    print(f"  tokens old/new/chunk: {len(ids_old)} / {len(ids_new)} / {len(ids_chunk)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/agent04_flat_code/results.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=256)
    args = ap.parse_args()

    rows, issues = check_csv(args.csv)
    print(f"CSV: {args.csv} ({len(rows)} rows)")
    if issues:
        print("ISSUES:")
        for issue in issues:
            print("  -", issue)
    else:
        print("CSV integrity: OK (no schema/value issues found)")

    syn = generate_synthetic(seed=args.seed)
    tok = train_tokenizer([syn.train_text], vocab_size=args.vocab, n_levels=1)
    print(f"\ntokenizer: vocab={tok.vocab_size}, levels={len(tok.levels)}")

    roundtrip_report(tok, syn.train_text[:2000], "synthetic train sample")
    roundtrip_report(tok, syn.test_text[:2000], "synthetic held-out sample")

    # Leakage: held-out word forms must not occur in the training text.
    train_set = set(re.findall(r"\w+", syn.train_text))
    leaked = [w.word for w in syn.test_words if w.word in train_set]
    print(f"\nheld-out word types: {len(syn.test_words)}")
    print(f"leaked into train text: {len(leaked)}")
    if leaked:
        print(f"  examples: {leaked[:10]}")

    # Boundary sanity: generated boundaries must split prefix|root|suffix.
    bad = []
    for w in syn.test_words[:200]:
        if w.word != w.prefix + w.root + w.suffix:
            bad.append((w.word, "concat"))
        elif w.boundaries != (len(w.prefix), len(w.prefix) + len(w.root)):
            bad.append((w.word, "boundaries"))
    print(f"boundary sanity on 200 words: {'OK' if not bad else bad[:5]}")


if __name__ == "__main__":
    main()
