"""Local real-text corpora used as compression baselines."""

from __future__ import annotations

import glob
import os
import sysconfig


def load_code_corpus(max_chars: int = 400_000) -> str:
    """Concatenate standard-library Python files into a real text corpus."""
    lib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(lib, "*.py")))
    parts = []
    total = 0
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars]
