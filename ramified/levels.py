"""Level assignment for the ramified tokenizer.

The v0.2 hierarchy was a budget partition: level 1 was the first b1 merges of
the same greedy sequence, level 2 the continuation. With a correct BPE those
two levels produce exactly the flat tokenization. The levels were labels, not
constraints.

This module tests the actual claim. It keeps the merge rules and the vocabulary
fixed, and changes only the level assignment:

- ``morph``: rules that merge inside a morpheme go to level 1; rules that cross
  a morpheme boundary go to level 2.
- ``reverse``: the opposite (a control for "any reordering helps").
- ``random``: a random assignment with the same level-1 fraction (a control for
  "the specific morphological assignment matters").

A level-1 rule may only depend on level-1 rules. If a dependency is in level 2,
the rule is demoted, otherwise it could never fire. The actual level sizes are
reported; they can differ from the requested fraction after closure.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .bpe import EOW, Tokenizer


def _walk(node, start, tok, boundaries, stats):
    """Walk a derivation tree, record rule provenance, return the end offset."""
    if not node.children:
        expansion = tok.expansions[node.symbol]
        if expansion == EOW:
            return start
        return start + len(expansion)

    end = start
    for child in node.children:
        end = _walk(child, end, tok, boundaries, stats)

    if len(node.children) == 2:
        left, right = node.children
        key = (left.symbol, right.symbol)
        crosses = any(start < b < end for b in boundaries)
        intra, cross = stats[key]
        if crosses:
            stats[key] = (intra, cross + 1)
        else:
            stats[key] = (intra + 1, cross)
    return end


def collect_provenance(tok, word_objects, text=None):
    """Count, for every merge rule, intra- and cross-morpheme occurrences.

    ``tok`` must be a flat tokenizer (one level) so the derivation tree uses
    the full rule set. If ``text`` is given, word occurrences are weighted by
    their frequency in that sampled text; otherwise every unique word counts
    once.
    """
    if len(tok.levels) != 1:
        raise ValueError("collect_provenance expects a flat tokenizer")

    stats = defaultdict(lambda: (0, 0))
    if text is not None:
        by_word = {w.word: w for w in word_objects}
        items = [by_word[t] for t in text.split() if t in by_word]
    else:
        items = list(word_objects)

    for word in items:
        pos = 0
        for node in tok.encode_nodes(word.word, eow=True):
            pos = _walk(node, pos, tok, word.boundaries, stats)
    return dict(stats)


def _creator_map(tok):
    creator = {}
    for level in tok.levels:
        for rule in level:
            creator[rule.new] = rule
    terminals = set(tok.char_to_id.values()) | {tok.unk_id, tok.eow_id}
    return creator, terminals


def _score(rule, stats, mode, rng):
    if mode == "random":
        return rng.random()
    intra, cross = stats.get((rule.left, rule.right), (0, 0))
    total = intra + cross
    if total == 0:
        return 0.5
    if mode == "morph":
        return intra / total
    if mode == "reverse":
        return cross / total
    raise ValueError(f"unknown mode: {mode}")


def assign_levels(tok, stats, mode="morph", level1_fraction=0.6, seed=42):
    """Return a new tokenizer with the same rules assigned to two levels."""
    if len(tok.levels) != 1:
        raise ValueError("assign_levels expects a flat tokenizer")

    rules = sorted(tok.levels[0], key=lambda m: m.rank)
    n1 = int(level1_fraction * len(rules))
    n1 = max(1, min(len(rules) - 1, n1))
    creator, terminals = _creator_map(tok)
    rng = random.Random(seed)

    if mode == "frequency":
        # The current budget split: the first n1 rules by rank, the rest in
        # level 2. It is closed by construction and is the control the project
        # has been using so far.
        level1 = {m.new for m in rules[:n1]}
    elif mode == "random_closed":
        # A size-matched random control. Any prefix of a topological order is
        # downward closed, so taking the first n1 rules of a random linear
        # extension gives a closed level 1 of exactly n1 rules.
        rule_by_new = {m.new: m for m in rules}
        deps = {m.new: set() for m in rules}
        dependents = defaultdict(list)
        for rule in rules:
            for symbol in (rule.left, rule.right):
                if symbol in terminals:
                    continue
                dependency = creator.get(symbol)
                if dependency is not None:
                    deps[rule.new].add(dependency.new)
                    dependents[dependency.new].append(rule.new)
        available = [m for m in rules if not deps[m.new]]
        rng.shuffle(available)
        emitted = set()
        order = []
        while available:
            idx = rng.randrange(len(available))
            rule = available.pop(idx)
            order.append(rule)
            emitted.add(rule.new)
            for child in dependents[rule.new]:
                if child not in emitted and deps[child].issubset(emitted):
                    available.append(rule_by_new[child])
        order.extend(m for m in rules if m.new not in emitted)
        level1 = {m.new for m in order[:n1]}
    else:
        scored = [(m, _score(m, stats, mode, rng)) for m in rules]
        order = sorted(scored, key=lambda pair: (-pair[1], pair[0].rank))
        level1 = {m.new for m, _ in order[:n1]}

        # Dependency closure: a level-1 rule whose left or right symbol is
        # created by a level-2 rule can never fire. Demote it. Rules are
        # processed in rank order, and dependencies always have a lower rank.
        for rule in rules:
            if rule.new not in level1:
                continue
            for symbol in (rule.left, rule.right):
                if symbol in terminals:
                    continue
                dependency = creator.get(symbol)
                if dependency is not None and dependency.new not in level1:
                    level1.discard(rule.new)
                    break

    l1 = [m for m in rules if m.new in level1]
    l2 = [m for m in rules if m.new not in level1]
    return Tokenizer(
        expansions=dict(tok.expansions),
        char_to_id=dict(tok.char_to_id),
        eow_id=tok.eow_id,
        levels=[l1, l2],
        unk_id=tok.unk_id,
    )


def level_summary(tok, stats):
    """How many rules in each level are intra vs cross by majority."""
    out = []
    for level_idx, level in enumerate(tok.levels):
        intra = cross = unseen = 0
        for rule in level:
            a, b = stats.get((rule.left, rule.right), (0, 0))
            if a + b == 0:
                unseen += 1
            elif a >= b:
                intra += 1
            else:
                cross += 1
        out.append(
            {
                "level": level_idx,
                "rules": len(level),
                "intra": intra,
                "cross": cross,
                "unseen": unseen,
            }
        )
    return out
