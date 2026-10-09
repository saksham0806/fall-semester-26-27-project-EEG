"""Nested cross-validation splits — metd_plan.md §6.1, §6.2.

Every split groups on ``subject_id``. Segments are 50%-overlapping windows, so a random
split puts the same person on both sides: measured on the old pipeline, 0.9393 random
against 0.6256 subject-grouped, same model. If a number jumps above ~0.85, suspect the
split before believing the model.

Folds are built PER CORPUS and unioned, never over the pooled array. Pooled splitting
balances group sizes, not corpora, so a fold can end up with no MODMA at all — and then
the depression head trains or validates on nothing while reporting a plausible number.

    python -m arch_metd.train.folds
"""
import hashlib
import json
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ..contract import BY_DOMAIN

TRAIN_DOMAINS = (0, 1, 2, 3)  # EEGMAT (4) is the held-out amplifier — never trained on

# §6.2, §7.5. Each runs with its own lambda = 0 control.
TRANSFERS = {
    "stress_sam40cog_to_eegmat": ((1, 2), (4,)),   # primary stress generalization
    "stress_eegmat_to_sam40cog": ((4,), (1, 2)),   # diagnostic only, EEGMAT is small
    "dep_modma_to_mumtaz": ((0,), (3,)),           # primary depression transfer
    "dep_mumtaz_to_modma": ((3,), (0,)),           # reciprocal direction
}


def greedy_group_folds(groups, n_splits):
    """Deterministic group -> fold assignment. Largest group first, into the lightest fold.

    Replaces sklearn's `GroupKFold`, whose result is not a property of the data alone:
    with `shuffle=False` it assigns greedily by group size and breaks ties with an argsort
    whose stability has changed across versions. Every SAM40 subject holds exactly 12
    trials, so its groups tie everywhere and the assignment followed the library version —
    measured, the workstation and this machine put 17 of 169 subjects in different folds
    running identical code.

    Ties here break on the group id, which is data. Pinning the library instead would
    version-lock an arbitrary tie-break and call it reproducible.
    """
    sizes = Counter(groups.tolist())
    load = [0] * n_splits
    assign = {}
    for g in sorted(sizes, key=lambda g: (-sizes[g], str(g))):
        f = min(range(n_splits), key=lambda i: (load[i], i))
        assign[g] = f
        load[f] += sizes[g]
    return assign


def split_hash(d, **kw):
    """Identity of a split, so two runs are immediately comparable or distinguishable."""
    sf = {k: int(v) for k, v in sorted(subject_folds(d, **kw).items())}
    return hashlib.sha256(json.dumps(sf).encode()).hexdigest()[:16]


def _corpus_fold_ids(d, domain, n_splits, seed):
    """Fold index per segment of one corpus.

    The split is computed on a CANONICALLY ORDERED view — sorted by subject id, ties by
    original position — and mapped back. Without that, fold assignment depends on the row
    order of the arrays: `GroupKFold` balances by group size and breaks ties in encounter
    order, and every SAM40 subject has exactly 12 trials, so its groups tie everywhere.
    Measured before this fix: two builds of the same corpora, differing only in row order,
    put 4-10 SAM40/CogLoad subjects in different folds. Both splits were valid and
    subject-disjoint; they were simply not the same split, so no stress number was
    reproducible across runs.
    """
    sel = np.flatnonzero(d["y_domain"] == domain)
    groups = d["subject_id"][sel]
    assert len(set(groups)) >= n_splits, f"{BY_DOMAIN[domain].name}: too few subjects"

    if BY_DOMAIN[domain].task == "depression":
        # the label is a per-subject trait, so plain k-fold on 53 subjects can hand you
        # a fold with almost no MDD — stratify (§6.1). Run on a canonically ordered view
        # so the result cannot depend on array row order.
        order = np.lexsort((sel, groups))
        sel_c, groups_c = sel[order], groups[order]
        folds_c = np.empty(len(sel), int)
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        for i, (_, test) in enumerate(splitter.split(sel_c, d["y_dep"][sel_c], groups_c)):
            folds_c[test] = i
        out = np.empty(len(sel), int)
        out[order] = folds_c
        return sel, out

    # every stress subject holds both classes, so no subject-level label exists to
    # stratify on — deterministic greedy assignment instead of sklearn's GroupKFold
    assign = greedy_group_folds(groups, n_splits)
    return sel, np.array([assign[g] for g in groups])


def outer_folds(d, domains=TRAIN_DOMAINS, n_splits=5, seed=42):
    """Joint outer folds: per-corpus splits unioned, so every fold spans every corpus."""
    fold = np.full(len(d["y_domain"]), -1, int)
    for domain in domains:
        sel, ids = _corpus_fold_ids(d, domain, n_splits, seed)
        fold[sel] = ids
    for i in range(n_splits):
        yield np.flatnonzero((fold >= 0) & (fold != i)), np.flatnonzero(fold == i)


def inner_folds(d, train_idx, n_splits=4, seed=42):
    """Inner folds for early stopping, lambda scheduling and threshold calibration.

    Built per corpus like the outer folds. §6.1 says "GroupKFold(4) on training
    subjects"; done pooled, an inner validation split can miss a corpus entirely and the
    head for that corpus then early-stops on no signal.
    """
    sub = {name: d[name][train_idx] for name in ("y_domain", "subject_id", "y_dep")}
    fold = np.full(len(train_idx), -1, int)
    for domain in np.unique(sub["y_domain"]):
        sel = np.flatnonzero(sub["y_domain"] == domain)
        groups = sub["subject_id"][sel]
        if len(set(groups)) < n_splits:
            # §6.1: CogLoad has 14-15 subjects; a 3-subject inner split gives uselessly
            # noisy early stopping. Such a corpus rides along in training only.
            continue
        assign = greedy_group_folds(groups, n_splits)
        fold[sel] = [assign[g] for g in groups]
    for i in range(n_splits):
        yield train_idx[(fold >= 0) & (fold != i)], train_idx[fold == i]


def loso(d, domain):
    """Leave-one-subject-out over one corpus (§6.1 for CogLoad, §7.4 for its metric)."""
    sel = np.flatnonzero(d["y_domain"] == domain)
    subjects = d["subject_id"][sel]
    for s in sorted(set(subjects)):
        yield sel[subjects != s], sel[subjects == s]


def transfer(d, name):
    """A named cross-corpus run (§7.5). Train and test corpora are disjoint by design."""
    train_domains, test_domains = TRANSFERS[name]
    assert not set(train_domains) & set(test_domains)
    return (np.flatnonzero(np.isin(d["y_domain"], train_domains)),
            np.flatnonzero(np.isin(d["y_domain"], test_domains)))


def _no_overlap(d, train, test, what):
    assert not (set(d["subject_id"][train]) & set(d["subject_id"][test])), \
        f"{what}: subject in both train and test"
    assert not (set(train.tolist()) & set(test.tolist())), f"{what}: segment in both"


def subject_folds(d, **kw):
    """{subject_id: outer fold index} — the split, independent of row order."""
    out = {}
    for i, (_, test) in enumerate(outer_folds(d, **kw)):
        for s in np.unique(d["subject_id"][test]):
            out[s] = i
    return out


if __name__ == "__main__":
    from ..data.build import load

    d = load()

    # §6.5 item 1: the split must depend only on WHICH subjects exist, never on how the
    # rows happen to be laid out. Three permutations, because one could get lucky.
    base = subject_folds(d)
    for seed in (1, 2, 3):
        order = np.random.default_rng(seed).permutation(len(d["y_domain"]))
        shuffled = {k: v[order] for k, v in d.items()}
        moved = {s: (base[s], f) for s, f in subject_folds(shuffled).items() if base[s] != f}
        assert not moved, (f"row order changed the split for {len(moved)} subjects, "
                           f"e.g. {list(moved)[:5]}")
    print(f"  split is row-order invariant over 3 permutations ({len(base)} subjects)")

    # the greedy assignment must balance folds, or a "deterministic" split could be
    # deterministic and useless
    for domain in (1, 2):
        sel = np.flatnonzero(d["y_domain"] == domain)
        a = greedy_group_folds(d["subject_id"][sel], 5)
        per = Counter(a.values())
        assert len(per) == 5, f"{BY_DOMAIN[domain].name}: a fold got no subjects"
        sizes = [sum((d["subject_id"][sel] == g).sum() for g in a if a[g] == f)
                 for f in range(5)]
        spread = (max(sizes) - min(sizes)) / np.mean(sizes)
        assert spread < 0.25, f"{BY_DOMAIN[domain].name}: fold sizes differ by {spread:.0%}"
        print(f"  {BY_DOMAIN[domain].name:8s} greedy folds: subjects {sorted(per.values())}, "
              f"segments {sizes}, spread {spread:.1%}")
    print(f"  split hash {split_hash(d)}")
    assert set(np.unique(d["y_domain"])) <= {0, 1, 2, 3, 4}, "domain 5 is reserved"

    for i, (train, test) in enumerate(outer_folds(d)):
        _no_overlap(d, train, test, f"outer fold {i}")
        assert 4 not in d["y_domain"][train], "EEGMAT leaked into a training fold"
        assert 4 not in d["y_domain"][test], "EEGMAT is evaluated separately, not in CV"
        assert set(np.unique(d["y_domain"][test])) == set(TRAIN_DOMAINS), \
            f"outer fold {i} does not span every corpus"
        for j, (itr, ival) in enumerate(inner_folds(d, train)):
            _no_overlap(d, itr, ival, f"inner fold {i}.{j}")
            assert not (set(d["subject_id"][ival]) & set(d["subject_id"][test])), \
                "inner validation touches an outer test subject"
        print(f"  outer {i}: train {len(train):6d} / test {len(test):5d} segments, "
              f"{len(set(d['subject_id'][test])):3d} test subjects, "
              f"domains {sorted(set(d['y_domain'][test].tolist()))}")

    n = sum(1 for _ in loso(d, 2))
    for train, test in loso(d, 2):
        _no_overlap(d, train, test, "cogload loso")
    print(f"  cogload LOSO: {n} folds")

    for name in TRANSFERS:
        train, test = transfer(d, name)
        _no_overlap(d, train, test, name)
        tr, te = sorted(set(d["y_domain"][train].tolist())), sorted(set(d["y_domain"][test].tolist()))
        assert not set(tr) & set(te)
        # depression transfer is the reason this assertion exists: MODMA and Mumtaz must
        # never sit on the same side, or "transfer" is measuring within-corpus CV
        assert not ({0, 3} <= set(tr)) and not ({0, 3} <= set(te)), name
        print(f"  {name:28s} train domains {tr} -> test {te}  "
              f"({len(train):6d} / {len(test):5d} segments)")

    print("phase 4 gate: pass")
