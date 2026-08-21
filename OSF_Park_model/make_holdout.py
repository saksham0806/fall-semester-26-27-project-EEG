"""
Create the frozen dev/test split.  Run ONCE per task; refuses to overwrite without --force.

    .venv/bin/python -m OSF_Park_model.make_holdout

A regenerated split is a silently different experiment, which is why the refusal exists.
Commit the JSON -- it is the provenance for every number produced afterwards.
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from .data import cohort

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
SEED = 1234
TEST_FRAC = 0.20


def path(task):
    return os.path.join(DIR, f"holdout_{task}.json")


def make(task, force=False):
    p = path(task)
    if os.path.exists(p) and not force:
        print(f"  {task}: exists, refusing to overwrite (--force to override) -> {p}")
        return json.load(open(p))
    c = cohort(task)
    idx = np.arange(len(c.y))
    dev, test = train_test_split(idx, test_size=TEST_FRAC, stratify=c.y, random_state=SEED)
    obj = {"task": task, "seed": SEED, "test_frac": TEST_FRAC, "n": len(idx),
           "classes": c.classes, "dev": sorted(map(int, dev)), "test": sorted(map(int, test)),
           "row_no": [int(v) for v in c.df["no."]]}
    os.makedirs(DIR, exist_ok=True)
    json.dump(obj, open(p, "w"), indent=1)
    print(f"  {task}: wrote {p}")
    return obj


def load(task):
    """dev, test index arrays into the cohort's row order."""
    o = json.load(open(path(task)))
    c = cohort(task)
    # the split is stored against `no.` order; fail loudly if the cohort definition drifted
    assert o["row_no"] == [int(v) for v in c.df["no."]], (
        "cohort changed since the holdout was frozen (DROP_NO or label mapping edited). "
        "Every result since the freeze is invalid; regenerate deliberately with --force.")
    return np.array(o["dev"]), np.array(o["test"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for task in ("4class", "binary"):
        o = make(task, a.force)
        c = cohort(task)
        dev, test = np.array(o["dev"]), np.array(o["test"])
        assert not set(dev) & set(test) and len(dev) + len(test) == len(c.y)
        pd_, pt = np.bincount(c.y[dev]) / len(dev), np.bincount(c.y[test]) / len(test)
        assert np.abs(pd_ - pt).max() < 0.02, "stratification drifted >2pp"
        print(f"    dev={len(dev)} test={len(test)}  test counts={np.bincount(c.y[test]).tolist()}"
              f"  max class-prop gap={np.abs(pd_ - pt).max():.4f}")


if __name__ == "__main__":
    main()
