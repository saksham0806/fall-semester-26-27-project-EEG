"""Cognitive Load loader — domain 2, stress (metd_plan.md §2.3).

Raw ADC counts at the 1e4 scale with a large DC pedestal; the 1 Hz high-pass in §3
removes it. Folders are globbed, never named: the arithmetic folder name ends in a space.

Deduplication (measured on this copy, and it is not what §2.3 describes):

  * `natural-N.txt` is BYTE-IDENTICAL between the two task folders for all 15 subjects.
    The baseline was recorded once and copied into both. Keeping both would double the
    negative class and put identical windows on both sides of a fold.
  * `Stroop_Data/highlevel-14.txt` and `highlevel-15.txt` are byte-identical across two
    DIFFERENT subjects. That one is unrecoverable — both copies are dropped.

The rule is content-based, not filename-based: same bytes under one subject collapse to
one recording, same bytes under two subjects drop both.
"""
import hashlib
import re

import numpy as np
import pandas as pd

from ...contract import BY_DOMAIN
from . import dedupe

C = BY_DOMAIN[2]

_LEVELS = {"natural": 0, "lowlevel": 1, "midlevel": 2, "highlevel": 3}


def _parse(name):
    level, _, rest = name.partition("-")
    assert level in _LEVELS, f"unknown level in {name}"
    m = re.search(r"(\d+)", rest)  # first digit run -> 15 subjects, not 120
    assert m, f"no subject id in {name}"
    return level, int(m.group(1))


def load():
    recs = []
    for f in sorted(C.path.glob("*/*.txt")):
        level, sub = _parse(f.name)
        x = pd.read_csv(f, header=None, usecols=list(C.channels)
                        ).to_numpy(np.float64).T  # cols 1,2,3,4,6,7 skip Fz and C2
        assert x.shape[0] == len(C.channels), x.shape
        task = "natural" if level == "natural" else (
            "arith" if "Arithmetic" in f.parent.name else "stroop")
        recs.append((hashlib.md5(x.tobytes()).hexdigest(), sub, (x, dict(
            subject_id=f"cogload_{sub}",
            trial_id=f"cogload_{sub}_{f.parent.name.strip()}_{level}",
            y_dep=-1, y_str=0 if level == "natural" else 1,
            domain=C.domain, protocol=C.protocol, condition=task,
            fs=C.fs, level=_LEVELS[level]))))

    out, merged, dropped = dedupe(recs)
    print(f"cogload: {len(recs)} files -> {len(out)} recordings "
          f"({merged} duplicate copies merged, {dropped} cross-subject duplicates dropped)")
    return out
