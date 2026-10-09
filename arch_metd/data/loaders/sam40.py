"""SAM40 loader — domain 1, stress (metd_plan.md §2.2).

Reads raw_data/ (key `Data`), never filtered_data/: the authors' cleaning is an
uncontrolled per-corpus filter chain, which would make preprocessing itself a corpus
signature. Run filtered_data/ as a separate ablation if wanted.

fs is 128 Hz. There is NO decimation here. The old pipeline did `data[:, ::2]` with a
"native is 256 Hz" comment and time-stretched every SAM40 segment 2x.
"""
import re

import numpy as np
import scipy.io

from ...contract import BY_DOMAIN

C = BY_DOMAIN[1]

_FN = re.compile(r"(?P<task>[A-Za-z_]+)_sub_(?P<sub>\d+)_trial(?P<trial>\d+)\.mat$")
_CONDITION = {"Relax": "relax", "Arithmetic": "arith",
              "Mirror_image": "mirror", "Stroop": "stroop"}


def load():
    out = []
    for f in sorted(C.path.glob("*.mat")):
        m = _FN.match(f.name)
        assert m, f"unparsed filename: {f.name}"
        task, sub, trial = m["task"], int(m["sub"]), int(m["trial"])
        assert task in _CONDITION, f"unknown task {task}"
        d = scipy.io.loadmat(f)["Data"]
        assert d.shape == (32, 3200), f"{f.name}: {d.shape} — 3200 = 128 Hz x 25 s"
        out.append((np.ascontiguousarray(d[list(C.channels)], dtype=np.float64), dict(
            subject_id=f"sam40_{sub}",
            trial_id=f"sam40_{task}_{sub}_{trial}",
            y_dep=-1, y_str=0 if task == "Relax" else 1,
            domain=C.domain, protocol=C.protocol, condition=_CONDITION[task],
            fs=C.fs)))
    return out
