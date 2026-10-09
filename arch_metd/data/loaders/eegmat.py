"""EEGMAT loader — domain 4, stress (metd_plan.md §2.5).

Neurocom 23-channel system at 500 Hz, 36 subjects. Two files each:
`_1` is the background before the task (y_str = 0), `_2` is serial subtraction
(y_str = 1). The paired design is what §7.3's sign test rests on.

EEGMAT never appears in a training fold — it is the held-out amplifier for stress.
"""
import re

import mne
import pandas as pd

from ...contract import BY_DOMAIN, EEGMAT_INFO
from . import pick_channels

C = BY_DOMAIN[4]

_FN = re.compile(r"Subject(\d+)_([12])\.edf$")


def load():
    info = pd.read_csv(EEGMAT_INFO).set_index("Subject")
    out = []
    for f in sorted(C.path.glob("*.edf")):
        m = _FN.match(f.name)
        assert m, f"unparsed filename: {f.name}"
        num, part = m[1], int(m[2])
        raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        assert raw.info["sfreq"] == C.fs, f"{f.name}: {raw.info['sfreq']} Hz"
        row = info.loc[f"Subject{num}"]
        out.append((pick_channels(raw), dict(
            subject_id=f"eegmat_{num}", trial_id=f"eegmat_{num}_{part}",
            y_dep=-1, y_str=part - 1,
            domain=C.domain, protocol=C.protocol,
            condition="background" if part == 1 else "arith", fs=C.fs,
            # carried for reporting only — never a training target
            count_quality=int(row["Count quality"]))))
    return out
