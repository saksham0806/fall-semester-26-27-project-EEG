"""MODMA resting-state loader — domain 0, depression (metd_plan.md §2.1).

Returns continuous (6, n) arrays at the native 250 Hz. No filtering, no windowing.
"""
import re

import numpy as np
import pandas as pd
import scipy.io

from ...contract import BY_DOMAIN, MODMA_LABELS

C = BY_DOMAIN[0]


def _eeg_key(mat):
    """The EEG variable is named after the file and differs in every one of the 53."""
    keys = [k for k, v in mat.items()
            if not k.startswith("__") and np.ndim(v) == 2
            and v.shape[0] in (128, 129) and v.shape[1] > 1000]
    # shape[1] > 1000 is what rejects Impedances_0, which is also (129, 1)
    assert len(keys) == 1, f"expected one EEG variable, got {keys}"
    return keys[0]


def subject_id(filename):
    """Filenames are inconsistent; the leading digit run is the only 53/53 match."""
    m = re.match(r"0*(\d+)", filename)
    assert m, f"no subject id in {filename}"
    return int(m.group(1))


def load():
    info = pd.read_excel(MODMA_LABELS).set_index("subject id")
    out = []
    for f in sorted(C.path.glob("*.mat")):
        sid = subject_id(f.name)
        assert sid in info.index, f"{f.name} -> {sid} is not in the spreadsheet"
        mat = scipy.io.loadmat(f)
        a = mat[_eeg_key(mat)]
        if a.shape[0] == 129:
            # E129 is the Cz reference and is identically zero — never select it
            assert np.abs(a[128]).max() == 0, f"{f.name}: E129 is not zero"
        assert max(C.channels) < a.shape[0], f"{f.name}: only {a.shape[0]} rows"
        row = info.loc[sid]
        out.append((np.ascontiguousarray(a[list(C.channels)], dtype=np.float64), dict(
            subject_id=f"modma_{sid}", trial_id=f"modma_{sid}",
            y_dep=int(row["type"] == "MDD"), y_str=-1,
            domain=C.domain, protocol=C.protocol, condition="EC",
            fs=C.fs, phq9=float(row["PHQ-9"]), gad7=float(row["GAD-7"]))))
    return out
