"""Mumtaz 2016 loader — domain 3, depression (metd_plan.md §2.4).

19-channel 10-20 cap at 256 Hz, HUSM Malaysia. The only external depression corpus in
this build, so it carries more weight than the others.

Two things this copy of the data forces:

  * `H S1` and `MDD S1` are DIFFERENT PEOPLE — subject numbering restarts per group.
    Keying on the digit run alone would merge 34 pairs of subjects across the class
    boundary, so `subject_id` carries the group.
  * Spacing in filenames is inconsistent (`MDD S1  EO.edf`), and two byte-identical
    copies of H S15's EO recording ship with numeric prefixes.

TASK sessions are dropped: P300 is an evoked-potential paradigm, not resting state.
"""
import hashlib
import re
from collections import defaultdict

import mne

from ...contract import BY_DOMAIN
from . import dedupe, pick_channels

C = BY_DOMAIN[3]

_FN = re.compile(r"(?:\d+_)?(H|MDD)\s+S(\d+)\s+(EC|EO|TASK)\.edf$", re.I)


def load():
    recs = []
    for f in sorted(C.path.glob("*.edf")):
        m = _FN.match(f.name)
        assert m, f"unparsed filename: {f.name}"
        group, num, session = m[1].upper(), int(m[2]), m[3].upper()
        if session == "TASK":
            continue
        raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        assert raw.info["sfreq"] == C.fs, f"{f.name}: {raw.info['sfreq']} Hz"
        x = pick_channels(raw)
        sid = f"mumtaz_{group}{num}"
        recs.append((hashlib.md5(x.tobytes()).hexdigest(), sid, (x, dict(
            subject_id=sid, trial_id=f"{sid}_{session}",
            y_dep=int(group == "MDD"), y_str=-1,
            domain=C.domain, protocol=C.protocol, condition=session, fs=C.fs))))

    out, merged, dropped = dedupe(recs)
    print(f"mumtaz: {len(recs)} EC/EO files -> {len(out)} recordings "
          f"({merged} duplicate copies merged, {dropped} ambiguous dropped)")
    return out


def sessions(recs):
    """Which sessions each subject actually has. Missing ones are logged, not dropped."""
    got = defaultdict(set)
    for _, m in recs:
        got[m["subject_id"]].add(m["condition"])
    return got
