"""Phase 1-2 gate — §11 Ingestion, all five corpora.

Every assertion here corresponds to a bug that has already shipped in this repo.

    python -m arch_metd.data.loaders
"""
import hashlib

import numpy as np

from ...contract import BY_DOMAIN, CHANNELS, n_windows
from . import cogload, eegmat, modma, mumtaz, sam40


def _common(recs, domain):
    """Invariants every loader owes the build step, whatever the corpus."""
    c = BY_DOMAIN[domain]
    by_content = {}
    for x, m in recs:
        # after dedupe, identical bytes must never sit under two subject ids — that is a
        # subject in train and test at once, and no split assertion can see it
        h = hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()
        first = by_content.setdefault(h, m["subject_id"])
        assert first == m["subject_id"], f"same recording under {first} and {m['subject_id']}"
        assert x.ndim == 2 and x.shape[0] == len(CHANNELS), (m["trial_id"], x.shape)
        assert np.isfinite(x).all(), f"{m['trial_id']}: non-finite samples"
        assert m["domain"] == domain and m["fs"] == c.fs and m["protocol"] == c.protocol
        assert (m["y_dep"] == -1) if c.task == "stress" else (m["y_str"] == -1)
        assert not (m["y_dep"] != -1 and m["y_str"] != -1), "both labels on one recording"
    return sum(n_windows(x.shape[1], c.fs) for x, _ in recs)


def check_sam40():
    recs = sam40.load()
    n = _common(recs, 1)
    assert len(recs) == 480, len(recs)
    assert all(x.shape[1] == 3200 for x, _ in recs), "SAM40 trials are 3200 samples"
    # 3200 samples at a true 128 Hz -> 11 windows. 5 windows / 2,400 total means the
    # `[:, ::2]` decimation bug is back and every segment is time-stretched 2x.
    assert n_windows(3200, 128) == 11
    assert n == 5280, f"{n} pre-gate windows, expected 5280"
    relax = sum(m["y_str"] == 0 for _, m in recs)
    assert (relax, len(recs) - relax) == (120, 360), (relax, len(recs) - relax)
    print(f"sam40   OK — {len(recs)} trials, {len(set(m['subject_id'] for _, m in recs))} subjects, "
          f"{n} pre-gate windows, {relax} relax / {len(recs) - relax} task")


def check_modma():
    recs = modma.load()
    n = _common(recs, 0)
    subs = {m["subject_id"] for _, m in recs}
    assert len(recs) == len(subs) == 53, (len(recs), len(subs))
    mdd = sum(m["y_dep"] == 1 for _, m in recs)
    assert (mdd, len(recs) - mdd) == (24, 29), (mdd, len(recs) - mdd)
    # a selected row that is identically zero means E129 leaked into the channel set
    assert all(np.abs(x).max(axis=1).min() > 0 for x, _ in recs), "a selected channel is flat"
    print(f"modma   OK — {len(recs)}/53 subjects resolved, {mdd} MDD / {len(recs) - mdd} HC, "
          f"{n} pre-gate windows")


def check_cogload():
    recs = cogload.load()
    n = _common(recs, 2)
    subs = {m["subject_id"] for _, m in recs}
    # the first digit run collapses four levels across two task folders into 15 IDs
    assert len(subs) == 15, sorted(subs)
    base = sum(m["y_str"] == 0 for _, m in recs)
    assert base == 15, f"{base} baselines — one natural recording per subject"
    assert {m["level"] for _, m in recs} == {0, 1, 2, 3}
    print(f"cogload OK — {len(recs)} recordings, {len(subs)} subjects, {n} pre-gate windows, "
          f"{base} natural / {len(recs) - base} load")


def check_mumtaz():
    recs = mumtaz.load()
    n = _common(recs, 3)
    got = mumtaz.sessions(recs)
    mdd = {m["subject_id"] for _, m in recs if m["y_dep"] == 1}
    assert {m["condition"] for _, m in recs} == {"EC", "EO"}, "TASK sessions leaked in"
    # H S1 and MDD S1 are different people; a merge would show up as too few subjects
    assert all(s.startswith(("mumtaz_H", "mumtaz_MDD")) for s in got), sorted(got)[:3]
    partial = {s: sorted(v) for s, v in got.items() if len(v) < 2}
    print(f"mumtaz  OK — {len(recs)} recordings, {len(got)} subjects "
          f"({len(mdd)} MDD / {len(got) - len(mdd)} HC), {n} pre-gate windows")
    print(f"          incomplete sessions ({len(partial)}): "
          f"{', '.join(f'{s}={v[0]}' for s, v in sorted(partial.items())) or 'none'}")


def check_eegmat():
    recs = eegmat.load()
    n = _common(recs, 4)
    subs = {m["subject_id"] for _, m in recs}
    assert len(recs) == 72 and len(subs) == 36, (len(recs), len(subs))
    # §7.3's paired sign test needs exactly one baseline and one task per subject
    pairs = {s: sorted(m["y_str"] for _, m in recs if m["subject_id"] == s) for s in subs}
    assert all(v == [0, 1] for v in pairs.values()), "unpaired subject"
    print(f"eegmat  OK — {len(recs)} recordings, {len(subs)} paired subjects, "
          f"{n} pre-gate windows")


if __name__ == "__main__":
    check_sam40()
    check_modma()
    check_cogload()
    check_mumtaz()
    check_eegmat()
    print("phase 1-2 gate: pass")
