"""Array contract and corpus registry for the METD build (metd_plan.md §1.3, §4).

The single place a dataset path, sampling rate or channel index is written down.
Nothing here imports the legacy ``src`` package — that would drag in TensorFlow.

Self-check: ``python -m arch_metd.contract``
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# §1.3 canonical representation
CHANNELS = ("Fp1", "Fp2", "F7", "F3", "F4", "F8")
FS = 128
WINDOW = 512   # 4.0 s
STRIDE = 256   # 50% overlap
N_DOMAINS = 5  # domain 5 is reserved for TDBRAIN (§2.6) — never emit it

# §3 preprocessing. Identical for every corpus: a gate applied to one corpus only makes
# the rejection rate itself a corpus signature.
PAD_SEC = 3.0        # reflect padding, so SAM40's standalone 25 s trials carry no
                     # filter transient that the continuous corpora lack
NOTCH_HZ = 50.0      # all five corpora are 50 Hz mains regions
NOTCH_Q = 30.0
BAND = (1.0, 45.0)   # 1 Hz, not 0.5: a third the settling transient, same DC removal
BAND_ORDER = 8       # §3.1 says 4; measured, 4 cannot meet §11's mains test. A 45 Hz
                     # corner is only 11% below 50 Hz, so even order 8 passes ~18% of
                     # the power there — the notch does the rejecting, the order decides
                     # how much broadband transition-band noise rides along. Order 4
                     # leaves MODMA at 1.24e-3 of total in 49-51 Hz, order 8 at 3.21e-4.
                     # Independently measured at order 8 in CLAUDE.md's filter note.
FLATLINE_MS = 50.0   # minimum zero-run marked as dropout
CLIP_MAD = 5.0
VAR_GATE_K = 10.0    # calibrated on MODMA (zero dropout, zero NaN by measurement):
                     # rejects 1.41% of its 7,959 windows, inside the 1-2% target.
                     # Its ratio distribution is p50=1.59 p90=3.78 p95=5.83 p99=10.99,
                     # so K=10 sits at 6.3x the median — clear of the 3-5x Fp1/F3 spread
                     # that normal frontal EEG shows because blinks load on the prefrontal
                     # pair. The old K=5 rejects 6.76% here. Re-run --calibrate-k if the
                     # filter chain changes; this number is a property of both together.


@dataclass(frozen=True)
class Corpus:
    name: str
    domain: int
    path: Path
    fs: int            # native rate, before resampling to FS
    protocol: str      # "resting" | "task" — an inference-time input, never a target
    task: str          # "depression" | "stress" — which head this corpus supervises
    channels: tuple    # row indices into the native array, or channel names for EDF


CORPORA = (
    Corpus("modma", 0, ROOT / "datasets/MODMA/EEG_128channels_resting_lanzhou_2015",
           250, "resting", "depression", (21, 8, 32, 23, 123, 121)),   # E22 E9 E33 E24 E124 E122
    Corpus("sam40", 1, ROOT / "datasets/SAM40/raw_data",
           128, "task", "stress", (2, 31, 3, 4, 29, 30)),              # native 128 Hz — do NOT decimate
    Corpus("cogload", 2, ROOT / "datasets/Cognitive Load Assessment Through EEG "
                                "A Dataset from Arithmetic and Stroop Tasks/raw_data",
           250, "task", "stress", (1, 2, 3, 4, 6, 7)),                 # CSV cols, skipping Fz and C2
    Corpus("mumtaz", 3, ROOT / "datasets/Mumtaz",
           256, "resting", "depression", CHANNELS),                    # EDF, matched by name
    Corpus("eegmat", 4, ROOT / "datasets/eeg-during-mental-arithmetic-tasks-1.0.0",
           500, "task", "stress", CHANNELS),                           # EDF, matched by name
)

MODMA_LABELS = CORPORA[0].path / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"
EEGMAT_INFO = CORPORA[4].path / "subject-info.csv"

BY_DOMAIN = {c.domain: c for c in CORPORA}

assert set(BY_DOMAIN) == set(range(N_DOMAINS)), "domain ids must be 0..4; 5 stays reserved"
assert {c.domain for c in CORPORA if c.protocol == "resting"} == {0, 3}
assert {c.domain for c in CORPORA if c.protocol == "task"} == {1, 2, 4}

FIELDS = ("X", "y_dep", "y_str", "y_domain", "anchor",
          "subject_id", "trial_id", "protocol", "condition")


def n_windows(n_samples, fs=None):
    """Windows the §1.3 scheme yields from a recording, after resampling to FS."""
    n = n_samples if fs in (None, FS) else int(round(n_samples * FS / fs))
    return 0 if n < WINDOW else 1 + (n - WINDOW) // STRIDE


def check_paths():
    """Every corpus directory and label file is where the registry says it is."""
    missing = [c.name for c in CORPORA if not c.path.is_dir()]
    assert not missing, f"missing corpora: {missing}"
    for p in (MODMA_LABELS, EEGMAT_INFO):
        assert p.exists(), f"missing label file: {p}"


def validate(d):
    """Assert the §4 contract on a built dataset. Returns the segment count."""
    missing = [k for k in FIELDS if k not in d]
    assert not missing, f"missing fields: {missing}"
    X, dep, stress, dom, anchor = (d["X"], d["y_dep"], d["y_str"],
                                   d["y_domain"], d["anchor"])
    n = len(X)

    assert X.dtype == np.float32, X.dtype
    assert X.shape == (n, len(CHANNELS), WINDOW), X.shape
    assert np.isfinite(X).all(), "non-finite values in X"
    for k in ("y_dep", "y_str", "y_domain"):
        assert d[k].dtype == np.int8, (k, d[k].dtype)
    assert anchor.dtype == np.bool_, anchor.dtype
    for k in ("subject_id", "trial_id", "protocol", "condition"):
        # object or fixed-width unicode — the latter is what round-trips through .npz
        # without allow_pickle, which is a footgun worth not handing to later phases
        assert d[k].dtype.kind in "OU", (k, d[k].dtype)
    # row i of every array is segment i — a stage that drops rows must mask all of them
    assert all(len(d[k]) == n for k in FIELDS), "row-count mismatch across fields"

    assert set(np.unique(dep)) <= {-1, 0, 1}, np.unique(dep)
    assert set(np.unique(stress)) <= {-1, 0, 1}, np.unique(stress)
    assert set(np.unique(dom)) <= set(BY_DOMAIN), "domain 5 is reserved for TDBRAIN"
    assert not ((dep != -1) & (stress != -1)).any(), "a segment carries both labels"
    assert (anchor == ((dep == 0) | (stress == 0))).all(), "anchor disagrees with labels"

    # masking follows the corpus, not the segment: a stress corpus has no depression label
    for c in CORPORA:
        m = dom == c.domain
        if not m.any():
            continue
        other = dep if c.task == "stress" else stress
        assert (other[m] == -1).all(), f"{c.name} carries a {c.task}-opposite label"
        assert (d["protocol"][m] == c.protocol).all(), f"{c.name}: wrong protocol"
    return n


def _synthetic(n_per=2):
    """A minimal contract-conforming dataset spanning all five domains."""
    rng = np.random.default_rng(0)
    rows = []
    for c in CORPORA:
        for i in range(n_per):
            y = i % 2  # alternate so every corpus contributes an anchor
            rows.append((rng.standard_normal((len(CHANNELS), WINDOW)),
                         y if c.task == "depression" else -1,
                         y if c.task == "stress" else -1,
                         c.domain, f"{c.name}_{i}", f"{c.name}_{i}_t0",
                         c.protocol, "EC" if c.protocol == "resting" else "task"))
    cols = list(zip(*rows))
    d = dict(zip(("X", "y_dep", "y_str", "y_domain",
                  "subject_id", "trial_id", "protocol", "condition"),
                 (np.asarray(cols[0], np.float32),
                  *(np.asarray(c, np.int8) for c in cols[1:4]),
                  *(np.asarray(c, object) for c in cols[4:]))))
    d["anchor"] = (d["y_dep"] == 0) | (d["y_str"] == 0)
    return d


if __name__ == "__main__":
    check_paths()
    print(f"paths OK — {len(CORPORA)} corpora, {MODMA_LABELS.name}, {EEGMAT_INFO.name}")

    d = _synthetic()
    print(f"contract OK — {validate(d)} synthetic segments, "
          f"domains {sorted(set(d['y_domain'].tolist()))}, {int(d['anchor'].sum())} anchors")

    # the contract has to actually reject the bugs it exists to catch
    bugs = {
        "both labels on one segment":
            lambda x: x.update(y_str=np.where(x["y_domain"] == 0, 1, x["y_str"]).astype(np.int8)),
        "anchor out of sync":
            lambda x: x.update(anchor=~x["anchor"]),
        "reserved domain 5":
            lambda x: x.update(y_domain=np.full_like(x["y_domain"], 5)),
        "dropped row in one field":
            lambda x: x.update(subject_id=x["subject_id"][:-1]),
        "non-finite X":
            lambda x: x["X"].__setitem__((0, 0, 0), np.nan),
    }
    for label, break_it in bugs.items():
        bad = _synthetic()
        break_it(bad)
        try:
            validate(bad)
        except AssertionError:
            continue
        raise SystemExit(f"FAIL — contract accepted: {label}")
    print(f"rejects OK — {len(bugs)} known bugs caught")
