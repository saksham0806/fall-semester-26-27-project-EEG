"""
Load the BRMH psychiatric cohort (datasets/OSF_park/EEG.machinelearing_data_BRMH.csv).

Precomputed FEATURE TABLE, not raw EEG: 114 absolute band powers (6 bands x 19 channels)
+ 1026 coherences (6 bands x C(19,2) pairs) + demographics, one row per recording.

    .venv/bin/python -m OSF_Park_model.data     # self-check
"""
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CSV = os.path.join(ROOT, "datasets/OSF_park/EEG.machinelearing_data_BRMH.csv")

CLASSES_4 = ["healthy", "depression", "anxiety", "stress"]
CLASSES_2 = ["healthy", "depression"]
BANDS = ["delta", "theta", "alpha", "beta", "highbeta", "gamma"]
DEMO_COLS = ["age", "sex", "IQ", "education"]

# Rows whose entire 114-column AB block is byte-identical to another row.  Verified pairs:
#   205 <-> 206   both Healthy control
#   226 <-> 944   Behavioral addiction  <->  Healthy control   <-- contradictory
#   264 <-> 265   both OCD
# 206 goes because 205<->206 is a pair *within* the 4-class subset: identical features in
# train and test is fold leakage.  944 goes because its AB block is byte-identical to a
# different person's, so its band powers are probably not its own -- a row whose spectral
# half is known-corrupt does not belong in a spectral experiment.
# Cost falls on the smallest class: healthy 95 -> 93.
DROP_NO = [206, 944]


def _blocks(df):
    ab = [c for c in df.columns if c.startswith("AB.")]
    coh = [c for c in df.columns if c.startswith("COH.")]
    return ab, coh


def load_brmh():
    """Full 945-row table with the all-NaN trailing column dropped.

    `Unnamed: 122` is a trailing-comma artifact and the ONLY all-NaN column; leave it in
    and a naive dropna() deletes all 945 rows.
    """
    df = pd.read_csv(CSV)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def _label(df):
    """4-class target. Depression is the *specific* diagnosis; the rest are main.disorder."""
    y = df["specific.disorder"].map({"Depressive disorder": "depression"})
    return y.fillna(df["main.disorder"].map({
        "Healthy control": "healthy",
        "Anxiety disorder": "anxiety",                      # panic + social anxiety
        "Trauma and stress related disorder": "stress",     # PTSD + acute + adjustment
    }))


class Cohort:
    """Feature blocks kept separate so ablations are a keyword argument, not a re-load."""

    def __init__(self, df, classes):
        self.classes = classes
        self.df = df.reset_index(drop=True)
        ab, coh = _blocks(self.df)
        self.ab_cols, self.coh_cols = ab, coh
        self.AB = self.df[ab].to_numpy(np.float32)
        self.COH = self.df[coh].to_numpy(np.float32)
        self.y = self.df["_y"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
        self.year = self.df["eeg.date"].str.slice(0, 4).astype(int).to_numpy()

        demo = self.df[DEMO_COLS].copy()
        demo["sex"] = (demo["sex"] == "M").astype(float)
        # the 28 stray NaNs in the file live only here (education/IQ), never in AB/COH
        self.DEMO = demo.fillna(demo.median()).to_numpy(np.float32)

        self.channels = [c.split(".")[4] for c in ab[:19]]
        self._parse_coh()

    def _parse_coh(self):
        """Index map for COH.<L>.<band>.<l1>.<CH1>.<l2>.<CH2> -> (band, i, j)."""
        ci = {c: i for i, c in enumerate(self.channels)}
        bi = {b: i for i, b in enumerate(BANDS)}
        self._coh_idx = []
        for k, c in enumerate(self.coh_cols):
            p = c.split(".")
            self._coh_idx.append((bi[p[2]], ci[p[4]], ci[p[6]], k))

    def channel_tensor(self):
        """(n, 19, 6) band power per channel.

        The AB block is BAND-major (cols 0-18 delta, 19-37 theta, ...), same convention as
        the project's v1 81 features.  Reshaping the other way silently scrambles channels
        into bands, so the self-check below pins it against named columns.
        """
        return self.AB.reshape(len(self.AB), 6, 19).transpose(0, 2, 1)

    def coh_matrix(self):
        """(n, 6, 19, 19) symmetric coherence adjacency, unit diagonal."""
        n = len(self.COH)
        m = np.zeros((n, 6, 19, 19), np.float32)
        m[:, :, np.arange(19), np.arange(19)] = 100.0     # COH is a percentage, 0-100
        for b, i, j, k in self._coh_idx:
            m[:, b, i, j] = m[:, b, j, i] = self.COH[:, k]
        return m

    def X(self, ab=True, coh=True, demo=False):
        parts = [b for b, keep in ((self.AB, ab), (self.COH, coh), (self.DEMO, demo)) if keep]
        return np.concatenate(parts, axis=1)

    def __len__(self):
        return len(self.y)


ILL6 = ["Mood disorder", "Anxiety disorder", "Trauma and stress related disorder",
        "Addictive disorder", "Schizophrenia", "Obsessive compulsive disorder"]


def ill_cohort():
    """All 850 non-healthy subjects labelled by main.disorder (6 classes).

    Plan 03C auxiliary task.  NOTE these 417 extra subjects add NO healthy controls -- there
    are only 95 in the entire file and all are already in the 4-class cohort.  So this data
    cannot relieve a bottleneck caused by too few healthy controls; it can only sharpen the
    model of what ill resting-state EEG looks like.
    """
    df = load_brmh()
    df = df.copy()
    df = df[df["main.disorder"].isin(ILL6) & ~df["no."].isin(DROP_NO)]
    df["_y"] = df["main.disorder"]
    return Cohort(df, ILL6)


def cohort(task="4class"):
    df = load_brmh()
    df = df.copy()
    df["_y"] = _label(df)
    df = df[df["_y"].notna() & ~df["no."].isin(DROP_NO)]
    classes = CLASSES_4 if task == "4class" else CLASSES_2
    return Cohort(df[df["_y"].isin(classes)], classes)


def splits(y, seeds=(0, 1, 2), n_splits=5):
    """StratifiedKFold over several seeds.

    There is no subject-identifier column -- `no.` indexes rows, not people -- so grouping
    is impossible.  This is grouping-by-necessity, not grouping-by-proof: if any person was
    recorded twice across the 2011-2019 span, those rows can straddle a fold.  n=529 also
    makes a single seed noisy, hence 3 x 5 = 15 folds.
    """
    for s in seeds:
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=s).split(y, y):
            yield tr, te



def to_blocks(c, AB=None, COH=None, DEMO=None, scaler=None):
    """Derive every model's view of the data from the three raw blocks.

    Single source of truth so XAI can permute a feature group in AB and have the flat,
    channel-tensor and coherence-matrix views all move together.  `scaler` is a dict of
    (mean, std) fitted on the training fold; pass None to fit one here.
    """
    AB = c.AB if AB is None else AB
    COH = c.COH if COH is None else COH
    DEMO = c.DEMO if DEMO is None else DEMO
    flat = np.concatenate([AB, COH], 1)
    if scaler is None:
        scaler = {}
        for k, v in (("flat", flat), ("demo", DEMO)):
            scaler[k] = (v.mean(0), v.std(0) + 1e-8)
    z = lambda k, v: (v - scaler[k][0]) / scaler[k][1]
    zf = z("flat", flat)
    chan = zf[:, :114].reshape(len(AB), 6, 19).transpose(0, 2, 1)

    n = len(COH)
    m = np.zeros((n, 6, 19, 19), np.float32)
    m[:, :, np.arange(19), np.arange(19)] = 1.0
    for b, i, j, k in c._coh_idx:
        m[:, b, i, j] = m[:, b, j, i] = COH[:, k] / 100.0      # COH is a percentage

    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    return {"flat": t(zf), "chan": t(chan), "coh": t(m),
            "cohflat": t(zf[:, 114:]), "demo": t(z("demo", DEMO)),
            "np_flat": flat, "np_demo": DEMO,
            "np_both": np.concatenate([flat, DEMO], 1)}, scaler


def fit_scaler(c, idx):
    """Scaler from TRAINING rows only -- fitting on all rows leaks test statistics."""
    flat = np.concatenate([c.AB, c.COH], 1)[idx]
    d = c.DEMO[idx]
    return {"flat": (flat.mean(0), flat.std(0) + 1e-8), "demo": (d.mean(0), d.std(0) + 1e-8)}


AB_CHANNEL_COLS = lambda ci: [b * 19 + ci for b in range(6)]     # band-major: 6 cols per channel
AB_BAND_COLS = lambda bi: list(range(bi * 19, (bi + 1) * 19))


def _selfcheck():
    c4, c2 = cohort("4class"), cohort("binary")
    assert len(c4) == 527, len(c4)          # 529 in the subset, minus rows 206 and 944
    assert len(c2) == 292, len(c2)
    assert c4.AB.shape[1] == 114 and c4.COH.shape[1] == 1026
    for name, blk in (("AB", c4.AB), ("COH", c4.COH), ("DEMO", c4.DEMO)):
        assert np.isfinite(blk).all(), f"{name} has NaN/inf"
    assert c4.channels[:3] == ["FP1", "FP2", "F7"] and c4.channels[-1] == "O2"

    # band-major round-trip: pin the tensor against explicitly named columns
    t = c4.channel_tensor()
    assert t.shape == (527, 19, 6)
    for band, ch in (("delta", "FP1"), ("gamma", "O2"), ("alpha", "Cz")):
        col = [x for x in c4.ab_cols if x.split(".")[2] == band and x.split(".")[4] == ch]
        assert len(col) == 1, col
        want = c4.df[col[0]].to_numpy(np.float32)
        got = t[:, c4.channels.index(ch), BANDS.index(band)]
        assert np.allclose(got, want), f"{band}/{ch} scrambled -- layout is band-major"

    m = c4.coh_matrix()
    assert m.shape == (527, 6, 19, 19)
    assert np.allclose(m, m.transpose(0, 1, 3, 2)), "coherence not symmetric"
    assert np.allclose(m[:, 0, 0, 1], c4.df["COH.A.delta.a.FP1.b.FP2"].to_numpy(np.float32))
    assert np.allclose(m[:, 5, 17, 18], c4.df["COH.F.gamma.r.O1.s.O2"].to_numpy(np.float32))

    folds = list(splits(c4.y))
    assert len(folds) == 15
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert len(set(c4.y[te])) == 4
    assert sorted(np.concatenate([te for _, te in folds[:5]])).__len__() == 527

    counts = dict(zip(*np.unique(c4.df["_y"], return_counts=True)))
    print(f"4class n={len(c4)} {counts}")
    print(f"binary n={len(c2)}  AB={c4.AB.shape} COH={c4.COH.shape} DEMO={c4.DEMO.shape}")
    print(f"years {c4.year.min()}-{c4.year.max()}  majority={max(counts.values())/len(c4):.4f}")
    b, sc = to_blocks(c4)
    assert b["chan"].shape == (527, 19, 6) and b["coh"].shape == (527, 6, 19, 19)
    assert b["flat"].shape == (527, 1140) and b["cohflat"].shape == (527, 1026)
    assert torch.allclose(b["coh"], b["coh"].transpose(2, 3))
    assert abs(b["flat"].mean().item()) < 1e-4 and abs(b["flat"].std().item() - 1) < 1e-2
    # a channel's AB columns must be exactly its slice of the channel tensor
    ci = c4.channels.index("Cz")
    assert torch.allclose(b["flat"][:, AB_CHANNEL_COLS(ci)], b["chan"][:, ci, :])
    _, sc2 = to_blocks(c4, scaler=sc)
    assert sc2 is sc
    print("all checks passed")


if __name__ == "__main__":
    _selfcheck()
