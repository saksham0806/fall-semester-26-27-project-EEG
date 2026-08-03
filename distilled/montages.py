"""
Electrode positions for all four corpora, in one place, with provenance.

Channel-agnostic models (braindecode's ChannelMerger, ChannelInterpolationLayer, LUNA) need 3-D
electrode coordinates. Two corpora ship real ones; two do not and are RECONSTRUCTED FROM
ASSUMPTIONS. That distinction is carried in the data structure rather than a comment, because
any result computed on an assumed montage is only as good as the guess.

    corpus   n    positions   source
    MODMA    128  real        GSN-HydroCel-128 standard montage (matches the EGI net used);
                              datasets/MODMA/.../chan_info_egi_128.mat has the same coords
    SAM40    32   real        datasets/SAM40/Coordinates.locs, real 10-20 labels
    AMIGOS   14   ASSUMED     Emotiv EPOC fixed layout — well documented for that headset,
                              but the channel ORDER is assumed from the loader, not read
    CogLoad  8    ASSUMED     no placement documented anywhere on disk; the metadata calls
                              them "EEG Channel 1..8". Highest-risk assumption in the project.

`assumed=True` means: if this guess is wrong, spatial interpolation for that corpus is
meaningless and its results must be discounted. Report those corpora separately.
"""
from dataclasses import dataclass, field

import numpy as np

# Canonical 6-channel frontal montage the student deploys on, and every corpus can supply.
STUDENT_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "F4", "F8"]


@dataclass
class Montage:
    corpus: str
    names: list                      # channel names, in the order the loader emits them
    assumed: bool                    # True => positions are a guess, discount accordingly
    source: str                      # provenance, quoted in the README and reports
    student_idx: list = field(default_factory=list)   # indices of STUDENT_CHANNELS within names

    @property
    def n_chans(self):
        return len(self.names)

    def positions(self):
        """(n_chans, 3) array of 3-D coordinates in metres, via MNE's standard montages."""
        import mne
        mont = mne.channels.make_standard_montage(self._mne_montage())
        pos = mont.get_positions()["ch_pos"]
        missing = [n for n in self.names if n not in pos]
        if missing:
            raise KeyError(f"{self.corpus}: no position for {missing[:5]} "
                           f"(montage {self._mne_montage()})")
        return np.array([pos[n] for n in self.names], dtype=np.float64)

    def chs_info(self):
        """braindecode/MNE `chs_info` dicts — what ChannelInterpolationLayer consumes."""
        pos = self.positions()
        return [{"ch_name": n, "loc": np.concatenate([p, np.zeros(9)]), "kind": "eeg"}
                for n, p in zip(self.names, pos)]

    def _mne_montage(self):
        return "GSN-HydroCel-128" if self.corpus == "MODMA" else "standard_1020"


# --- MODMA: 128-channel EGI HydroCel net -------------------------------------------------
# The .mat arrays have 129 rows: E1..E128 then the Cz reference. Row i == E(i+1).
# Frontal six used by combine_datasets.py:93 are rows [21, 8, 32, 23, 123, 121]
# = E22, E9, E33, E24, E124, E122, which the EGI documentation maps to Fp1/Fp2/F7/F3/F4/F8.
_MODMA_NAMES = [f"E{i}" for i in range(1, 129)]
_MODMA_STUDENT_ROWS = [21, 8, 32, 23, 123, 121]

MODMA = Montage(
    corpus="MODMA",
    names=_MODMA_NAMES,
    assumed=False,
    source="mne GSN-HydroCel-128; cross-checkable against "
           "datasets/MODMA/EEG_128channels_ERP_lanzhou_2015/chan_info_egi_128.mat",
    student_idx=_MODMA_STUDENT_ROWS,
)

# --- SAM40: 32-channel Emotiv Epoc Flex, real 10-20 labels -------------------------------
# Order is the row order of Clean_data, taken from datasets/SAM40/Coordinates.locs.
_SAM40_NAMES = ["Cz", "Fz", "Fp1", "F7", "F3", "FC1", "C3", "FC5", "FT9", "T7", "CP5",
                "CP1", "P3", "P7", "PO9", "O1", "Pz", "Oz", "O2", "PO10", "P8", "P4",
                "CP2", "CP6", "T8", "FT10", "FC6", "C4", "FC2", "F4", "F8", "Fp2"]

SAM40 = Montage(
    corpus="SAM40",
    names=_SAM40_NAMES,
    assumed=False,
    source="datasets/SAM40/Coordinates.locs (EEGLAB polar, real 10-20 names)",
    student_idx=[_SAM40_NAMES.index(c) for c in STUDENT_CHANNELS],
)

# --- AMIGOS: Emotiv EPOC, 14 EEG channels then ECG/GSR ------------------------------------
# The headset's channel set is fixed and documented; what is ASSUMED here is that
# joined_data columns 0..13 follow this order. combine_datasets.py:231 relies on the same
# assumption (it takes [0, 13, 1, 2, 11, 12] as Fp1/Fp2/F7/F3/F4/F8).
# AF3/AF4 stand in for Fp1/Fp2 — the closest EPOC electrodes, an approximation the repo
# already makes (see CLAUDE.md, "only approximation in the pipeline").
_AMIGOS_NAMES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                 "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

AMIGOS = Montage(
    corpus="AMIGOS",
    names=_AMIGOS_NAMES,
    assumed=True,
    source="ASSUMED: Emotiv EPOC standard layout; channel ORDER inferred from "
           "combine_datasets.py:231, not read from any file on disk",
    student_idx=[0, 13, 1, 2, 11, 12],     # AF3, AF4, F7, F3, F4, F8
)

# --- CogLoad: 8-channel OpenBCI Cyton, NO documented placement -----------------------------
# HIGHEST-RISK ASSUMPTION IN THE PROJECT. datasets/metadatas/Cognitive_Load/ calls these
# "EEG Channel 1..8" and nothing on disk says where they sat. This layout is reverse-engineered
# from combine_datasets.py:193, which takes raw columns [1,2,3,4,6,7] as Fp1/Fp2/F7/F3/F4/F8 —
# implying EEG channels 1,2,3,4,6,7 are those, and that the two skipped (5, 8) are Fz and C2
# (per CLAUDE.md). If that is wrong, spatial interpolation for CogLoad is meaningless.
_COGLOAD_NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "Cz"]

COGLOAD = Montage(
    corpus="CogLoad",
    names=_COGLOAD_NAMES,
    assumed=True,
    source="ASSUMED, UNVERIFIABLE: reverse-engineered from combine_datasets.py:193. "
           "No electrode placement is documented for this corpus anywhere on disk.",
    student_idx=[0, 1, 2, 3, 5, 6],
)

ALL = {m.corpus: m for m in (MODMA, SAM40, AMIGOS, COGLOAD)}
CORPUS_ID = {"MODMA": 0, "SAM40": 1, "CogLoad": 2, "AMIGOS": 3}


def assumed_corpora():
    """Corpora whose positions are guessed — their results carry a permanent caveat."""
    return [m.corpus for m in ALL.values() if m.assumed]


def selfcheck():
    """Positions resolve, are unique, and the frontal six really are frontal."""
    for name, m in ALL.items():
        pos = m.positions()
        assert pos.shape == (m.n_chans, 3), f"{name}: {pos.shape}"
        assert np.isfinite(pos).all(), f"{name}: non-finite positions"
        # unique electrodes
        assert len({tuple(np.round(p, 6)) for p in pos}) == m.n_chans, \
            f"{name}: duplicate electrode positions"
        assert len(m.student_idx) == len(STUDENT_CHANNELS), f"{name}: student_idx wrong length"

        # The student channels must sit in the ANTERIOR half (+y in MNE head coords).
        # This catches an index-mapping error, which is otherwise silent and ruinous.
        sy = pos[m.student_idx][:, 1]
        assert (sy > 0).all(), (f"{name}: student channels not all anterior, y={sy.round(3)} "
                                f"-> student_idx is probably mis-mapped")
        flag = "ASSUMED" if m.assumed else "real"
        print(f"  {name:8s} {m.n_chans:3d} ch  positions {flag:8s}  "
              f"student y-range [{sy.min():+.3f}, {sy.max():+.3f}]")
    print(f"\n  assumed montages (discount their results): {assumed_corpora()}")


if __name__ == "__main__":
    print("montage selfcheck\n")
    selfcheck()
