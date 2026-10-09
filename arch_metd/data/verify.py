"""Phase 3 gate — §11 Preprocessing and Labels, asserted on the BUILT arrays.

build.py checks the §4 contract as it assembles. This checks the things only visible
once the filter chain has actually run on real data.

    python -m arch_metd.data.verify
"""
import numpy as np

from ..contract import BY_DOMAIN, CORPORA, FS, NOTCH_HZ, VAR_GATE_K, WINDOW, n_windows
from .build import load, three_class


def _band_fraction(x, lo, hi):
    f = np.fft.rfftfreq(x.shape[-1], 1 / FS)
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    sel = (f >= lo) & (f <= hi)
    return p[..., sel].sum() / p.sum()


def check_spectrum(d, rng):
    """Mains must be gone in EVERY corpus — a residue in one is a corpus signature.

    §11 states this as "power in 49-51 Hz < 0.001 of total". Measured, that test conflates
    two different things. What it is *for* is mains: MODMA showed a 2.3x peak at 49-51 Hz
    over its stopband neighbours before the order was raised. What it also catches is
    broadband noise sitting in the filter's transition band, which is not mains and which
    no notch can touch — CogLoad's raw signal is 99.9999% DC pedestal, and once that is
    high-passed away the residue is wideband amplifier and EMG noise with no line peak at
    all. So the assertion is a PEAK test against the adjacent stopband, which is the
    question the spec meant to ask; the absolute fraction is reported next to it.

    Both are asserted. CogLoad's residual HF excess is still an 8x corpus signature after
    gating — a finding for the phase 6 probes, not something preprocessing can fix without
    a per-corpus filter, which §3.2 forbids for exactly this reason.
    """
    stops = {}
    for c in CORPORA:
        x = d["X"][d["y_domain"] == c.domain]
        x = x[rng.choice(len(x), min(500, len(x)), replace=False)]
        line = _band_fraction(x, NOTCH_HZ - 1, NOTCH_HZ + 1)
        neighbours = max(_band_fraction(x, 46, 48), _band_fraction(x, 52, 54))
        stop = stops[c.name] = _band_fraction(x, 46, 64)
        peak = line / neighbours
        assert peak < 2.0, f"{c.name}: 50 Hz peak {peak:.2f}x its stopband neighbours"
        # §11's literal test. It holds for every corpus on the GATED data, though not on
        # raw pre-gate windows — the variance gate removes the noisy windows that carry
        # the broadband HF, taking CogLoad from 6.8e-2 above 45 Hz to 4.1e-3. Measure
        # this before the gates and CogLoad fails; that is a property of which windows
        # survive, not of the filter.
        assert line < 1e-3, f"{c.name}: {line:.2e} of power at 49-51 Hz"
        print(f"  {c.name:8s} 49-51 Hz {line:.2e}  peak {peak:.2f}x  >45 Hz {stop:.2e}")
    worst = max(stops.values()) / min(stops.values())
    print(f"  above-45 Hz spread across corpora: {worst:.0f}x "
          f"({max(stops, key=stops.get)} vs {min(stops, key=stops.get)}) — corpus signature")


def check_normalization(d):
    assert np.isfinite(d["X"]).all(), "non-finite values in X"
    m = np.abs(d["X"].mean(axis=(1, 2))).max()
    s = np.abs(d["X"].std(axis=(1, 2)) - 1).max()
    assert m < 1e-6 and s < 1e-6, f"|mean| {m:.2e}, |std-1| {s:.2e}"
    print(f"  per-segment |mean| <= {m:.1e}, |std-1| <= {s:.1e}, "
          f"|X| <= {np.abs(d['X']).max():.2f}")


def check_labels(d):
    dom = d["y_domain"]
    for c in CORPORA:
        sel = dom == c.domain
        masked = "y_dep" if c.task == "stress" else "y_str"
        assert (d[masked][sel] == -1).all(), f"{c.name}: {masked} is not masked"
    assert not ((d["y_dep"] != -1) & (d["y_str"] != -1)).any()
    assert (d["anchor"] == ((d["y_dep"] == 0) | (d["y_str"] == 0))).all()

    # class counts by SUBJECT against the source spreadsheets, not by segment
    for domain, expect in ((0, (24, 29)), (3, (32, 29))):
        sel = dom == domain
        pos = {s for s in d["subject_id"][sel & (d["y_dep"] == 1)]}
        neg = {s for s in d["subject_id"][sel & (d["y_dep"] == 0)]}
        assert (len(pos), len(neg)) == expect, (BY_DOMAIN[domain].name, len(pos), len(neg))
        assert not (pos & neg), "a subject holds both depression classes"
    print(f"  labels OK — modma 24/29, mumtaz 32/29 subjects, "
          f"{int(d['anchor'].sum())} anchors")


def check_sam40_transient():
    """§11: SAM40 amplitude must be flat across trial time.

    SAM40 trials are standalone 25 s files with no continuous context, so without the
    reflect padding their edges carry a filter transient no other corpus has — a corpus
    signature by construction. Reprocessed here rather than read from the arrays,
    because the stored segments are shuffled and carry no within-trial position.
    """
    from .loaders import sam40
    from .preprocess import filter_recording, resample

    recs = [r for r in sam40.load() if r[1]["condition"] == "relax"]
    n = n_windows(3200, 128)
    profile = np.zeros(n)
    for x, m in recs:
        y = resample(filter_recording(x, m["fs"]), m["fs"])
        for i in range(n):
            profile[i] += np.abs(y[:, i * 256:i * 256 + WINDOW]).mean()
    profile /= len(recs)
    edge = max(profile[0], profile[-1]) / np.median(profile[1:-1])
    assert edge < 1.15, f"edge windows {edge:.2f}x the middle — transient survived padding"
    print(f"  sam40 edge/middle amplitude {edge:.3f} over {len(recs)} relax trials")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = load()
    print(f"loaded {len(d['X'])} segments, K={VAR_GATE_K}")
    check_spectrum(d, rng)
    check_normalization(d)
    check_labels(d)
    check_sam40_transient()
    y3 = three_class(d["y_dep"], d["y_str"])
    print(f"  3-class: {(y3 == 0).sum()} healthy / {(y3 == 1).sum()} depression / "
          f"{(y3 == 2).sum()} stress")
    print("phase 3 gate: pass")
