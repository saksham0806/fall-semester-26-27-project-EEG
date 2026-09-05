"""Shared loader helpers."""
from collections import defaultdict

from ...contract import CHANNELS


def dedupe(recs):
    """Collapse byte-identical recordings; drop the ones that stay ambiguous.

    ``recs`` is ``[(content_hash, subject, payload)]`` in file order. Returns
    ``(payloads, n_merged, n_dropped)``.

    Three cases, all of them real in this project:

    * Same bytes, one subject — a copied file. Keep one. CogLoad copies every
      `natural-N.txt` into both task folders; Mumtaz ships two copies of H S15's EO.
    * Same bytes across subjects whose recording sets are *identical throughout* — one
      person filed under two ids, so merge them into one id and keep a single copy.
      Mumtaz has two such pairs (H S27/S30, MDD S33/S34), duplicated across both
      sessions. They share a class label, so only the subject number was ever ambiguous.
    * Same bytes across subjects who otherwise hold distinct recordings — one recording
      attributed to two real people. Unrecoverable: assigning it to either files one
      person's data under the other's id, so both copies drop. This is CogLoad's
      `Stroop_Data/highlevel-{14,15}.txt`.

    The merge matters because a subject kept under two ids defeats grouped CV silently —
    the same person lands in train and test and no assertion can see it.
    """
    subjects, by_subject = defaultdict(set), defaultdict(set)
    for h, s, _ in recs:
        subjects[h].add(s)
        by_subject[s].add(h)

    merge, ambiguous = {}, set()
    for h, ss in subjects.items():
        if len(ss) < 2:
            continue
        # identical throughout => one person under several ids; else unrecoverable
        if len({frozenset(by_subject[s]) for s in ss}) == 1:
            merge.update(dict.fromkeys(ss, "+".join(sorted(ss))))
        else:
            ambiguous.add(h)

    out, seen = [], set()
    for h, subject, payload in recs:
        if h in ambiguous or h in seen:
            continue
        seen.add(h)
        if subject in merge:
            payload = _reid(payload, merge[subject])
        out.append(payload)
    dropped = sum(1 for h, _, _ in recs if h in ambiguous)
    return out, len(recs) - len(out) - dropped, dropped


def _reid(payload, subject_id):
    x, meta = payload
    meta = dict(meta, subject_id=subject_id,
                trial_id=meta["trial_id"].replace(meta["subject_id"], subject_id, 1))
    return x, meta


def _bare(name):
    """'EEG Fp1-LE' -> 'fp1'. EDF labels carry montage prefixes and reference suffixes."""
    return name.removeprefix("EEG ").split("-")[0].strip().lower()


def pick_channels(raw):
    """The six frontals from an mne Raw, in canonical order. Asserts, never warns."""
    found = {_bare(n): n for n in raw.ch_names}
    missing = [c for c in CHANNELS if c.lower() not in found]
    assert not missing, f"unresolved channels {missing} in {raw.ch_names}"
    return raw.get_data(picks=[found[c.lower()] for c in CHANNELS])
