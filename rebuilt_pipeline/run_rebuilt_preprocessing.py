"""Raw-only preprocessing and validation for the plan-2 rebuilt EEG pipeline.

Writes exclusively below rebuilt_pipeline/artifacts.  It deliberately does not
read legacy NumPy datasets or legacy results.
"""
from __future__ import annotations

import csv
import argparse
import io
import json
import platform
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
DEFAULT_OUT = Path(__file__).resolve().parent / "artifacts_v2"
FS_OUT, WINDOW, STRIDE = 128, 512, 256
# ceil(0.050 * 128): a flatline must occupy at least 7 output samples.
FLATLINE_MIN_SAMPLES_OUT = 7
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "F4", "F8"]
MODMA_IDX = [21, 8, 32, 23, 123, 121]  # E22, E9, E33, E24, E124, E122
SAM40_IDX = [2, 31, 3, 4, 29, 30]  # documented in the existing SAM40 loader
COG_IDX = [1, 2, 3, 4, 6, 7]  # EEG columns 1,2,3,4,6,7; col. 0 is sample index


def bandpass(x: np.ndarray, fs: int) -> np.ndarray:
    b, a = butter(4, [0.5, 45.0], btype="bandpass", fs=fs)
    return filtfilt(b, a, x, axis=1)


def flatline_mask_raw(x: np.ndarray, fs: int, out_len: int) -> np.ndarray:
    """Map any exact-zero raw-channel run >=50 ms to the output timeline."""
    threshold = int(np.ceil(0.050 * fs))
    raw_mask = np.zeros(x.shape[1], dtype=bool)
    for ch in x:
        is_zero = ch == 0
        starts = np.flatnonzero(np.diff(np.r_[False, is_zero, False].astype(np.int8)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, is_zero, False].astype(np.int8)) == -1)
        for start, end in zip(starts, ends):
            if end - start >= threshold:
                raw_mask[start:end] = True
    # Nearest-neighbour interval mapping preserves a raw marked span through resampling.
    idx = np.minimum((np.arange(out_len) * fs / FS_OUT).astype(int), len(raw_mask) - 1)
    return raw_mask[idx]


def windows(x_raw: np.ndarray, fs: int, padded: bool = False):
    flat_raw = None
    if padded:
        pad = 384
        x_pad = np.pad(x_raw, ((0, 0), (pad, pad)), mode="reflect")
        x_filtered = bandpass(x_pad, fs)[:, pad:-pad]
    else:
        x_filtered = bandpass(x_raw, fs)
    if fs == FS_OUT:
        x = x_filtered
    else:
        x = resample_poly(x_filtered, up=64, down=125, axis=1)
    flat = flatline_mask_raw(x_raw, fs, x.shape[1])
    for start in range(0, x.shape[1] - WINDOW + 1, STRIDE):
        yield x[:, start:start + WINDOW], bool(flat[start:start + WINDOW].any())


def robust_normalize(segment: np.ndarray) -> np.ndarray:
    median = np.median(segment)
    mad = np.median(np.abs(segment - median))
    # 1.4826 makes the MAD estimate Gaussian-consistent; epsilon only handles a true zero MAD.
    return ((segment - median) / max(1.4826 * mad, 1e-8)).astype(np.float32)


def modma_records():
    archives = sorted((DATA / "MODMA").glob("*.zip"))
    metadata = None
    recordings = []
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            if metadata is None:
                names = [n for n in z.namelist() if n.endswith(".xlsx") and "128channels_resting" in n]
                if names:
                    metadata = pd.read_excel(io.BytesIO(z.read(names[0])))
            for name in z.namelist():
                if "128channels_resting" in name and name.lower().endswith(".mat"):
                    recordings.append((archive, name))
    if metadata is None:
        raise RuntimeError("MODMA 128-channel resting metadata workbook was not found in the readable ZIP files.")
    labels = {str(int(row["subject id"])): int(str(row["type"]).strip() == "MDD")
              for _, row in metadata.iterrows() if pd.notna(row["subject id"])}
    for archive, name in recordings:
        match = re.search(r"(020\d{5})", Path(name).name)
        if not match:
            continue
        native_id = str(int(match.group(1)))
        if native_id not in labels:
            continue
        with zipfile.ZipFile(archive) as z:
            mat = loadmat(io.BytesIO(z.read(name)))
        arrays = [v for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2 and 128 <= v.shape[0] <= 129]
        if not arrays:
            raise RuntimeError(f"No 128-channel signal found in MODMA recording {name}")
        yield "MODMA", f"modma_{native_id}", f"modma_{native_id}", labels[native_id], -1, arrays[0][MODMA_IDX], 250, False


def sam40_records():
    for path in sorted((DATA / "SAM40" / "raw_data").glob("*.mat")):
        match = re.match(r"(.+)_sub_(\d+)_trial(\d+)\.mat$", path.name, re.I)
        if not match:
            continue
        task, subject, trial = match.groups()
        task_l = task.lower()
        stress = 0 if task_l == "relax" else 1 if task_l in {"arithmetic", "stroop", "mirror", "mirror_image"} else None
        if stress is None:
            continue
        mat = loadmat(path)
        if "Data" not in mat:
            raise RuntimeError(f"SAM40 raw file lacks Data variable: {path}")
        yield "SAM40", f"sam40_{subject}", f"sam40_{subject}_{task_l}_trial{trial}", -1, stress, mat["Data"][SAM40_IDX], 128, True


def cogload_records():
    base = DATA / "COGLoad" / "raw_data-20260817T114751Z-1-001" / "raw_data"
    for path in sorted(base.rglob("*.txt")):
        task = path.parent.name.lower()
        # The condition is encoded in the filename, not the parent folder.  The
        # supplied Stroop directory contains natural-*.txt baselines.
        condition = path.stem.split("-", 1)[0].lower()
        stress = 0 if condition == "natural" else 1 if task in {"arithmetic_data", "stroop_data"} else None
        if stress is None:
            continue
        subject_match = re.search(r"(\d+)", path.stem)
        subject = subject_match.group(1) if subject_match else path.stem
        frame = pd.read_csv(path, header=None, skipinitialspace=True)
        if frame.shape[1] <= max(COG_IDX):
            raise RuntimeError(f"Cognitive Load file has insufficient columns: {path}")
        signal = frame.iloc[:, COG_IDX].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64).T
        if not np.isfinite(signal).all():
            raise RuntimeError(f"Non-numeric EEG values in Cognitive Load file: {path}")
        yield "Cognitive Load", f"cogload_{subject}", f"cogload_{task}_{path.stem}", -1, stress, signal, 250, False


def process_pass(write_segments: bool, output_dir: Path | None = None, expected_segments: int | None = None):
    """Run either an audit pass or a disk-backed write pass over raw recordings."""
    stats = defaultdict(lambda: Counter())
    errors = []
    writer = None
    metadata_file = None
    metadata_writer = None
    written = 0
    if write_segments:
        assert output_dir is not None and expected_segments is not None
        writer = np.lib.format.open_memmap(output_dir / "X_rebuilt.npy", mode="w+", dtype=np.float32,
                                           shape=(expected_segments, len(CHANNELS), WINDOW))
        metadata_file = (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8")
        fields = ["subject_id", "trial_id", "corpus", "depression_label", "stress_label", "domain", "anchor_mask"]
        metadata_writer = csv.DictWriter(metadata_file, fieldnames=fields)
        metadata_writer.writeheader()
    for source in (modma_records, sam40_records, cogload_records):
        try:
            for corpus, subject, trial, dep, stress, raw, fs, padded in source():
                stats[corpus]["recordings"] += 1
                stats[corpus][f"subject_seen::{subject}"] += 1
                try:
                    for segment, flat in windows(np.asarray(raw, dtype=np.float64), fs, padded):
                        stats[corpus]["initial_segments"] += 1
                        if flat:
                            stats[corpus]["flatline_rejected"] += 1
                            stats[corpus][f"flatline_subject::{subject}"] += 1
                            continue
                        variances = np.var(segment, axis=1)
                        if np.max(variances) > 5.0 * np.median(variances):
                            stats[corpus]["variance_rejected"] += 1
                            stats[corpus][f"variance_subject::{subject}"] += 1
                            continue
                        anchor = dep == 0 or stress == 0
                        stats[corpus]["final_segments"] += 1
                        if writer is not None:
                            writer[written] = robust_normalize(segment)
                            metadata_writer.writerow({"subject_id": subject, "trial_id": trial, "corpus": corpus,
                                                      "depression_label": dep, "stress_label": stress,
                                                      "domain": {"MODMA": 0, "SAM40": 1, "Cognitive Load": 2}[corpus],
                                                      "anchor_mask": anchor})
                            written += 1
                except Exception as exc:
                    errors.append({"corpus": corpus, "subject_id": subject, "trial_id": trial, "error": repr(exc)})
        except Exception as exc:
            errors.append({"corpus": source.__name__, "error": repr(exc)})
    if metadata_file is not None:
        metadata_file.close()
        del writer
    return stats, errors, written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # First pass completes all rejection accounting before allocating the rebuilt
    # array.  This avoids the prior in-memory duplication of segment data.
    stats, errors, _ = process_pass(write_segments=False)
    final_segments = sum(values["final_segments"] for values in stats.values())
    if not final_segments:
        raise RuntimeError(f"No retained segments were produced. Errors: {errors[:3]}")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "parameters": {"final_fs": FS_OUT, "window_samples": WINDOW, "stride": STRIDE, "flatline_min_samples_at_128hz": FLATLINE_MIN_SAMPLES_OUT, "channels": CHANNELS}, "corpora": {}, "errors": errors, "output_shape": [int(final_segments), len(CHANNELS), WINDOW]}
    for corpus, values in stats.items():
        report["corpora"][corpus] = {k: int(v) for k, v in values.items() if "::" not in k}
        subject_counts = defaultdict(lambda: {"flatline_rejected": 0, "variance_rejected": 0})
        for k, v in values.items():
            if k.startswith("flatline_rejected::") or k.startswith("variance_rejected::"):
                reason, subject = k.split("::", 1)
                subject_counts[subject][reason] = int(v)
        # Include every source subject, including those with no rejected segments.
        subjects = sorted({k.split("::", 1)[1] for k in values if k.startswith("subject_seen::")})
        with (out / f"{corpus.lower().replace(' ', '_')}_rejection_by_subject.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject_id", "flatline_rejected", "variance_rejected"])
            writer.writeheader()
            for subject in subjects: writer.writerow({"subject_id": subject, **subject_counts[subject]})
    (out / "preprocessing_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Second pass writes a disk-backed array and metadata.  It must reproduce the
    # audited retained count exactly or the run is marked failed.
    _, write_errors, written = process_pass(write_segments=True, output_dir=out, expected_segments=int(final_segments))
    if written != final_segments or write_errors:
        raise RuntimeError(f"Write pass mismatch: wrote {written}/{final_segments}; errors={write_errors[:3]}")
    meta = pd.read_csv(out / "metadata.csv")
    report["label_counts"] = {col: {str(k): int(v) for k, v in meta[col].value_counts().items()} for col in ["corpus", "depression_label", "stress_label", "anchor_mask"]}
    report["unique_subjects_by_corpus"] = {c: int(meta.loc[meta.corpus == c, "subject_id"].nunique()) for c in meta.corpus.unique()}
    # Calculate in batches so diagnostics do not require a second full-sized copy of X.
    median_abs_max, mads = 0.0, []
    X = np.load(out / "X_rebuilt.npy", mmap_mode="r")
    for chunk in np.array_split(X, max(1, int(np.ceil(len(X) / 512)))):
        medians = np.median(chunk, axis=(1, 2))
        median_abs_max = max(median_abs_max, float(np.max(np.abs(medians))))
        mads.extend(np.median(np.abs(chunk[i] - medians[i])) for i in range(len(chunk)))
    report["normalization_diagnostics"] = {"global_segment_median_abs_max": median_abs_max, "global_segment_mad_median": float(np.median(mads))}
    (out / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
