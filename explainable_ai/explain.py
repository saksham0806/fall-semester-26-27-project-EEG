"""Explain the saved screening SVMs without fitting or changing any model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
# Import the unchanged extractor without the package's TensorFlow/torch imports.
sys.path.insert(0, str(ROOT / "src" / "proposed_model"))
from features import ASYMMETRY_PAIRS, BANDS, CHANNEL_MAP, extract_all_features

OUT = Path(__file__).resolve().parent / "results"
PROTOCOLS = {"resting": "resting_healthy_vs_depressed",
             "screen": "screen_healthy_vs_ill", "task": "task_healthy_vs_stressed"}
CORPORA = {0: "MODMA", 1: "SAM40", 2: "CogLoad"}
SEED, REPEATS, PER_SUBJECT = 42, 5, 4


def feature_names():
    """Same ordering as scripts/extract_all3_features.py:handcrafted_names()."""
    channels = list(CHANNEL_MAP)
    names = [f"relpsd_{b}_{c}" for b in BANDS for c in channels]
    names += [f"de_{b}_{c}" for b in BANDS for c in channels]
    names += [f"faa_{r}_{l}" for r, l in ASYMMETRY_PAIRS]
    names += [f"specent_{c}" for c in channels]
    names += [f"std_{c}" for c in channels]
    names += [f"rms_{c}" for c in channels]
    return names


def feature_groups(names):
    return {
        "channel": {c: [i for i, n in enumerate(names) if c in n.split("_")[1:]]
                    for c in CHANNEL_MAP},
        "family": {label: [i for i, n in enumerate(names) if n.startswith(prefix + "_")]
                   for prefix, label in [("relpsd", "Relative PSD"), ("de", "DE"),
                                         ("faa", "Alpha asymmetry"), ("specent", "Spectral entropy"),
                                         ("std", "STD"), ("rms", "RMS")]},
        "band": {b: [i for i, n in enumerate(names)
                     if n.startswith((f"relpsd_{b}_", f"de_{b}_"))] for b in BANDS},
    }


def sample_rows(subjects, corpus, mask, per_subject=PER_SUBJECT, seed=SEED):
    """Cap window counts per (corpus, subject), without selecting by label or score."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in np.unique(corpus[mask]):
        for s in np.unique(subjects[mask & (corpus == c)]):
            candidates = np.flatnonzero(mask & (corpus == c) & (subjects == s))
            rows.extend(rng.choice(candidates, min(per_subject, len(candidates)), replace=False))
    return np.sort(rows)


def shuffle_group(F, columns, corpus, rng):
    """Use one donor row for every column in the group, within each corpus."""
    changed = F.copy()
    for c in np.unique(corpus):
        rows = np.flatnonzero(corpus == c)
        changed[np.ix_(rows, columns)] = F[np.ix_(rng.permutation(rows), columns)]
    return changed


def permutation_effects(model, F, y, corpus, groups, repeats=REPEATS, seed=SEED):
    rng = np.random.default_rng(seed)
    original = model.predict_proba(F)[:, 1]
    base = float(roc_auc_score(y, original))
    effects = {}
    for name, columns in groups.items():
        drops, shifts = [], []
        for _ in range(repeats):
            p = model.predict_proba(shuffle_group(F, columns, corpus, rng))[:, 1]
            drops.append(base - float(roc_auc_score(y, p)))
            shifts.append(float(np.abs(original - p).mean()))
        effects[name] = {"columns": columns, "auc_drop_mean": float(np.mean(drops)),
                         "auc_drop_std": float(np.std(drops)), "auc_drops": drops,
                         "mean_abs_probability_change": float(np.mean(shifts))}
    return {"sample_auc": base, "groups": effects}


def local_effects(model, row, reference):
    """Signed P(original) - P(one feature replaced); effects are NOT additive."""
    changed = np.tile(row, (len(row), 1))
    changed[np.arange(len(row)), np.arange(len(row))] = reference
    p = float(model.predict_proba(row[None])[:, 1][0])
    return p, p - model.predict_proba(changed)[:, 1]


def save_figure(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_results(result, window, folder):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, (kind, summary) in zip(axes, result["global"].items()):
        rows = sorted(summary["groups"].items(), key=lambda kv: kv[1]["auc_drop_mean"])
        means = [v["auc_drop_mean"] for _, v in rows]
        ax.barh([n for n, _ in rows], means,
                xerr=[v["auc_drop_std"] for _, v in rows], capsize=3,
                color=["#0072B2" if m >= 0 else "#D55E00" for m in means])
        ax.axvline(0, color="0.4", linewidth=0.7)
        ax.set(title=kind.capitalize(), xlabel="AUC drop after within-corpus shuffle")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{result['protocol']} | saved model, training subset\n"
                 "Error bars: shuffle standard deviation; descriptive, not validation")
    save_figure(fig, folder, "global_importance")

    local = result["local"]
    effects = np.asarray(local["probability_effects"])
    top = np.argsort(np.abs(effects))[-12:]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.barh([result["feature_names"][i] for i in top], effects[top],
            color=["#0072B2" if effects[i] >= 0 else "#D55E00" for i in top])
    ax.axvline(0, color="0.4", linewidth=0.7)
    ax.set(xlabel=f"P({result['positive_class']}) original minus feature-replaced",
           title=f"{result['protocol']} | one window, row {local['source_row']}\n"
                 f"P({result['positive_class']})={local['probability']:.3f}; "
                 f"threshold={result['threshold']:.3f}; predicted {local['predicted_class']}")
    fig.text(0.5, -0.025, "Reference: same-corpus median, other sampled subjects. "
             "Independent replacement effects; they do not sum to the prediction.", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, folder, "local_feature_effects")

    fig, axes = plt.subplots(6, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for ax, (c, i) in zip(axes, CHANNEL_MAP.items()):
        ax.plot(np.arange(512) / 128, window[i], color="#0072B2", linewidth=0.7)
        ax.set_ylabel(c)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title(f"Explained window | {local['corpus']}, {local['subject']} | "
                      f"dataset label: {local['observed_class']}\n"
                      "Training normalization; amplitude is not in microvolts")
    axes[-1].set_xlabel("Time (seconds at stored 128 Hz)")
    save_figure(fig, folder, "explained_window")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def explain_protocol(protocol, X, y3, subjects, corpus, provenance):
    model_path = ROOT / "models" / f"{PROTOCOLS[protocol]}.joblib"
    art = joblib.load(model_path)
    if (art.get("task") != PROTOCOLS[protocol]
            or art.get("meta", {}).get("feature_version") != "v1-81"
            or not np.array_equal(art["model"].classes_, [0, 1])):
        raise ValueError(f"Incompatible screening artifact: {model_path}")
    if not 0 <= float(art["threshold"]) <= 1:
        raise ValueError("Invalid model threshold")
    mask = (corpus == 0 if protocol == "resting" else
            np.isin(corpus, [1, 2]) if protocol == "task" else np.ones(len(y3), bool))
    idx = sample_rows(subjects, corpus, mask)
    windows = X[idx]
    if windows.shape[1:] != (6, 512) or not np.isfinite(windows).all():
        raise ValueError("Expected finite stored windows of shape (N, 6, 512)")
    F = extract_all_features(windows, fs=128).astype(np.float32)
    if F.shape != (len(idx), 81) or not np.isfinite(F).all():
        raise ValueError("Invalid v1 features")
    y = (y3[idx] != 0).astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError("Both labels are needed for permutation AUC")
    labels = {int(k): v for k, v in art["classes"].items()}
    p = art["model"].predict_proba(F)[:, 1]
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid model probabilities")
    names = feature_names()
    result = {"protocol": protocol, "model": str(model_path.relative_to(ROOT)),
              "model_sha256": file_hash(model_path), "inputs": provenance,
              "evaluation": "descriptive training subset; NOT held-out validation",
              "seed": SEED, "repeats": REPEATS, "windows_per_subject_cap": PER_SUBJECT,
              "sample_rows": idx.tolist(), "sample_size": len(idx),
              "sample_subjects": len(set(zip(corpus[idx].tolist(), subjects[idx].tolist()))),
              "positive_class": labels[1], "threshold": float(art["threshold"]),
              "existing_cv": art["cv"], "feature_names": names, "global": {}}
    for kind, groups in feature_groups(names).items():
        print(f"  {protocol}: {kind} permutation ({len(groups)} groups x {REPEATS})", flush=True)
        result["global"][kind] = permutation_effects(art["model"], F, y, corpus[idx], groups)

    # shortcut: one fixed sampled window, inspect more windows before generalising local effects.
    j = 0
    reference_mask = (corpus[idx] == corpus[idx[j]]) & (subjects[idx] != subjects[idx[j]])
    if not reference_mask.any():
        raise ValueError("Local explanation needs another sampled subject from the same corpus")
    reference = np.median(F[reference_mask], axis=0)
    probability, effects = local_effects(art["model"], F[j], reference)
    result["local"] = {"source_row": int(idx[j]), "subject": str(subjects[idx[j]]),
                       "corpus": CORPORA[int(corpus[idx[j]])], "probability": probability,
                       "observed_class": labels[int(y[j])],
                       "predicted_class": labels[int(probability >= art["threshold"])],
                       "reference_windows": int(reference_mask.sum()),
                       "feature_values": F[j].tolist(), "reference_values": reference.tolist(),
                       "probability_effects": effects.tolist()}
    folder = OUT / protocol
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / "global_importance.csv", [
        {"group_type": kind, "group": name, "auc_drop_mean": v["auc_drop_mean"],
         "auc_drop_std": v["auc_drop_std"],
         "mean_abs_probability_change": v["mean_abs_probability_change"]}
        for kind, summary in result["global"].items() for name, v in summary["groups"].items()])
    write_csv(folder / "local_feature_effects.csv", [
        {"feature": name, "value": float(F[j, i]), "reference": float(reference[i]),
         "probability_effect": float(effects[i])} for i, name in enumerate(names)])
    write_csv(folder / "sample_predictions.csv", [
        {"source_row": int(row), "subject": str(subjects[row]), "corpus": CORPORA[int(corpus[row])],
         "observed_class": labels[int(y[k])], "positive_probability": float(p[k]),
         "predicted_class": labels[int(p[k] >= art["threshold"])]} for k, row in enumerate(idx)])
    (folder / "explanation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_results(result, windows[j], folder)
    return result


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_report():
    lines = ["# Screening model explanations", "",
             "These explain existing fitted models on sampled **training data**. "
             "No models were trained. Sample AUC and permutation drops are descriptive, "
             "not estimates of performance on new subjects or devices.", "",
             "Four windows per (corpus, subject), seed 42; five shuffles per group. "
             "Shuffles stay within each corpus and move all group columns together. "
             "Positive AUC drops indicate reliance on that group in this sample; negative "
             "drops mean shuffling improved ranking. Error bars are shuffle standard deviations, "
             "not subject-level confidence intervals.", "",
             "Feature dependencies remain a limitation: group replacement can create combinations "
             "outside the training distribution. Channel groups overlap because each FAA pair "
             "belongs to both channels. Band groups contain only relative PSD and DE. "
             "Importances across groups do not add up.", "",
             "The legacy SAM40 windows used by these saved models have a documented sampling-rate "
             "error. Their band labels and time axis inherit that error; this run does not rebuild "
             "the data. Screen/task explanations can also reflect corpus acquisition differences. "
             "The task model's existing CV accuracy is below its majority baseline; explanations "
             "do not establish model quality or clinical biomarkers.", "",
             "Method reference: [scikit-learn permutation importance]"
             "(https://scikit-learn.org/stable/modules/permutation_importance.html).", ""]
    for protocol in PROTOCOLS:
        path = OUT / protocol / "explanation.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        lines += [f"## {protocol}: {r['positive_class']}", "",
                  f"Model: `{r['model']}`. Explained sample: {r['sample_size']} windows "
                  f"from {r['sample_subjects']} subjects. "
                  f"Sample AUC (training subset): {r['global']['family']['sample_auc']:.4f}.", "",
                  f"Existing subject-grouped CV, copied from the saved artifact: "
                  f"AUC {r['existing_cv']['auc']:.4f}, accuracy {r['existing_cv']['accuracy']:.4f}, "
                  f"majority {r['existing_cv']['majority_baseline']:.4f}. "
                  "These CV metrics were not recomputed by this explanation run.", "",
                  "| Group type | Largest AUC drop | Mean ± shuffle SD |",
                  "|---|---|---:|"]
        for kind, summary in r["global"].items():
            name, v = max(summary["groups"].items(), key=lambda kv: kv[1]["auc_drop_mean"])
            lines.append(f"| {kind} | {name} | {v['auc_drop_mean']:+.4f} ± {v['auc_drop_std']:.4f} |")
        local = r["local"]
        lines += ["", f"![Global importance]({protocol}/global_importance.png)", "",
                  f"Local example: source row {local['source_row']}, {local['corpus']}, "
                  f"subject `{local['subject']}`. Dataset label **{local['observed_class']}**; "
                  f"model predicts **{local['predicted_class']}** "
                  f"(P={local['probability']:.3f}, threshold={r['threshold']:.3f}). "
                  "The first sampled window is chosen without selecting for correctness or confidence.", "",
                  "Local bars are P(original) minus P(one feature replaced by the same-corpus median "
                  "from other sampled subjects). Positive bars support the positive class relative "
                  "to this reference. These are independent sensitivity effects, **not SHAP values**, "
                  "and do not sum to the prediction.", "",
                  f"![Local effects]({protocol}/local_feature_effects.png)", "",
                  f"![Explained EEG]({protocol}/explained_window.png)", "",
                  f"[Full numbers and input hashes]({protocol}/explanation.json) · "
                  f"[Global CSV]({protocol}/global_importance.csv) · "
                  f"[Local CSV]({protocol}/local_feature_effects.csv) · "
                  f"[Sample predictions]({protocol}/sample_predictions.csv)", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["all", *PROTOCOLS], default="all")
    args = parser.parse_args()
    data = ROOT / "datasets" / "all3"
    X = np.load(data / "X.npy", mmap_mode="r")
    y = np.load(data / "y.npy")
    # This is the repository's own trusted subject metadata (object dtype).
    subjects = np.load(data / "subjects.npy", allow_pickle=True)
    corpus = np.load(data / "corpus.npy")
    if (X.ndim != 3 or any(a.shape != (len(X),) for a in (y, subjects, corpus))
            or not np.isin(y, [0, 1, 2]).all() or not np.isin(corpus, [0, 1, 2]).all()
            or np.any((y == 2) & (corpus != 0)) or np.any((y == 1) & (corpus == 0))):
        raise ValueError("Unexpected all3 shapes, corpus IDs or label domains")
    provenance = {str(p.relative_to(ROOT)): file_hash(p)
                  for p in [data / name for name in ("X.npy", "y.npy", "subjects.npy", "corpus.npy")]}
    protocols = PROTOCOLS if args.protocol == "all" else [args.protocol]
    for protocol in protocols:
        print(f"Explaining {protocol} (no fitting)", flush=True)
        explain_protocol(protocol, X, y, subjects, corpus, provenance)
        write_report()
        print(f"  saved {OUT / protocol}", flush=True)


if __name__ == "__main__":
    main()
