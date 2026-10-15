"""Generate the paper figures from committed result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#6B7280"


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_results(root):
    results = root / "Results"
    riemann = read_json(results / "phase2" / "riemannian.json")
    sam40 = read_json(results / "phase1" / "sam40_ceiling.json")
    depression = read_json(results / "phase1" / "depression_cv.json")
    cbramod = read_json(results / "phase2" / "cbramod_probe.json")
    ssl = read_json(results / "phase2" / "ssl_pretrain.json")
    density = read_json(results / "phase2" / "channel_density.json")
    dense_eegnet = read_json(results / "distilled" / "premise_gate.json")
    all3 = read_json(results / "all3" / "hierarchical_v2_v1.json")[
        "Hierarchical corpus-free / router=v2 spec=v1"
    ]["within_corpus"]

    stress = {
        "Riemannian": tuple(riemann["SAM40 only"]["auc_within"]),
        "SVM": tuple(sam40["SVM"]["auc_within_subject"]),
        "Random forest": tuple(sam40["Random Forest"]["auc_within_subject"]),
        "SSL random init": tuple(ssl["random init (control)"]["auc_within"]),
        "CBraMod pretrained": tuple(cbramod["pretrained"]["auc_within"]),
        "SSL pretrained": tuple(ssl["ssl pretrained"]["auc_within"]),
        "1D CNN": tuple(sam40["1D CNN"]["auc_within_subject"]),
        "EEGNet": tuple(sam40["EEGNet"]["auc_within_subject"]),
        "TSception": tuple(sam40["TSception"]["auc_within_subject"]),
        "CBraMod random init": tuple(cbramod["random init (control)"]["auc_within"]),
    }
    dep = {name: row["subject_level"]["auc"] for name, row in depression.items()}
    dep["Gradient boosting fallback"] = dep.pop("XGBoost")
    return {
        "stress": stress,
        "depression": dep,
        "density": {
            "riemannian": {int(k): v["auc_subject"] for k, v in density.items()},
            "eegnet": {
                int(k): v["subject_auc"]
                for k, v in dense_eegnet.items()
                if k != "verdict"
            },
        },
        "all3": all3,
    }


def finish(fig, output, name):
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pipeline(output):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")

    boxes = [
        (0.2, 1.5, 2.0, 2.0, "Multi-corpus EEG\nresting + task\nheterogeneous devices", BLUE),
        (2.7, 1.5, 2.0, 2.0, "Common input\n6 frontal channels\n128 Hz; 4 s windows", GREEN),
        (5.2, 2.8, 2.1, 1.3, "Depression head\nMODMA only", ORANGE),
        (5.2, 0.9, 2.1, 1.3, "Stress head\nother corpora", RED),
        (7.9, 1.5, 1.8, 2.0, "Models\nhandcrafted\nSVM | Riemannian\ndeep networks", BLUE),
        (10.2, 1.5, 1.6, 2.0, "Evaluation\nsubject-grouped CV\ncross-device\ntransfer", GREEN),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08", facecolor="white",
            edgecolor=color, linewidth=2
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    arrows = [((2.2, 2.5), (2.7, 2.5)), ((4.7, 2.5), (5.2, 3.45)),
              ((4.7, 2.5), (5.2, 1.55)), ((7.3, 3.45), (7.9, 2.8)),
              ((7.3, 1.55), (7.9, 2.2)), ((9.7, 2.5), (10.2, 2.5))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "lw": 1.8, "color": GRAY})
    ax.text(5.0, 4.55, "−1 masks unsupported labels; no sample carries both targets",
            ha="center", fontsize=10, color=GRAY)
    ax.set_title("EEG mental-state modelling and evaluation pipeline", weight="bold", pad=12)
    finish(fig, output, "01_pipeline")


def model_comparison(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    stress = sorted(data["stress"].items(), key=lambda item: item[1][0])
    labels, values = zip(*stress)
    means = [v[0] for v in values]
    errors = [v[1] for v in values]
    colors = [BLUE if label == "Riemannian" else GRAY for label in labels]
    axes[0].barh(labels, means, xerr=errors, color=colors, alpha=.9, capsize=3)
    axes[0].axvline(.5, color="black", ls="--", lw=1)
    axes[0].set(xlim=(.4, .84), xlabel="Within-subject AUC", title="Stress — SAM40")

    dep = sorted(data["depression"].items(), key=lambda item: item[1])
    labels, values = zip(*dep)
    colors = [ORANGE if label == "SVM" else GRAY for label in labels]
    axes[1].barh(labels, values, color=colors, alpha=.9)
    axes[1].axvline(.5, color="black", ls="--", lw=1)
    axes[1].set(xlim=(.4, .84), xlabel="Subject-level out-of-fold AUC",
                title="Depression — MODMA (n=53)")
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Subject-grouped cross-validation favours classical representations",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    finish(fig, output, "02_model_comparison")


def channel_density(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    configs = [("Riemannian tangent space", "riemannian", BLUE),
               ("EEGNet", "eegnet", ORANGE)]
    for ax, (title, key, color) in zip(axes, configs):
        series = data["density"][key]
        x = sorted(series)
        y = [series[n] for n in x]
        ax.plot(x, y, "o-", color=color, lw=2, ms=7)
        best = max(series, key=series.get)
        ax.scatter(best, series[best], s=110, facecolors="none", edgecolors=RED, lw=2)
        ax.annotate(f"peak: {best} ch", (best, series[best]), xytext=(6, 10),
                    textcoords="offset points", fontsize=9)
        ax.axhline(.5, color="black", ls="--", lw=1)
        ax.set(xlabel="Number of channels", title=title)
        ax.set_xticks(x)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Subject-level AUC")
    axes[0].set_ylim(.58, .76)
    fig.suptitle("More electrodes do not monotonically improve depression detection",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    finish(fig, output, "03_channel_density")


def cross_corpus_transfer(output):
    # Stable values summarized across the two committed split draws in arch_metd/PROGRESS.md.
    directions = ["Mumtaz → MODMA", "MODMA → Mumtaz"]
    split_a = [.6638, .8675]
    split_b = [.6667, .8438]
    permuted = [.5431, .3998]
    x = range(len(directions))
    width = .23
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([i - width for i in x], split_a, width, label="Real labels — split A", color=BLUE)
    ax.bar(x, split_b, width, label="Real labels — split B", color=GREEN)
    ax.bar([i + width for i in x], permuted, width, label="Permutation null", color=GRAY)
    ax.axhline(.5, color="black", ls="--", lw=1, label="Chance AUC")
    ax.set(xticks=list(x), xticklabels=directions, ylim=(.3, .92), ylabel="Subject-level AUC")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Depression transfer survives two split draws", weight="bold")
    fig.text(.98, .015, "Caution: Mumtaz diagnosis is confounded with montage",
             ha="right", color=RED, fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    finish(fig, output, "04_cross_corpus_transfer")


def all3_diagnostic(data, output):
    corpora = ["MODMA", "SAM40", "CogLoad"]
    margins = [data["all3"][name]["margin"] for name in corpora]
    colors = [GREEN if value > 0 else (GRAY if value == 0 else RED) for value in margins]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(corpora, margins, color=colors, width=.62)
    ax.axhline(0, color="black", lw=1)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in margins], padding=4)
    ax.set(ylabel="Accuracy minus within-corpus majority baseline", ylim=(-.06, .16))
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.set_title("The pooled three-class task does not improve every corpus", weight="bold")
    ax.text(.98, .04, "Hierarchical corpus-free model; subject-grouped CV",
            transform=ax.transAxes, ha="right", color=GRAY, fontsize=9)
    finish(fig, output, "05_three_class_diagnostic")


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "diagrams"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    data = load_results(root)
    pipeline(output)
    model_comparison(data, output)
    channel_density(data, output)
    cross_corpus_transfer(output)
    all3_diagnostic(data, output)
    expected = [output / f"{i:02d}_{name}.{ext}"
                for i, name in enumerate(("pipeline", "model_comparison", "channel_density",
                                           "cross_corpus_transfer", "three_class_diagnostic"), 1)
                for ext in ("png", "svg")]
    assert all(path.stat().st_size > 1_000 for path in expected)
    print(f"wrote {len(expected)} files to {output}")


if __name__ == "__main__":
    main()
