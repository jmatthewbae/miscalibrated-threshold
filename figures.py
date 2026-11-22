"""Paper figures. Greyscale, white background."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

RUN_LABEL = {"run_a": "Run A, trained FST I-II only", "run_b": "Run B, trained balanced"}
FILL = {"run_a": "0.75", "run_b": "0.35"}


def grouped_bars(df: pd.DataFrame, metric: str, title: str, ylabel: str, out):
    bands = [b for b in (["ALL"] + list(C.BANDS)) if b in set(df["band"])]
    x = np.arange(len(bands))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, run in enumerate(("run_a", "run_b")):
        d = df[df["run"] == run].set_index("band")
        vals = [d.loc[b, metric] if b in d.index else np.nan for b in bands]
        lo = [d.loc[b, f"{metric}_lo"] if b in d.index else np.nan for b in bands]
        hi = [d.loc[b, f"{metric}_hi"] if b in d.index else np.nan for b in bands]
        err = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
        ax.bar(x + (i - 0.5) * width, vals, width, label=RUN_LABEL[run],
               color=FILL[run], edgecolor="black", linewidth=0.8)
        ax.errorbar(x + (i - 0.5) * width, vals, yerr=np.abs(err),
                    fmt="none", ecolor="black", capsize=3, linewidth=0.8)

    if metric == "auroc":
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
        ax.text(len(bands) - 0.45, 0.505, "chance", fontsize=8, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(["Aggregate" if b == "ALL" else f"FST {b}" for b in bands])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def training_composition(out):
    """Training images per band for each run."""
    a = pd.read_csv(C.SPLIT_DIR / "run_a.csv")
    b = pd.read_csv(C.SPLIT_DIR / "run_b.csv")
    bands = list(C.BANDS)
    x = np.arange(len(bands))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, (run, d) in enumerate((("run_a", a), ("run_b", b))):
        counts = [int((d["band"] == band).sum()) for band in bands]
        bars = ax.bar(x + (i - 0.5) * width, counts, width, label=RUN_LABEL[run],
                      color=FILL[run], edgecolor="black", linewidth=0.8)
        ax.bar_label(bars, fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"FST {b}" for b in bands])
    ax.set_ylabel("Training images")
    ax.set_title(
        f"Training composition. Both runs hold {len(a)} images at "
        f"{a['target'].mean():.0%} malignant.",
        fontsize=11, pad=12,
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    path = C.RESULT_DIR / "ddi_results.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run evaluate.py first.")
    df = pd.read_csv(path)

    training_composition(C.FIG_DIR / "fig_training_composition.png")
    grouped_bars(df, "auroc",
                 "Discrimination on held-out DDI, by Fitzpatrick band",
                 "AUROC", C.FIG_DIR / "fig_auroc_by_band.png")
    grouped_bars(df, "sensitivity",
                 "Sensitivity at the fixed Run A threshold, by Fitzpatrick band",
                 "Sensitivity", C.FIG_DIR / "fig_sensitivity_by_band.png")


if __name__ == "__main__":
    main()
