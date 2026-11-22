"""Score both runs on DDI.

Fits one Youden's J threshold on Run A out-of-fold predictions, applies it to
both runs and every band, and bootstraps 95% CIs.
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

import config as C
from dataset import LesionDataset, eval_transform, load_ddi
from train import build_model, device, predict

rng = np.random.default_rng(C.SEED)


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def metrics_at(y, p, thr) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    pos, neg = y == 1, y == 0
    return {
        "n": int(len(y)),
        "n_malignant": int(pos.sum()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "sensitivity": float(pred[pos].mean()) if pos.any() else np.nan,
        "specificity": float(1 - pred[neg].mean()) if neg.any() else np.nan,
    }


def bootstrap_ci(y, p, thr, key, n=C.N_BOOTSTRAP):
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(metrics_at(y[idx], p[idx], thr)[key])
        except ValueError:
            continue
    if not vals:
        return (np.nan, np.nan)
    lo, hi = (1 - C.CI) / 2 * 100, (1 + C.CI) / 2 * 100
    return float(np.nanpercentile(vals, lo)), float(np.nanpercentile(vals, hi))


def fit_threshold() -> float:
    """Youden's J on pooled Run A out-of-fold predictions."""
    files = sorted(C.PRED_DIR.glob("run_a_fold*_val.csv"))
    if not files:
        raise FileNotFoundError(
            "No Run A fold predictions. Train the cross-validation folds first."
        )
    oof = pd.concat([pd.read_csv(f) for f in files])
    thr = youden_threshold(oof["target"].values, oof["prob"].values)
    print(f"Youden's J threshold from Run A out-of-fold predictions "
          f"(n={len(oof)}): {thr:.4f}")
    return thr


def score_on_ddi(run: str, ddi: pd.DataFrame) -> pd.DataFrame:
    ckpt = C.CKPT_DIR / f"{run}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found. Train {run} first.")
    dev = device()
    model = build_model().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))

    loader = DataLoader(
        LesionDataset(ddi, C.DDI_IMAGES, eval_transform()),
        C.BATCH_SIZE, shuffle=False, num_workers=C.NUM_WORKERS,
    )
    p, y, bands, names = predict(model, loader, dev)
    out = pd.DataFrame({"filename": names, "band": bands, "target": y, "prob": p})
    out.to_csv(C.PRED_DIR / f"{run}_ddi.csv", index=False)
    return out


def summarise(run: str, preds: pd.DataFrame, thr: float) -> list[dict]:
    rows = []
    for band in ["ALL"] + list(C.BANDS):
        sub = preds if band == "ALL" else preds[preds["band"] == band]
        if len(sub) < 2 or sub["target"].nunique() < 2:
            continue
        m = metrics_at(sub["target"], sub["prob"], thr)
        for key in ("auroc", "sensitivity", "specificity"):
            lo, hi = bootstrap_ci(sub["target"], sub["prob"], thr, key)
            m[f"{key}_lo"], m[f"{key}_hi"] = lo, hi
        rows.append({"run": run, "band": band, **m})
    return rows


def main() -> None:
    thr = fit_threshold()
    ddi = load_ddi()
    print(f"DDI test set: n={len(ddi)}, malignant={int(ddi['target'].sum())}")
    print(ddi.groupby("band")["target"].agg(["count", "sum"]).to_string())

    rows = []
    for run in ("run_a", "run_b"):
        print(f"\nScoring {run} on DDI")
        rows += summarise(run, score_on_ddi(run, ddi), thr)

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULT_DIR / "ddi_results.csv", index=False)

    print("\n" + "=" * 78)
    print("DDI held-out results. Threshold fixed from Run A, applied unchanged.")
    print("=" * 78)
    for run in ("run_a", "run_b"):
        d = df[df["run"] == run]
        label = "Run A (trained FST I-II only)" if run == "run_a" else "Run B (trained balanced)"
        print(f"\n{label}")
        for _, r in d.iterrows():
            print(f"  {r['band']:>7}  n={r['n']:>3}  "
                  f"AUROC {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}]   "
                  f"sens {r['sensitivity']:.3f} [{r['sensitivity_lo']:.3f}, {r['sensitivity_hi']:.3f}]   "
                  f"spec {r['specificity']:.3f}")

    # Aggregate minus worst band
    print("\n" + "=" * 78)
    print("Aggregate versus worst subgroup. The quantity the paper argues for.")
    print("=" * 78)
    gaps = {}
    for run in ("run_a", "run_b"):
        d = df[df["run"] == run]
        agg = d[d["band"] == "ALL"]
        sub = d[d["band"] != "ALL"]
        if agg.empty or sub.empty:
            continue
        g = {}
        for key in ("auroc", "sensitivity"):
            a = float(agg[key].iloc[0])
            worst_i = sub[key].idxmin()
            g[key] = {
                "aggregate": a,
                "worst_band": sub.loc[worst_i, "band"],
                "worst_value": float(sub.loc[worst_i, key]),
                "gap": a - float(sub.loc[worst_i, key]),
            }
            print(f"  {run}  {key:12s} aggregate {a:.3f}   "
                  f"worst {g[key]['worst_band']} {g[key]['worst_value']:.3f}   "
                  f"gap {g[key]['gap']:.3f}")
        gaps[run] = g

    (C.RESULT_DIR / "gap_summary.json").write_text(
        json.dumps({"threshold": thr, "gaps": gaps}, indent=2)
    )
    print(f"\nWrote {C.RESULT_DIR / 'ddi_results.csv'} and gap_summary.json")


if __name__ == "__main__":
    main()
