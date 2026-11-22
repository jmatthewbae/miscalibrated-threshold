"""Build the Run A and Run B splits.

Runs are size-matched, class-matched and disjoint. Writes outputs/splits/.
"""

import sys

import numpy as np
import pandas as pd

import config as C


def band_of(scale: int) -> str:
    for name, (lo, hi) in C.BANDS.items():
        if scale in (lo, hi):
            return name
    return "unknown"


def load_fitzpatrick() -> pd.DataFrame:
    if not C.FITZ_CSV.exists():
        sys.exit(f"Missing {C.FITZ_CSV}. Check the data folder.")
    df = pd.read_csv(C.FITZ_CSV)

    df = df[df["three_partition_label"].isin([C.POSITIVE_LABEL, C.NEGATIVE_LABEL])].copy()
    df["fitzpatrick_scale"] = pd.to_numeric(df["fitzpatrick_scale"], errors="coerce")
    df = df[df["fitzpatrick_scale"].between(1, 6)].copy()
    df["band"] = df["fitzpatrick_scale"].astype(int).map(band_of)
    df["target"] = (df["three_partition_label"] == C.POSITIVE_LABEL).astype(int)
    df["filename"] = df["md5hash"].astype(str) + ".jpg"

    present = {p.name for p in C.FITZ_IMAGES.glob("*.jpg") if not p.name.startswith("._")}
    missing = (~df["filename"].isin(present)).sum()
    if missing:
        print(f"  note: dropping {missing} rows whose image file is absent")
        df = df[df["filename"].isin(present)].copy()

    return df.reset_index(drop=True)


def draw(df: pd.DataFrame, band: str, target: int, n: int, rng) -> pd.DataFrame:
    pool = df[(df["band"] == band) & (df["target"] == target)]
    if len(pool) < n:
        sys.exit(
            f"Cannot draw {n} images for band {band} target {target}. "
            f"Only {len(pool)} available. Lower PER_CELL in config.py."
        )
    idx = rng.choice(pool.index.values, size=n, replace=False)
    return df.loc[idx]


def build(df: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_bands = len(C.BANDS)

    # Run A: FST I-II only
    run_a = pd.concat(
        [draw(df, C.RUN_A_BAND, t, C.PER_CELL * n_bands, rng) for t in (0, 1)]
    )

    # Run B: equal per band, no overlap with Run A
    remaining = df.drop(index=run_a.index)
    run_b = pd.concat(
        [draw(remaining, b, t, C.PER_CELL, rng) for b in C.BANDS for t in (0, 1)]
    )

    return run_a.reset_index(drop=True), run_b.reset_index(drop=True)


def add_val_flag(split: pd.DataFrame, rng) -> pd.DataFrame:
    """Stratified validation slice."""
    split = split.copy()
    split["is_val"] = False
    for (band, target), grp in split.groupby(["band", "target"]):
        k = max(1, int(round(len(grp) * C.VAL_FRACTION)))
        picked = rng.choice(grp.index.values, size=k, replace=False)
        split.loc[picked, "is_val"] = True
    return split


def add_folds(split: pd.DataFrame, rng) -> pd.DataFrame:
    """Stratified CV folds (Run A only)."""
    split = split.copy()
    split["fold"] = -1
    for _, grp in split.groupby(["band", "target"]):
        order = rng.permutation(grp.index.values)
        split.loc[order, "fold"] = np.arange(len(order)) % C.N_FOLDS
    return split


def describe(name: str, split: pd.DataFrame) -> None:
    print(f"\n{name}  n={len(split)}  malignant={split['target'].mean():.1%}")
    table = split.pivot_table(
        index="band", columns="target", values="filename", aggfunc="count", fill_value=0
    )
    table.columns = ["benign", "malignant"]
    print(table.to_string())


def main() -> None:
    rng = np.random.default_rng(C.SEED)

    print("Loading Fitzpatrick17k metadata")
    df = load_fitzpatrick()
    print(f"  usable malignant/benign rows with images present: {len(df)}")

    run_a, run_b = build(df, rng)

    run_a = add_folds(add_val_flag(run_a, rng), rng)
    run_b = add_val_flag(run_b, rng)
    run_b["fold"] = -1

    describe("Run A  (skewed, FST I-II only)", run_a)
    describe("Run B  (balanced across bands)", run_b)

    assert len(run_a) == len(run_b), "runs must be the same size"
    assert run_a["target"].mean() == run_b["target"].mean(), "runs must share class balance"
    assert not set(run_a["filename"]) & set(run_b["filename"]), "runs must not overlap"
    print("\nChecks passed. Runs are size-matched, class-matched and disjoint.")

    cols = ["filename", "band", "fitzpatrick_scale", "target", "is_val", "fold"]
    run_a[cols].to_csv(C.SPLIT_DIR / "run_a.csv", index=False)
    run_b[cols].to_csv(C.SPLIT_DIR / "run_b.csv", index=False)
    print(f"Wrote splits to {C.SPLIT_DIR}")


if __name__ == "__main__":
    main()
