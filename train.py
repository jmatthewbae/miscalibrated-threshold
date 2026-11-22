"""Train one ResNet-50 on a split or CV fold.

    python train.py --run run_a
    python train.py --run run_a --fold 0
    python train.py --run run_b
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

import config as C
from dataset import LesionDataset, eval_transform, load_split, train_transform


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model() -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def loaders(run: str, fold: int | None):
    df = load_split(run)
    if fold is None:
        tr, va = df[~df["is_val"]], df[df["is_val"]]
    else:
        tr, va = df[df["fold"] != fold], df[df["fold"] == fold]

    train_ds = LesionDataset(tr, C.FITZ_IMAGES, train_transform())
    val_ds = LesionDataset(va, C.FITZ_IMAGES, eval_transform())

    pin = torch.cuda.is_available()
    return (
        DataLoader(train_ds, C.BATCH_SIZE, shuffle=True,
                   num_workers=C.NUM_WORKERS, pin_memory=pin, drop_last=False),
        DataLoader(val_ds, C.BATCH_SIZE, shuffle=False,
                   num_workers=C.NUM_WORKERS, pin_memory=pin),
        len(tr), len(va),
    )


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    probs, ys, bands, names = [], [], [], []
    for x, y, band, name in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast(dev.type, enabled=C.AMP and dev.type == "cuda"):
            logits = model(x).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.numpy())
        bands.extend(band)
        names.extend(name)
    return np.concatenate(probs), np.concatenate(ys), bands, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["run_a", "run_b"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = ap.parse_args()

    set_seed(C.SEED + (args.fold or 0))
    dev = device()
    tag = args.run if args.fold is None else f"{args.run}_fold{args.fold}"
    print(f"[{tag}] device={dev}")

    train_loader, val_loader, n_tr, n_va = loaders(args.run, args.fold)
    print(f"[{tag}] train={n_tr} val={n_va}")

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(dev.type, enabled=C.AMP and dev.type == "cuda")

    best_auc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y, _, _ in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=C.AMP and dev.type == "cuda"):
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.size(0)
        sched.step()

        p, y, _, _ = predict(model, val_loader, dev)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        print(f"[{tag}] epoch {epoch:2d}/{args.epochs} "
              f"loss={total / n_tr:.4f} val_auc={auc:.4f} ({time.time() - t0:.0f}s)")

        if auc == auc and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = C.CKPT_DIR / f"{tag}.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"[{tag}] best val auc={best_auc:.4f} -> {ckpt}")

    # Validation predictions. For CV folds these are the out-of-fold scores.
    p, y, bands, names = predict(model, val_loader, dev)
    pd.DataFrame({"filename": names, "band": bands, "target": y, "prob": p}).to_csv(
        C.PRED_DIR / f"{tag}_val.csv", index=False
    )
    (C.RESULT_DIR / f"{tag}_meta.json").write_text(
        json.dumps({"run": args.run, "fold": args.fold, "best_val_auc": best_auc,
                    "epochs": args.epochs, "n_train": n_tr, "n_val": n_va}, indent=2)
    )


if __name__ == "__main__":
    main()
