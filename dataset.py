"""Datasets and transforms.

No hue or saturation jitter, since skin tone is the variable under study.
"""

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config as C

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform():
    return transforms.Compose([
        transforms.Resize((C.IMAGE_SIZE + 32, C.IMAGE_SIZE + 32)),
        transforms.RandomResizedCrop(C.IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((C.IMAGE_SIZE, C.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LesionDataset(Dataset):
    """Needs columns filename, target, band."""

    def __init__(self, frame: pd.DataFrame, image_dir, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        img = Image.open(self.image_dir / row["filename"]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(float(row["target"]))
        return x, y, row["band"], row["filename"]


def load_split(name: str) -> pd.DataFrame:
    path = C.SPLIT_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_splits.py first.")
    return pd.read_csv(path)


def load_ddi() -> pd.DataFrame:
    """Load DDI (test set)."""
    df = pd.read_csv(C.DDI_CSV)
    df = df.rename(columns={"DDI_file": "filename"})
    df["target"] = df["malignant"].astype(str).str.lower().eq("true").astype(int)
    df["band"] = pd.to_numeric(df["skin_tone"], errors="coerce").map(C.DDI_TONE_TO_BAND)

    present = {p.name for p in C.DDI_IMAGES.glob("*") if not p.name.startswith("._")}
    df = df[df["filename"].isin(present)].copy()
    if df["band"].isna().any():
        df = df[df["band"].notna()].copy()
    return df.reset_index(drop=True)
