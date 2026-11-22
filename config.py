"""Paths, hyperparameters and experiment constants."""

from pathlib import Path

# Paths. Falls back to ../data if ./data/ddi is missing.
ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
if not (DATA / "ddi").is_dir() and (ROOT.parent / "data" / "ddi").is_dir():
    DATA = ROOT.parent / "data"

FITZ_DIR = DATA / "fitzpatrick17k"
FITZ_CSV = FITZ_DIR / "fitzpatrick17k.csv"
FITZ_IMAGES = FITZ_DIR / "images"

DDI_DIR = DATA / "ddi"
DDI_CSV = DDI_DIR / "ddi_metadata.csv"
DDI_IMAGES = DDI_DIR / "images"

OUT = ROOT / "outputs"
SPLIT_DIR = OUT / "splits"
CKPT_DIR = OUT / "checkpoints"
PRED_DIR = OUT / "predictions"
FIG_DIR = OUT / "figures"
RESULT_DIR = OUT / "results"

for _d in (OUT, SPLIT_DIR, CKPT_DIR, PRED_DIR, FIG_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Design
SEED = 1729

# Malignant vs benign. Non-neoplastic rows are dropped to match DDI.
POSITIVE_LABEL = "malignant"
NEGATIVE_LABEL = "benign"

# Images per band per class. Capped by FST V-VI (208 malignant, 203 benign).
PER_CELL = 200

# Run A: all 1,200 from I-II. Run B: 400 per band. Both 50% malignant.
BANDS = {"I-II": (1, 2), "III-IV": (3, 4), "V-VI": (5, 6)}
RUN_A_BAND = "I-II"

VAL_FRACTION = 0.20      # validation slice per run
N_FOLDS = 5              # Run A CV folds for the threshold

# Training
ARCH = "resnet50"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
AMP = True               # mixed precision on CUDA

# Evaluation
N_BOOTSTRAP = 2000
CI = 0.95

# DDI skin_tone codes
DDI_TONE_TO_BAND = {12: "I-II", 34: "III-IV", 56: "V-VI"}
