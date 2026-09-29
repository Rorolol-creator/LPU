import random
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image

from src.pretreatment import pretreat

try:
    import kagglehub
except ImportError:
    kagglehub = None  # type: ignore

DATASET_SLUG = "gpiosenka/butterfly-images40-species"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42


def _download_dataset() -> Path:
    if kagglehub is None:
        raise ImportError("Le package 'kagglehub' est requis : pip install kagglehub")
    return Path(kagglehub.dataset_download(DATASET_SLUG))


def _list_split(root: Path, split: str) -> list[dict]:
    split_dir = root / split
    return [
        {"label": class_dir.name, "path": img_path}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir())
        for img_path in sorted(class_dir.iterdir())
        if img_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def extract_features(entries: list[dict], batch_size: int = 32):
    features, labels = [], []
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        feats = Parallel(n_jobs=-1)(
            delayed(pretreat)(load_image(entry["path"])) for entry in batch
        )
        features.extend(feats)
        labels.extend(entry["label"] for entry in batch)
        print(
            f"[INFO] extraction features : {(i + len(batch)) * 100 / len(entries):.1f}%"
        )
    return np.array(features), labels


def load_data():
    root = _download_dataset()

    train_entries = _list_split(root, "train") + _list_split(root, "valid")
    test_entries = _list_split(root, "test")

    random.Random(SEED).shuffle(train_entries)

    all_classes = sorted({e["label"] for e in train_entries + test_entries})

    return train_entries, test_entries, all_classes
