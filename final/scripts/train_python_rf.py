"""Python reference for final/'s C++ random forest, used to check that both
implementations agree (same algorithm, comparable features) and to compare
training/prediction speed and memory usage.

Usage:
    uv run final/scripts/train_python_rf.py [predictions_output.csv]
"""

import csv
import resource
import sys
import time
from pathlib import Path

import kagglehub
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage import color
from skimage.feature import hog
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier

DATASET_SLUG = "gpiosenka/butterfly-images40-species"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

N_TREES = 50
MAX_DEPTH = 14
MIN_SAMPLES_LEAF = 2
SEED = 42

BASE_SIZE = (64, 64)
PYRAMID_LEVELS = 2
PYRAMID_SCALE = 0.75
MIN_PYRAMID_SIZE = 16
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
COLOR_BINS = 32


def list_split(root: Path, split: str) -> list[dict]:
    split_dir = root / split
    return [
        {"label": class_dir.name, "path": img_path}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir())
        for img_path in sorted(class_dir.iterdir())
        if img_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def color_histogram(image: np.ndarray, bins: int = COLOR_BINS) -> np.ndarray:
    hsv = color.rgb2hsv(image)
    channels = []
    for ch in range(3):
        counts, _ = np.histogram(hsv[..., ch], bins=bins, range=(0, 1))
        total = counts.sum()
        channels.append(counts / total if total > 0 else counts.astype(np.float64))
    return np.concatenate(channels).astype(np.float32)


def extract_hog(image: np.ndarray) -> np.ndarray:
    gray = color.rgb2gray(image)
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
    ).astype(np.float32)


def build_pyramid(base: np.ndarray) -> list[np.ndarray]:
    pyramid = [base]
    current = base
    for _ in range(1, PYRAMID_LEVELS):
        new_h = int(current.shape[0] * PYRAMID_SCALE)
        new_w = int(current.shape[1] * PYRAMID_SCALE)
        if new_h < MIN_PYRAMID_SIZE or new_w < MIN_PYRAMID_SIZE:
            break
        current = resize(current, (new_h, new_w), anti_aliasing=True)
        pyramid.append(current)
    return pyramid


def extract_features(image: np.ndarray) -> np.ndarray:
    """Mirrors final/src/pretreatment.cc's extractFeatures(): a HOG pyramid
    (finer levels weighted more) concatenated with an HSV color histogram."""
    base = resize(image, BASE_SIZE, anti_aliasing=True)
    pyramid = build_pyramid(base)

    features = []
    for level, level_img in enumerate(pyramid):
        normalized = resize(level_img, BASE_SIZE, anti_aliasing=True)
        weight = 1.0 / max(2 * level, 1)
        features.append(extract_hog(normalized) * weight)

    features.append(color_histogram(base))
    return np.concatenate(features)


def extract_dataset_features(entries: list[dict], batch_size: int = 32):
    features, labels = [], []
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        feats = Parallel(n_jobs=-1)(
            delayed(extract_features)(load_image(entry["path"])) for entry in batch
        )
        features.extend(feats)
        labels.extend(entry["label"] for entry in batch)
        print(
            f"[INFO] extraction features : {(i + len(batch)) * 100 / len(entries):.1f}%",
            end="\r",
        )
    print()
    return np.array(features, dtype=np.float32), labels


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("predictions_python_rf.csv")

    print("[INFO] Téléchargement/localisation du dataset Kaggle...")
    root = Path(kagglehub.dataset_download(DATASET_SLUG))

    train_entries = list_split(root, "train")
    test_entries = list_split(root, "test")

    print(f"[INFO] Extraction des features d'entraînement ({len(train_entries)} images)...")
    X_train, y_train = extract_dataset_features(train_entries)
    print(f"[INFO] Extraction des features de test ({len(test_entries)} images)...")
    X_test, y_test = extract_dataset_features(test_entries)

    print(
        f"[INFO] {X_train.shape[0]} échantillons train, {X_test.shape[0]} échantillons test, "
        f"{X_train.shape[1]} features"
    )

    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=1,
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    predict_seconds = time.perf_counter() - t0

    top5_idx = np.argsort(-proba, axis=1)[:, :5]
    rows = []
    correct_top1 = 0
    correct_top5 = 0
    for true_label, idx_row in zip(y_test, top5_idx):
        top5 = [clf.classes_[i] for i in idx_row]
        if top5[0] == true_label:
            correct_top1 += 1
        if true_label in top5:
            correct_top5 += 1
        rows.append([true_label, *top5])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "top1", "top2", "top3", "top4", "top5"])
        writer.writerows(rows)

    top1_accuracy = correct_top1 / len(y_test)
    top5_accuracy = correct_top5 / len(y_test)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f"[INFO] Entraînement terminé en {train_seconds:.4f}s")
    print(f"[INFO] Prédiction terminée en {predict_seconds:.4f}s")
    print(f"[RESULT] top1_accuracy={top1_accuracy}")
    print(f"[RESULT] top5_accuracy={top5_accuracy}")
    print(f"[RESULT] train_seconds={train_seconds}")
    print(f"[RESULT] predict_seconds={predict_seconds}")
    print(f"[RESULT] peak_memory_kb={peak_kb}")
    print(f"[OK] Prédictions écrites : {output_path}")


if __name__ == "__main__":
    main()
