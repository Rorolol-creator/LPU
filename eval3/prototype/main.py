import random
import sys

import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.ml import load_data, extract_features

BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


def train():
    train_data, _test_data, all_classes = load_data()

    print("[INFO] Extraction des features d'entraînement...")
    X_train, y_train = extract_features(train_data, batch_size=BATCH_SIZE)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    print("[INFO] Entraînement du classifieur...")
    clf = SGDClassifier(loss="log_loss", random_state=SEED)
    rng = random.Random(SEED)
    indices = list(range(len(X_train)))
    for epoch in range(EPOCHS):
        rng.shuffle(indices)
        for i in range(0, len(indices), BATCH_SIZE):
            idx = indices[i : i + BATCH_SIZE]
            clf.partial_fit(X_train[idx], [y_train[j] for j in idx], classes=all_classes)
        print(f"[INFO] epoch {epoch + 1}/{EPOCHS} terminée")

    print(clf)
    joblib.dump({"clf": clf, "scaler": scaler}, "model.joblib")


def main(arg: str | None = None) -> None:
    print("Hello World!")
    if arg is not None and arg == "train":
        print("training")
        train()
        return

    print("getting saved  model at model.joblib")
    saved = joblib.load("model.joblib")
    print(saved["clf"].classes_)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
