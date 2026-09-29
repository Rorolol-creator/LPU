import csv

import joblib
import numpy as np

from src.ml import load_data, extract_features

BATCH_SIZE = 32


def predict(model_path: str = "model.joblib", output_csv: str = "predictions.csv"):
    saved = joblib.load(model_path)
    clf, scaler = saved["clf"], saved["scaler"]
    _train_data, test_data, _all_classes = load_data()

    print("[INFO] Extraction des features de test...")
    X_test, y_test = extract_features(test_data, batch_size=BATCH_SIZE)
    X_test = scaler.transform(X_test)

    proba = clf.predict_proba(X_test)
    top5_idx = np.argsort(-proba, axis=1)[:, :5]

    rows = []
    for true_label, idx_row in zip(y_test, top5_idx):
        top5 = [clf.classes_[i] for i in idx_row]
        rows.append([true_label, *top5])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "top1", "top2", "top3", "top4", "top5"])
        writer.writerows(rows)
    print(f"[OK] Prédictions écrites : {output_csv} ({len(rows)} lignes)")


if __name__ == "__main__":
    predict()
