import argparse
import csv
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("[ERREUR] Le package 'kagglehub' est requis : pip install kagglehub")
    sys.exit(1)


DATASET_SLUG = "gpiosenka/butterfly-images40-species"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_and_filter(split: str = "test"):
    print(f"[INFO] Téléchargement de '{DATASET_SLUG}'...")
    root = Path(kagglehub.dataset_download(DATASET_SLUG))

    split_dir = root / split
    if not split_dir.is_dir():
        available = [p.name for p in root.iterdir() if p.is_dir()]
        print(f"[ERREUR] Split '{split}' introuvable dans {root}. "
              f"Dossiers disponibles : {available}")
        sys.exit(1)

    entries = [
        {"label": class_dir.name, "path": img_path}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir())
        for img_path in sorted(class_dir.iterdir())
        if img_path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"[INFO] {len(entries)} images chargées pour le split '{split}'.")

    return entries


def export_template_csv(filtered, output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "predicted_label"])
        for entry in filtered:
            writer.writerow([entry["label"], ""])
    print(f"[OK] Template CSV exporté : {output_path}")
    print("     → Remplissez la colonne 'predicted_label' avec vos prédictions,")
    print(f"       puis lancez : python evaluate.py --predictions {output_path}")


def print_summary(filtered):
    labels = [e["label"] for e in filtered]
    classes = sorted(set(labels))
    from collections import Counter
    counts = Counter(labels)
    most_common = counts.most_common(5)
    least_common = counts.most_common()[-5:]

    print("\n" + "="*55)
    print("   RÉSUMÉ — Butterfly Images 40 Species")
    print("="*55)
    print(f"  Nombre d'exemples      : {len(filtered)}")
    print(f"  Nombre de classes      : {len(classes)}")
    print("\n  5 classes les plus représentées :")
    for cls, cnt in most_common:
        print(f"    {cls:<40} {cnt} exemples")
    print("\n  5 classes les moins représentées :")
    for cls, cnt in least_common:
        print(f"    {cls:<40} {cnt} exemples")
    print("="*55)


def main():
    parser = argparse.ArgumentParser(
        description="Charge le dataset Butterfly Images 40 Species (Kaggle) pour évaluation."
    )
    parser.add_argument("--output", "-o", default="predictions_template.csv",
                        help="Fichier CSV de sortie (défaut: predictions_template.csv)")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"],
                        help="Split du dataset à utiliser (défaut: test)")
    args = parser.parse_args()

    filtered = load_and_filter(split=args.split)

    if not filtered:
        print("[ERREUR] Aucune entrée valide après filtrage.")
        sys.exit(1)

    print_summary(filtered)
    export_template_csv(filtered, args.output)

    print("\n[DONE] Dataset prêt.")


if __name__ == "__main__":
    main()
