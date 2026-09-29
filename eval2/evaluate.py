"""
Format du CSV attendu (deux colonnes) :
    true_label,predicted_label
    Papilio_machaon,Papilio_machaon
    Vanessa_atalanta,Vanessa_cardui
    ...

Ou avec top-5 (six colonnes) :
    true_label,top1,top2,top3,top4,top5
    Papilio_machaon,Papilio_machaon,Vanessa_atalanta,...
"""

import argparse
import csv
import sys
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[AVERTISSEMENT] scikit-learn ou matplotlib non disponible — sortie texte uniquement.")


def load_predictions(path: str):
    """
    Lit un CSV et retourne (y_true, y_pred, y_top5_or_None).
    Accepte 2 colonnes (true, pred) ou 6 colonnes (true, top1…top5).
    """
    y_true, y_pred, y_top5 = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and header[0].lower() in ("true_label", "true", "ground_truth", "label"):
            pass
        else:
            if header:
                _parse_row(header, y_true, y_pred, y_top5)

        for row in reader:
            _parse_row(row, y_true, y_pred, y_top5)

    has_top5 = len(y_top5) == len(y_true) and len(y_top5) > 0
    return y_true, y_pred, y_top5 if has_top5 else None


def _parse_row(row, y_true, y_pred, y_top5):
    row = [c.strip() for c in row if c.strip()]
    if not row:
        return
    if len(row) >= 6:
        y_true.append(row[0])
        y_pred.append(row[1])
        y_top5.append(row[1:6])
    elif len(row) >= 2:
        y_true.append(row[0])
        y_pred.append(row[1])

def compute_top1_accuracy(y_true, y_pred):
    if SKLEARN_AVAILABLE:
        return accuracy_score(y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def compute_top5_accuracy(y_true, y_top5):
    if y_top5 is None:
        return None
    correct = sum(t in top5 for t, top5 in zip(y_true, y_top5))
    return correct / len(y_true)


def compute_macro_f1(y_true, y_pred):
    if SKLEARN_AVAILABLE:
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    classes = list(set(y_true))
    f1s = []
    for cls in classes:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def compute_per_class_f1(y_true, y_pred):
    if not SKLEARN_AVAILABLE:
        return {}
    classes = sorted(set(y_true))
    f1s = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return dict(zip(classes, f1s))


def confusion_matrix_data(y_true, y_pred):
    if not SKLEARN_AVAILABLE:
        return None, None
    classes = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return cm, classes

def plot_confusion_matrix(cm, classes, output_path="confusion_matrix.png", max_classes=30):
    """Affiche la matrice de confusion (limitée à max_classes pour lisibilité)."""
    if cm is None:
        return
    if len(classes) > max_classes:
        print(f"[INFO] {len(classes)} classes — matrice tronquée aux {max_classes} premières pour lisibilité.")
        cm = cm[:max_classes, :max_classes]
        classes = classes[:max_classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5), max(6, len(classes) * 0.45)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Vraie classe",
        xlabel="Classe prédite",
        title="Matrice de confusion"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Matrice de confusion sauvegardée : {output_path}")


def plot_per_class_f1(per_class_f1, output_path="per_class_f1.png", top_n=20):
    """Barplot des F1 par classe (top_n les plus mauvaises en rouge)."""
    if not per_class_f1:
        return
    classes = list(per_class_f1.keys())
    scores  = list(per_class_f1.values())
    sorted_pairs = sorted(zip(scores, classes))
    scores, classes = zip(*sorted_pairs)
    scores, classes = list(scores), list(classes)

    if len(classes) > top_n * 2:
        worst = list(zip(scores[:top_n], classes[:top_n]))
        best  = list(zip(scores[-top_n:], classes[-top_n:]))
        combined = worst + best
        scores, classes = zip(*combined)
        scores, classes = list(scores), list(classes)
        print(f"[INFO] {len(per_class_f1)} classes — graphique limité aux {top_n} meilleures et {top_n} pires.")

    colors = ["#e74c3c" if s < 0.5 else "#3498db" for s in scores]
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.4), 5))
    ax.barh(classes, scores, color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Seuil 0.5")
    ax.set_xlabel("F1-score")
    ax.set_title("F1-score par classe (rouge < 0.5)")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] F1 par classe sauvegardée : {output_path}")


def print_report(y_true, y_pred, y_top5):
    n = len(y_true)
    top1 = compute_top1_accuracy(y_true, y_pred)
    top5 = compute_top5_accuracy(y_true, y_top5)
    macro_f1 = compute_macro_f1(y_true, y_pred)

    lines = []
    lines.append("="*55)
    lines.append("   RAPPORT D'ÉVALUATION — Classification de papillons")
    lines.append("="*55)
    lines.append(f"  Nombre d'exemples      : {n}")
    lines.append(f"  Nombre de classes      : {len(set(y_true))}")
    lines.append(f"  Top-1 Accuracy         : {top1:.4f}  ({top1*100:.2f}%)")
    if top5 is not None:
        lines.append(f"  Top-5 Accuracy         : {top5:.4f}  ({top5*100:.2f}%)")
    else:
        lines.append("  Top-5 Accuracy         : N/A (CSV à 2 colonnes)")
    lines.append(f"  Macro F1-score         : {macro_f1:.4f}")
    lines.append("="*55)

    if SKLEARN_AVAILABLE:
        lines.append("\n--- Rapport par classe (scikit-learn) ---")
        lines.append(classification_report(y_true, y_pred, zero_division=0))

    report = "\n".join(lines)
    print("\n" + report)
    return report


def save_per_class_csv(y_true, y_pred, output_path):
    """Exporte precision/recall/f1/support par classe — sert de 'liste de tests avec scores'."""
    if not SKLEARN_AVAILABLE:
        return
    classes = sorted(set(y_true))
    report_dict = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["classe", "precision", "recall", "f1_score", "support"])
        for cls in classes:
            m = report_dict[cls]
            writer.writerow([cls, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                              f"{m['f1-score']:.4f}", int(m["support"])])
    print(f"[OK] Scores par classe (tests) sauvegardés : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Évalue un classifieur de papillons à partir d'un CSV de prédictions."
    )
    parser.add_argument("--predictions", "-p", required=True,
                        help="Chemin vers le CSV (true_label, predicted_label[, top2…top5])")
    parser.add_argument("--output-dir", "-o", default=".",
                        help="Dossier de sortie pour les graphiques (défaut: répertoire courant)")
    parser.add_argument("--max-classes-cm", type=int, default=30,
                        help="Nombre max de classes dans la matrice de confusion (défaut: 30)")
    args = parser.parse_args()

    print(f"[INFO] Chargement : {args.predictions}")
    y_true, y_pred, y_top5 = load_predictions(args.predictions)

    if not y_true:
        print("[ERREUR] Aucune donnée chargée. Vérifiez le format du CSV.")
        sys.exit(1)

    report = print_report(y_true, y_pred, y_top5)

    if SKLEARN_AVAILABLE:
        import os
        out = args.output_dir
        os.makedirs(out, exist_ok=True)

        with open(os.path.join(out, "rapport.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"[OK] Rapport texte sauvegardé : {os.path.join(out, 'rapport.txt')}")

        save_per_class_csv(y_true, y_pred, os.path.join(out, "scores_par_classe.csv"))

        cm, classes = confusion_matrix_data(y_true, y_pred)
        plot_confusion_matrix(cm, classes,
                              output_path=os.path.join(out, "confusion_matrix.png"),
                              max_classes=args.max_classes_cm)

        per_class_f1 = compute_per_class_f1(y_true, y_pred)
        plot_per_class_f1(per_class_f1,
                          output_path=os.path.join(out, "per_class_f1.png"))

    print("\n[DONE] Évaluation terminée.")


if __name__ == "__main__":
    main()
