# Critères d'évaluation
*LPU (Classification de papillons)*

Pour évaluer la qualité d'un modèle de classification de papillons, nous retenons quatre critères complémentaires : **Top-1 Accuracy**, **Top-5 Accuracy**, **Macro F1-score** et **Matrice de confusion**. Ces métriques permettent d'évaluer la performance globale, la robustesse face au déséquilibre des classes, et d'identifier les espèces les plus souvent confondues.

---

## 1. Top-1 Accuracy

La Top-1 Accuracy est le rapport entre le nombre de prédictions correctes et le nombre total d'exemples :

```
Top-1 Accuracy = Nombre de prédictions exactes / Nombre total d'exemples
```

C'est la métrique la plus "simple" : elle mesure la proportion d'exemples pour lesquels le modèle donne directement la bonne réponse. Elle ne fait pas la différence entre les erreurs légères (confondre deux espèces voisines) et les erreurs grossières (confondre un morpho avec un piéride).

**Valeur raisonnable**

| Contexte | Valeur typique |
|---|---|
| Baseline aléatoire (N classes) | 1 / N |
| Modèle faible (VGG sans fine-tuning) | < 60 % |
| Modèle correct (ResNet fine-tuné) | 70 – 85 % |
| Modèle fort (EfficientNet / ViT) | > 90 % |

---

## 2. Top-5 Accuracy

La Top-5 Accuracy nous dit qu'une prédiction est correcte si la vraie classe figure parmi les 5 premières classes prédites :

```
Top-5 Accuracy = (Exemples où true_label ∈ {top1…top5}) / N
```

Elle s'applique plus particulièrement en classification fine-grained (FGVC), là où de nombreuses espèces sont visuellement très proches. Elle mesure si le modèle est au moins « dans la bonne direction », ce qui est vraiment important lorsqu'un expert humain aurait lui-même du mal à distinguer certaines espèces au premier coup d'œil.

**Valeur raisonnable**

On attend une Top-5 Accuracy significativement supérieure à la Top-1 (+10 à +20 points). Un très bon modèle dépasse 95 %. Si Top-5 ≈ Top-1, le modèle manque de diversité dans ses prédictions.

---

## 3. Macro F1-score

Pour chaque classe c, on calcule précision et rappel, puis le F1 par classe, et on en fait la moyenne non pondérée :

```
P_c = TP_c / (TP_c + FP_c)
R_c = TP_c / (TP_c + FN_c)
F1_c = 2 × P_c × R_c / (P_c + R_c)
Macro F1 = (1/C) × Σ F1_c   [moyenne sur les C classes]
```

Le Macro F1-score donne le même poids à chaque classe, indépendamment de sa fréquence. C'est essentiel pour les datasets de papillons, qui présentent souvent un fort déséquilibre (espèces communes très représentées, espèces rares peu représentées). Un modèle qui ignore les espèces rares aurait une bonne Accuracy mais un mauvais Macro F1.

**Valeur raisonnable**

Un Macro F1 > 0.80 est considéré comme bon pour un problème de classification fine-grained avec plusieurs dizaines de classes. En dessous de 0.60, le modèle peine sur les classes sous-représentées.

---

## 4. Matrice de confusion

La matrice de confusion C est une matrice carrée C × C où C[i][j] représente le nombre d'exemples de la vraie classe i prédits comme appartenant à la classe j. La diagonale représente les prédictions correctes ; les cases hors-diagonale les erreurs.

Cet outil de diagnostic permet d'identifier les paires d'espèces souvent confondues. Elle peut orienter des décisions d'augmentation de données, de fine-tuning ciblé, ou d'utilisation de modules d'attention sur des zones discriminantes (marquages alaires, etc.).

**Valeur raisonnable**

Une bonne matrice de confusion est fortement diagonale. Les erreurs hors-diagonale les plus fréquentes devraient impliquer des espèces biologiquement proches (même genre ou coloration similaire).

![Matrice de confusion](Pasted%20image.png)
*Figure 1 — Exemple de matrice de confusion sur données synthétiques (10 espèces, 200 exemples).*

![F1-score par classe](Pasted%20image%20(2).png)
*Figure 2 — F1-score par classe (rouge : F1 < 0.5). Outil de diagnostic du déséquilibre.*

---

## 5. Base d'évaluation

Le dataset que nous avons choisi d'utiliser est **Butterfly Images 40 Species** (`gpiosenka/butterfly-images40-species`), disponible sur Kaggle (compte + token API requis).

| Propriété | Valeur |
|---|---|
| Nom | Butterfly Images 40 Species |
| Source | Kaggle (gpiosenka) |
| Licence | CC0 (domaine public) |
| Accès | Kaggle, compte + token API requis |
| Lien | kaggle.com/datasets/gpiosenka/butterfly-images40-species |

Ce dataset est adapté car il fournit directement des classes propres et suffisamment représentées (100 espèces, ≥ 5 images par classe en test, images déjà recadrées sur le spécimen), avec des splits `train/valid/test` prêts à l'emploi, contrairement à Smithsonian Butterflies (dataset utilisé initialement), dont les 7493 espèces distinctes rendaient toute évaluation par espèce impossible.
