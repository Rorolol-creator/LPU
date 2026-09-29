# LPU

## Overview

We tried designing a butterfly classificator using only machine learning.
We benchmarked different solutions and tested implemeting a frozen
deep-learning to see the different results.

## final/ (C++ Random Forest)

### Build & tests

```bash
cd final
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/tests      # tests du prétraitement (Catch2)
./build/test_ml    # tests du random forest, si présent sur la branche
```

### Coverage

Pour instrumenter le build et générer les données de couverture (`gcov`), passer `-DENABLE_COVERAGE=ON` :

```bash
cd final
cmake -B build -DENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/tests
./build/test_ml
```

Chaque exécutable écrit alors des fichiers `.gcda` à côté de ses `.o` dans `build/`, exploitables avec `gcov` ou [`gcovr`](https://gcovr.com/) :

```bash
gcovr --root final --filter 'final/src/.*' --object-directory build --print-summary
```

L'option est désactivée par défaut (pas d'impact sur un build normal).
