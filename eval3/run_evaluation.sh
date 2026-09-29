#!/bin/bash
set -e

(cd prototype && uv run python main.py train)

(cd prototype && uv run python predict.py)
cp prototype/predictions.csv predictions_prototype.csv

uv run python ../eval2/evaluate.py --predictions predictions_prototype.csv --output-dir ./resultats
