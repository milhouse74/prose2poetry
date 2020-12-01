#!/usr/bin/env bash

echo "rand seed 1"
python3 ./evaluate_metrics.py --rand-seed 1

echo "rand seed 3384"
python3 ./evaluate_metrics.py --rand-seed 3384

echo "rand seed 116"
python3 ./evaluate_metrics.py --rand-seed 116

echo "rand seed 42"
python3 ./evaluate_metrics.py --rand-seed 42

echo "rand seed 95555"
python3 ./evaluate_metrics.py --rand-seed 95555
