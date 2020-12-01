black:
	black evaluate_metrics.py prose2poetry.py prose2poetry/*.py

fmt: black

.PHONY: black fmt
