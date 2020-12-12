black:
	black evaluate_metrics.py evaluate_doc2vec.py prose2poetry.py prose2poetry/*.py

fmt: black

.PHONY: black fmt
