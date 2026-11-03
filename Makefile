PY ?= python

.PHONY: reproduce full test score stats leaderboard figures

reproduce:      ## committed kernels -> scores, tests, leaderboard, figures (CPU, no downloads)
	$(PY) -m pbi reproduce

full:           ## raw data -> stimuli -> brain/model kernels -> reproduce
	$(PY) -m pbi full

test:
	$(PY) -m pytest -q tests

score stats leaderboard figures:
	$(PY) -m pbi $@
