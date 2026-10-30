PROFILE ?= default

.PHONY: install install-all ingest ask eval eval-offline baseline test lint ablation serve corpus

install:            ## core + dev tools (enough for the offline profile)
	pip install -e ".[dev,api,loaders]"

install-all:        ## everything: neural models, Claude, RAGAS
	pip install -e ".[dev,api,loaders,models,anthropic,ragas]"

corpus:             ## re-download the pinned FastAPI docs snapshot
	bash scripts/fetch_corpus.sh

ingest:
	askmydocs --profile $(PROFILE) ingest

ask:                ## make ask Q="How do I add CORS?"
	askmydocs --profile $(PROFILE) ask "$(Q)"

eval:
	askmydocs --profile $(PROFILE) eval

eval-offline:
	askmydocs --profile offline eval

baseline:           ## accept current metrics as the new regression baseline
	askmydocs --profile $(PROFILE) eval --update-baseline

ablation:
	python scripts/ablation.py --profile $(PROFILE)

test:
	pytest -q

lint:
	ruff check .

serve:
	uvicorn askmydocs.api:app --reload
