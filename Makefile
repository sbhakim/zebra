PY ?= python3
RUN = PYTHONPATH=src $(PY)
CORPUS ?= corpus/IoT-Attacks-IDS/src/attack_data

.PHONY: install lint typecheck test test-unit test-invariants test-integration \
        corpus probe-rules probe-invariants memory-report figures experiment-report \
        icc-report clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(RUN) -m ruff check src tests tools

typecheck:
	$(RUN) -m mypy

test: test-unit test-invariants
test-unit:
	$(RUN) -m pytest tests/unit -q
test-invariants:
	$(RUN) -m pytest tests/invariants -q
test-integration:
	$(RUN) -m pytest tests/integration -q -m integration

corpus:
	@test -d $(CORPUS) || (echo "corpus missing; see README" && exit 1)
	@echo "corpus OK: $$(ls $(CORPUS) | wc -l) domains"

probe-rules:
	$(RUN) -m zebra.eval.probe

# Backward-compatible command name; outputs are explicitly exploratory rule probes.
probe-invariants: probe-rules

memory-report:
	$(RUN) -m zebra.eval.memory_report

figures: memory-report
	$(RUN) -m zebra.cli figures

experiment-report:
	$(RUN) -m zebra.cli summarize

icc-report: memory-report probe-rules experiment-report
	@echo "wrote generated evidence artifacts"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
