PYTHON ?= python3

.PHONY: test syntax verify preview

test:
	"$(PYTHON)" -m pytest tests -q

syntax:
	"$(PYTHON)" -m compileall -q src tests scripts audits/benchmark_overlap_audit.py
	bash -n scripts/container.sh scripts/train_sft.sh

verify:
	"$(PYTHON)" scripts/verify_release.py

preview:
	"$(PYTHON)" -m ace_sql train --config configs/slime/paper.toml --dry-run
