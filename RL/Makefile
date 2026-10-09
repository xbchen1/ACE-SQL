PYTHON ?= python3
.PHONY: syntax test check-paths

syntax:
	@for script in scripts/*.sh; do bash -n "$$script" || exit; done
	@"$(PYTHON)" -m compileall -q ace_sql_slime rewards scripts utils trainer tests

test:
	@PYTHONPATH=. "$(PYTHON)" -m unittest discover -s tests -p 'test_slime_*.py'

check-paths:
	@"$(PYTHON)" ../scripts/verify_release.py
