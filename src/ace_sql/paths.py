"""Resolve user assets independently of where the package is installed."""
import os
from pathlib import Path


def project_root():
    return Path(os.environ.get('ACE_SQL_HOME', '.')).expanduser().resolve()
