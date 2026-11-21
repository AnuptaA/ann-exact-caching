#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Shared test helpers."""

import tempfile
import shutil
from pathlib import Path

#-------------------------------------------------------------------------------


def get_temp_dir(prefix: str = "test_") -> Path:
    """Create and return a temporary directory."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_temp_dir(temp_dir: Path) -> None:
    """Clean up temporary directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
