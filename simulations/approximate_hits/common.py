#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Helpers shared by the command modules."""

from pathlib import Path

# set by the dispatcher, not part of a command's own configuration
_DISPATCH_KEYS = {"run", "command", "workload", "evaluation", "action", "report", "results_dir", "sift_dir", "aker_dir"}


def config_of(args):
    """The command's options as a JSON-serializable dict."""
    return {k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items() if k not in _DISPATCH_KEYS}
