#!/usr/bin/env python3
"""Online simulation commands: prepare inputs, run a method, summarize runs."""

import argparse
import importlib

COMMANDS = {
    "prepare": ("prepare", "exact top-R ground truth and query streams for the ESCI database"),
    "run": ("sim", "run one method (static, periodic, triggered, lru, aker, qvcache) on a stream"),
    "summary": ("summary", "table of hit rate, retention and backend work over result files"),
}


def add_args(parser):
    sub = parser.add_subparsers(dest="action", required=True)
    for name, (module, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text,
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        importlib.import_module(f"simulations.approximate_hits.online.{module}").add_args(p)
