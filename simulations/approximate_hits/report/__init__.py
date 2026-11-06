#!/usr/bin/env python3
"""Report commands: python -m simulations.approximate_hits report <command> ..."""

import argparse
import importlib
from pathlib import Path

from simulations.approximate_hits.report import common

COMMANDS = {
    "figures": ("figures", "summary figures: guaranteed vs oracle hit rates"),
    "calibrated": ("calibrated", "hit rate vs cache size: oracle, calibrated admission, certificates"),
    "certificates": ("certificates", "online hit rate: oracle retention vs provable certificates"),
    "master": ("master", "online hit rate vs cache size, one panel per workload"),
    "feature-diagram": ("feature_diagram", "2-D illustration of the 7 model inputs"),
    "trends": ("trends", "trends over k and database size"),
    "tables": ("tables", "hit-rate tables from results.json files"),
    "aker": ("aker", "table: Aker's hit rule vs calibrated admission"),
    "calibration": ("calibration", "table: threshold-selection methods"),
    "predictors": ("predictors", "table: predictor comparison"),
    "transfer": ("transfer", "table: threshold transfer across caches"),
}


def add_args(parser):
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="results directory (default: results/ next to the report package)")
    sub = parser.add_subparsers(dest="report", required=True)
    for name, (module, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text,
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        importlib.import_module(f"simulations.approximate_hits.report.{module}").add_args(p)
        command = p.get_default("run")
        p.set_defaults(run=_with_results_dir(command))


def _with_results_dir(command):
    def run(args):
        if args.results_dir is not None:
            common.RESULTS = args.results_dir
        return command(args)
    return run
