#!/usr/bin/env python3
"""Evaluation commands: python -m simulations.approximate_hits eval <command> RUN_DIR ..."""

import argparse
import importlib

COMMANDS = {
    "core": ("core", "oracle vs retention models: knob sweep over balls and shell sample"),
    "simple": ("simple", "simple models vs gradient-boosted trees on the 7 features"),
    "final": ("final", "7-feature logistic model at every cache checkpoint, with calibration sweep"),
    "shell-ablation": ("shell_ablation", "which shell-sample features matter"),
    "shell-size": ("shell_size", "larger shell samples and calibration-set size study"),
    "shell-budget": ("shell_budget", "split a memory budget between cache and shell sample"),
    "generalize": ("generalize", "transfer of the 7-feature model across caches and across k"),
    "calibrated": ("calibrated", "ridge predictor with a calibrated threshold (12 base features)"),
    "predictors": ("predictors", "compare ridge and gradient-boosted predictors (12 + 8 features)"),
    "transfer": ("transfer", "does a calibrated threshold transfer across caches"),
    "calibration": ("calibration", "validate threshold-selection procedures by repeated draws"),
    "certificates": ("certificates", "provable retention lower bounds on saved caches"),
    "aker": ("aker", "Aker's per-entry threshold rule replayed on saved caches (trace runs)"),
}


def add_args(parser):
    sub = parser.add_subparsers(dest="evaluation", required=True)
    for name, (module, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text,
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        importlib.import_module(f"simulations.approximate_hits.eval.{module}").add_args(p)
