#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""One command-line entry point: python -m simulations.approximate_hits <command> ...

Each command group is a module with add_args(parser); it registers subcommands that
set args.run to the function to call. Groups are imported only when selected.
"""

import argparse
import importlib

from pathlib import Path

from utils.base.logs import setup_logging
from utils.cache.workloads import PATHS

GROUPS = {
    "populate": ("simulations.approximate_hits.populate",
                 "greedy cache population on a workload (sift, esci, synthetic, trace)"),
    "eval": ("simulations.approximate_hits.eval",
             "evaluate admission models and baselines on a saved run"),
    "online": ("simulations.approximate_hits.online",
               "online simulator: prepare, run, summary"),
}


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m simulations.approximate_hits",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--sift-dir", type=Path, default=None,
                    help=f"SIFT1M fvecs directory (default {PATHS.sift})")
    ap.add_argument("--aker-dir", type=Path, default=None,
                    help=f"Aker workloads directory (default {PATHS.aker})")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, (module, help_text) in GROUPS.items():
        p = sub.add_parser(name, help=help_text,
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        importlib.import_module(module).add_args(p)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logging()
    if args.sift_dir is not None:
        PATHS.sift = args.sift_dir
    if args.aker_dir is not None:
        PATHS.aker = args.aker_dir
    return args.run(args)
