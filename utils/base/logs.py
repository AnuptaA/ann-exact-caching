#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""One logging setup for the command-line drivers."""

import logging

#-------------------------------------------------------------------------------


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
