#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Logging setup (spec 13): dual console+persistent-file handlers with a dated
timestamp, plus a Heartbeat helper for throughput/ETA logging every N items or T seconds."""

import logging
import resource
import sys
import time
from pathlib import Path

#-------------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S" 

#-------------------------------------------------------------------------------

class FlushingFileHandler(logging.FileHandler):
    """Flushes unconditionally on every emit, spec requires every log
    line to be flushed immediately, not relying on default buffering."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()

#-------------------------------------------------------------------------------

def setup_logging(stage_name: str, log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = log_dir / f"{stage_name}.log"
    file_handler = FlushingFileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")
    return logger

#-------------------------------------------------------------------------------

def current_memory_mb() -> float:
    """Peak resident set size so far, in MB (ru_maxrss is KB on Linux, bytes on macOS)."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return max_rss / divisor

#-------------------------------------------------------------------------------

class MemoryLimitExceeded(RuntimeError):
    pass

def check_memory_safe(limit_mb: float, logger: logging.Logger = None) -> None:
    mb = current_memory_mb()
    if mb > limit_mb:
        if logger:
            logger.error(f"MEMORY LIMIT EXCEEDED: {mb:.0f}MB > {limit_mb:.0f}MB -- aborting to protect the host machine")
        raise MemoryLimitExceeded(f"process memory {mb:.0f}MB exceeded limit {limit_mb:.0f}MB")

#-------------------------------------------------------------------------------

class Heartbeat:
    """Logs throughput/ETA every `every_n` items or `every_sec` seconds,
    whichever comes first. Usage:
        hb = Heartbeat(logger, total=n, label="TEST queries")
        for i, item in enumerate(items):
            ...
            hb.tick(i + 1)
        hb.done()
    """
    def __init__(self, logger: logging.Logger, total: int, label: str = "items",
                 every_n: int = 500, every_sec: float = 120.0):
        self.logger = logger
        self.total = total
        self.label = label
        self.every_n = every_n
        self.every_sec = every_sec
        self.start_time = time.time()
        self.last_log_i = 0
        self.last_log_time = self.start_time
        self.logger.info(f"START {label}: {total} total")

    def tick(self, i: int) -> None:
        now = time.time()
        if (i - self.last_log_i) < self.every_n and (now - self.last_log_time) < self.every_sec:
            return
        elapsed = now - self.start_time
        rate = i / elapsed if elapsed > 0 else 0.0
        remaining = self.total - i
        eta_sec = remaining / rate if rate > 0 else float("inf")
        self.logger.info(
            f"{self.label}: {i}/{self.total} ({100 * i / self.total:.1f}%) "
            f"throughput={rate:.1f}/s elapsed={elapsed:.0f}s ETA={eta_sec:.0f}s "
            f"mem={current_memory_mb():.0f}MB"
        )
        self.last_log_i = i
        self.last_log_time = now

    def done(self) -> None:
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0.0
        self.logger.info(
            f"END {self.label}: {self.total} total, elapsed={elapsed:.1f}s, "
            f"throughput={rate:.1f}/s, mem={current_memory_mb():.0f}MB"
        )

#-------------------------------------------------------------------------------

def load_completed_keys(out_file: Path, key_fields: tuple) -> set:
    """Read the given fields as keys already present in a partially written
    CSV file, implementing a resumable-sweep pattern copied from run_method_c.py"""
    import csv
    if not out_file.exists():
        return set()
    completed = set()
    with open(out_file, newline="") as f:
        for row in csv.DictReader(f):
            completed.add(tuple(row[field] for field in key_fields))
    return completed
