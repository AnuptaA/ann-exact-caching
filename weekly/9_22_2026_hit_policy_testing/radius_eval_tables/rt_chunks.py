#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Resumable per-(seed, query chunk) outputs, abort rules, and RSS / free-disk probes.

Each chunk writes its files to <name>.tmp, fsyncs, and renames atomically. A
<name>_done.json holding the row counts is written last; a chunk is valid only if
that file exists and every data file matches its recorded row count. A stale .tmp
is deleted on start and nothing is ever appended to a partial file."""

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import rt_config

#-------------------------------------------------------------------------------

class RunAbort(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason

def current_rss_mb() -> float:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        return float(out) / 1024.0
    except Exception:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0

def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024 ** 3

#-------------------------------------------------------------------------------
# Table io

def write_table_atomic(df: pd.DataFrame, final_path: Path, header_comments: list = None) -> None:
    final_path = Path(final_path)
    tmp = final_path.with_name(final_path.name + ".tmp")
    with gzip.open(tmp, "wt", compresslevel=4) as f:
        for line in header_comments or []:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final_path)

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", low_memory=False, float_precision="round_trip")

#-------------------------------------------------------------------------------

def run_fingerprint(dataset: str, stage: int, seeds: list, n_train: int, n_test: int, configs: list, nu_grid: list,
                    s_grid: list, chunk_size: int, N: int, query_digest: str) -> str:
    payload = dict(dataset=dataset, stage=stage, seeds=list(seeds), n_train=n_train, n_test=n_test,
                   configs=[list(c) for c in configs], nu_grid=list(nu_grid), s_grid=list(s_grid),
                   chunk_size=chunk_size, K=rt_config.K_TOPK, N=N, data_seed=rt_config.DATA_SEED,
                   ref_s=rt_config.REF_S, union_all=list(rt_config.UNION_S_ALL_SEEDS), query_digest=query_digest)
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

class ChunkLog:
    HEADER = ("timestamp,dataset,stage,seed,chunk,status,elapsed_s,rss_mb,t_matmul,t_hist,t_estimate,t_union,"
              "t_dump\n")

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            with open(self.path, "w") as f:
                f.write(self.HEADER)

    def write(self, dataset, stage, seed, chunk, status, elapsed_s, rss_mb, timings: dict = None) -> str:
        t = timings or {}
        line = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{dataset},{stage},{seed},{chunk},{status},"
                f"{elapsed_s:.2f},{rss_mb:.0f},{t.get('matmul', 0):.2f},{t.get('histograms', 0):.2f},"
                f"{t.get('estimate_and_union', 0):.2f},{t.get('union_masks', 0):.2f},{t.get('dump', 0):.2f}\n")
        with open(self.path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return line.strip()

#-------------------------------------------------------------------------------

def integrity_reason(rows: pd.DataFrame, agg: pd.DataFrame, nu_check: bool = True):
    """None if the chunk is clean, else the abort reason. Checked before a chunk is written."""
    value_cols = ["r_hat", "r_hat_sample_only", "mean_cutoff_h_over_m"]
    for c in value_cols:
        if rows[c].isna().any():
            return f"{int(rows[c].isna().sum())} NaN values in {c}"
    if nu_check and (rows["n_distinct_total"] < rows["nu"]).any():
        return "rows with fewer distinct neighbors than nu"
    if rows.duplicated(subset=["query_id", "seed", "config", "nu"]).any():
        return "duplicate (query_id, seed, config, nu) keys"
    sums = ["n", "sum_theta", "sum_rhat", "sum_theta2", "sum_rhat2", "sum_theta_rhat", "sum_relerr", "sum_relerr2"]
    if agg[sums].isna().any().any():
        return "NaN in aggregate sums"
    return None

#-------------------------------------------------------------------------------

class ChunkStore:
    def __init__(self, root: Path, dataset: str, stage: int, fingerprint: str):
        self.dir = Path(root) / "chunks" / f"{dataset}_stage{stage}_{fingerprint}"
        self.dir.mkdir(parents=True, exist_ok=True)
        for tmp in self.dir.glob("*.tmp"):
            tmp.unlink()

    def name(self, seed: int, chunk: int, part: str) -> Path:
        return self.dir / f"seed{seed}_chunk{chunk:04d}_{part}"

    def done_path(self, seed: int, chunk: int) -> Path:
        return self.name(seed, chunk, "done.json")

    def is_valid(self, seed: int, chunk: int, expected_rows: int, expected_agg: int, expect_dump: bool) -> bool:
        dp = self.done_path(seed, chunk)
        if not dp.exists():
            return False
        try:
            info = json.loads(dp.read_text())
            if info["rows"] != expected_rows or info["agg"] != expected_agg:
                return False
            if len(read_table(self.name(seed, chunk, "rows.csv.gz"))) != info["rows"]:
                return False
            if len(read_table(self.name(seed, chunk, "agg.csv.gz"))) != info["agg"]:
                return False
            if expect_dump and len(read_table(self.name(seed, chunk, "dump.csv.gz"))) != info["dump"]:
                return False
        except Exception:
            return False
        return True

    def write_chunk(self, seed: int, chunk: int, rows: pd.DataFrame, agg: pd.DataFrame, dump) -> None:
        write_table_atomic(rows, self.name(seed, chunk, "rows.csv.gz"))
        write_table_atomic(agg, self.name(seed, chunk, "agg.csv.gz"))
        info = dict(rows=len(rows), agg=len(agg), dump=0 if dump is None else len(dump))
        if dump is not None:
            write_table_atomic(dump, self.name(seed, chunk, "dump.csv.gz"))
        tmp = self.done_path(seed, chunk).with_name("done.tmp")
        tmp.write_text(json.dumps(info))
        os.replace(tmp, self.done_path(seed, chunk))

    def remove(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

def marker_path(out_stem: Path) -> Path:
    return Path(f"{out_stem}.ok")

def stage_complete(out_stem: Path, fingerprint: str) -> bool:
    mp = marker_path(out_stem)
    if not mp.exists():
        return False
    try:
        return json.loads(mp.read_text()).get("fingerprint") == fingerprint
    except Exception:
        return False

def write_abort_report(results_dir: Path, dataset: str, stage: int, reason: str) -> Path:
    path = Path(results_dir) / f"ABORT_{dataset}_stage{stage}.txt"
    path.write_text(f"{datetime.now().isoformat()}\n{dataset} stage {stage} aborted: {reason}\n")
    return path
