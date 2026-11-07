"""Per-dataset specs: how to fetch, embed and read each Q1 dataset's base and official queries."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from workloads.q1 import streams, universe
from workloads.q1.datasets import esci, sift10m, sphere, spacev, triviaqa

REPO = Path(__file__).resolve().parents[1]
PRODUCTION = ("esci", "spacev", "sift10m", "sphere")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_base: int
    dim: int
    fetch: Callable[[Path], None]
    base: Callable[[Path], np.ndarray]
    official: Callable[[Path], np.ndarray]
    embed: Optional[Callable[[Path, dict], None]] = None
    no_locality: bool = False
    n_groups: int = universe.N_GROUPS
    group_size: int = universe.GROUP_SIZE
    n_requests: int = streams.N_REQUESTS


def _esci_embed(raw: Path, env: dict) -> None:
    if (raw / "esci_products.npy").exists() and (raw / "esci_queries.npy").exists():
        return
    esci.build(raw, raw, esci.make_encoder(device=env["device"]))


def _sphere_fetch(raw: Path) -> None:
    sphere.fetch(raw)
    triviaqa.fetch_questions(raw)
    triviaqa.fetch_encoder(raw)


def _sphere_embed(raw: Path, env: dict) -> None:
    """Encode the TriviaQA questions in a separate interpreter with a minimal environment.

    The encoder checkpoint is a third-party pickle: it runs in its own venv, with no
    credentials and no network use after the download (best effort).
    """
    out = raw / "triviaqa_enc.npy"
    if out.exists():
        return
    clean = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO), "HOME": str(raw),
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([env["encode_python"], "-m", "workloads.q1", "triviaqa-encode",
                    "--questions", str(raw / "triviaqa_questions.npy"),
                    "--encoder-dir", str(raw / "dpr_web_query_encoder_hf"),
                    "--out", str(out.with_suffix(".part.npy")), "--device", env["device"]],
                   check=True, cwd=REPO, env=clean)
    out.with_suffix(".part.npy").rename(out)


def _tiny_fetch(raw: Path) -> None:
    """Synthetic clustered data for verification only (never a production dataset)."""
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(12, 16)).astype(np.float32) * 4
    pick = lambda n: centers[rng.integers(0, 12, n)] + rng.normal(size=(n, 16)).astype(np.float32) * 0.6
    np.save(raw / "base.npy", pick(4000))
    np.save(raw / "official.npy", pick(600))


SPECS = {
    "esci": DatasetSpec(
        "esci", esci.EXPECTED_PRODUCTS, esci.DIM, esci.fetch_raw,
        lambda raw: np.load(raw / "esci_products.npy", mmap_mode="r"),
        lambda raw: np.load(raw / "esci_queries.npy", mmap_mode="r"),
        embed=_esci_embed, no_locality=True),
    "spacev": DatasetSpec(
        "spacev", spacev.N_BASE, spacev.DIM, lambda raw: spacev.fetch(raw),
        lambda raw: spacev.load(raw)[0], lambda raw: spacev.load(raw)[1]),
    "sift10m": DatasetSpec(
        "sift10m", sift10m.N_BASE, sift10m.DIM, lambda raw: sift10m.fetch(raw),
        lambda raw: sift10m.load(raw)[0], lambda raw: sift10m.load(raw)[1]),
    "sphere": DatasetSpec(
        "sphere", sphere.N_BASE, sphere.DIM, _sphere_fetch,
        lambda raw: sphere.load(raw), lambda raw: np.load(raw / "triviaqa_enc.npy", mmap_mode="r"),
        embed=_sphere_embed, no_locality=True),
    "tiny": DatasetSpec(
        "tiny", 4000, 16, _tiny_fetch,
        lambda raw: np.load(raw / "base.npy"), lambda raw: np.load(raw / "official.npy"),
        no_locality=True, n_groups=100, group_size=5, n_requests=2000),
}


def get_spec(name: str) -> DatasetSpec:
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(SPECS)}")
    return SPECS[name]
