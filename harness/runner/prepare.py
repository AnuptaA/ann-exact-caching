"""Prepare one dataset directory: base and official vectors, universe, streams,
answer table and ground truth. This is the layout `replay.DatasetDir` reads."""

import json
from pathlib import Path

import numpy as np

from harness import data
from harness.backend.answer_table import build_answer_table
from workloads.q1 import groundtruth, streams, universe


class RequestVectors:
    """Lazy (n_ids, d) view over every request vector of a universe."""

    def __init__(self, uni: universe.Universe, official: np.ndarray):
        self.uni, self.official = uni, official
        self.shape = (uni.n_ids, official.shape[1])

    def __len__(self) -> int:
        return self.uni.n_ids

    def __getitem__(self, item):
        ids = np.arange(self.uni.n_ids)[item]
        return universe.request_vectors(self.uni, self.official, ids)


def prepare_dataset(root: Path, name: str, base, official, backend, ef_search: int,
                    list_size: int = 50, n_groups: int = universe.N_GROUPS,
                    group_size: int = universe.GROUP_SIZE, n_requests: int = streams.N_REQUESTS,
                    zipf=streams.ZIPF_EXPONENTS, no_locality: bool = False,
                    history_fraction: float = streams.HISTORY_FRACTION, universe_seed: int = 0,
                    stream_seed: int = 0, gt_device: str = "numpy") -> Path:
    """Write base.f32bin, official.f32bin, universe.npz, streams, answers/, gt/."""
    ds = Path(root) / name
    ds.mkdir(parents=True, exist_ok=True)
    data.write_f32bin(ds / "base.f32bin", base)
    official_f32 = data.read_f32bin(data.write_f32bin(ds / "official.f32bin", official))
    uni = universe.build_universe(len(official_f32), n_groups, group_size, universe_seed)
    np.savez(ds / "universe.npz", base_idx=uni.base_idx, partner_idx=uni.partner_idx,
             lam=uni.lam, n_official=uni.n_official)
    out = {f"zipf_{s:.2f}": streams.zipf_stream(uni, n_requests, s, stream_seed) for s in zipf}
    out["uniform"] = streams.uniform_stream(uni, n_requests, stream_seed)
    if no_locality:
        out["no_locality"] = streams.no_locality_stream(uni, stream_seed)
    np.savez(ds / f"streams_seed{stream_seed}.npz", history_fraction=history_fraction, **out)
    vectors = RequestVectors(uni, official_f32)
    base_f32 = data.read_f32bin(ds / "base.f32bin")
    build_answer_table(backend, vectors, base_f32, ds / "answers", list_size, ef_search)
    groundtruth.build_ground_truth(vectors, base_f32, ds / "gt", device=gt_device)
    (ds / "prepared.json").write_text(json.dumps(
        {"universe_seed": universe_seed, "stream_seed": stream_seed, "list_size": list_size}))
    return ds
