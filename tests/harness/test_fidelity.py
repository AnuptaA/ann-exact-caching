"""Fidelity plumbing: restricted unpickling, recall criterion, plot, and a toy run."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from harness import fidelity
from harness.runner import manifest, replay, results

DRIVERS = Path(os.environ.get("HARNESS_REPLAY_DIR", Path(__file__).resolve().parents[2] / "build" / "replay"))


class Boom:
    def __reduce__(self):
        return (os.system, ("true",))


def test_restricted_unpickler_accepts_numpy_and_rejects_everything_else(tmp_path):
    ok = tmp_path / "ok.pkl"
    ok.write_bytes(pickle.dumps([{"operation": "search", "payload": np.arange(6).reshape(2, 3)}]))
    loaded = fidelity.load_released_pickle(ok)
    assert loaded[0]["payload"].shape == (2, 3)
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps(Boom()))
    with pytest.raises(pickle.UnpicklingError):
        fidelity.load_released_pickle(bad)


def test_cumulative_ratio_recall_check_and_plot(tmp_path):
    curve = fidelity.cumulative_hit_ratio(np.array([0, 2, 1, 0]))
    assert curve.tolist() == [0.0, 0.5, 2 / 3, 0.5]
    assert fidelity.check_recall(0.85)["pass"] and not fidelity.check_recall(0.7)["pass"]
    assert fidelity.plot_hit_ratio({"1%": curve}, tmp_path / "p.png").exists()


@pytest.mark.skipif(not (DRIVERS / "ak_replay_standard").exists(), reason="drivers not built")
def test_released_stream_runs_through_the_drivers(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(-20, 20, size=(2000, 6)).astype(np.int8)
    np.save(tmp_path / "base.npy", base)
    distinct = rng.integers(-20, 20, size=(40, 6)).astype(np.int8)
    payload = distinct[rng.integers(0, 40, size=500)]

    class Exact:
        name = "exact-fake"

        def search(self, q, k, ef):
            d = ((q[:, None, :] - base.astype(np.float32)[None]) ** 2).sum(2)
            return np.argsort(d, axis=1, kind="stable")[:, :k]

    ds = fidelity.FidelityData(tmp_path / "fid", tmp_path / "base.npy", payload, Exact(), 50, list_size=20)
    cell = manifest.Cell("fid", "released", "aker", "paper", 10)
    header, arrays = results.read_result(replay.run_cell(cell, ds, DRIVERS, tmp_path / "r", tmp_path / "w"))
    assert header["summary"]["requests_measured"] == 500
    curve = fidelity.cumulative_hit_ratio(arrays["outcome"])
    assert curve[-1] > 0.5 and fidelity.check_recall(header["summary"]["delivered_recall"])["pass"]
