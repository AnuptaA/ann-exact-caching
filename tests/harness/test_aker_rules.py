"""Aker's threshold rules on hand-made geometry, through the real replay drivers.

Units are squared L2: with the nearest neighbor at Euclidean distance 2 the
initial per-entry threshold is minD/4 = 4/4 = 1.0 (squared).
Skipped when the drivers are not built.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from harness import config, data

DRIVERS = Path(os.environ.get("HARNESS_REPLAY_DIR", Path(__file__).resolve().parents[2] / "build" / "replay"))
pytestmark = pytest.mark.skipif(not (DRIVERS / "ak_replay_standard").exists(), reason="drivers not built")

MISS, EXACT, APPROX, LINKED = 0, 1, 2, 3


def run(tmp_path, mode, xs, uids, settings):
    """Replay requests at positions (x, 0); every answer list is base rows 0..9."""
    base = np.zeros((40, 2), dtype=np.float32)
    base[:, 0] = np.arange(40) * 100.0                      # far away from the requests
    requests = np.zeros((len(xs), 2), dtype=np.float32)
    requests[:, 0] = xs
    n = len(xs)
    data.write_f32bin(tmp_path / "base.f32bin", base)
    data.write_f32bin(tmp_path / "req.f32bin", requests)
    np.asarray(uids, dtype="<i8").tofile(tmp_path / "stream.i64")
    np.tile(np.arange(10, dtype="<i8"), (n, 1)).tofile(tmp_path / "answers.i64")
    dists = np.linspace(2.0, 3.0, 10, dtype=np.float32)      # nearest answer at distance 2
    np.tile(dists, (n, 1)).astype("<f4").tofile(tmp_path / "dists.f32")
    subprocess.run([str(DRIVERS / f"ak_replay_{mode}"), "--base", str(tmp_path / "base.f32bin"),
                    "--requests", str(tmp_path / "req.f32bin"), "--stream", str(tmp_path / "stream.i64"),
                    "--answers", str(tmp_path / "answers.i64"), "--dists", str(tmp_path / "dists.f32"),
                    "--out-dir", str(tmp_path), *settings.driver_args()], check=True, timeout=60,
                   capture_output=True)
    return np.fromfile(tmp_path / "outcome.u8", dtype=np.uint8).tolist()


STANDARD = config.AkerSettings(mode="standard", pool_size=40)


def test_standard_initial_threshold_is_min_dist_over_four_then_tightens(tmp_path):
    # B at 0.9 (sq 0.81 < 1.0) hits; the hit tightens to 0.25, so C at 0.6 (sq 0.36) misses
    assert run(tmp_path, "standard", [0.0, 0.9, 0.6], [0, 1, 2], STANDARD) == [MISS, APPROX, MISS]


def test_standard_just_outside_the_initial_threshold_misses(tmp_path):
    assert run(tmp_path, "standard", [0.0, 1.1], [0, 1], STANDARD) == [MISS, MISS]    # sq 1.21 > 1.0


def test_standard_equality_hit_loosens_by_alpha_l(tmp_path):
    # thresh: 1.0 -> 0.25 (approx hit) -> 0.275 (equality hit); E at sq 0.27 then hits
    e = np.sqrt(0.27)
    assert run(tmp_path, "standard", [0.0, 0.9, 0.0, e], [0, 1, 0, 2], STANDARD) == [MISS, APPROX, EXACT, APPROX]
    # without the equality hit the tightened 0.25 rejects the same query
    assert run(tmp_path, "standard", [0.0, 0.9, e], [0, 1, 2], STANDARD) == [MISS, APPROX, MISS]


def test_proximity_threshold_is_fixed_and_not_capped(tmp_path):
    prox = config.AkerSettings(mode="proximity", pool_size=40, global_thresh=1.0)
    # B (sq 0.81) and C (sq 0.36) both hit the same representative: no tightening;
    # F at sq 1.21 misses
    assert run(tmp_path, "proximity", [0.0, 0.9, 0.6, 1.1], [0, 1, 2, 3], prox) == [MISS, APPROX, APPROX, MISS]
    # a threshold above minD (4.0 squared) is honored, unlike standard mode's cap at minD
    wide = config.AkerSettings(mode="proximity", pool_size=40, global_thresh=9.0)
    assert run(tmp_path, "proximity", [0.0, 2.9], [0, 1], wide) == [MISS, APPROX]       # sq 8.41 < 9
    assert run(tmp_path, "standard", [0.0, 2.9], [0, 1], STANDARD) == [MISS, MISS]      # capped at minD=4


def test_linked_repeat_is_approximate_and_proximity_never_links(tmp_path):
    # standard: B approx-hits A and is linked; a repeat of B is served by A's entry (linked)
    assert run(tmp_path, "standard", [0.0, 0.9, 0.9, 0.0], [0, 1, 1, 0], STANDARD) == [MISS, APPROX, LINKED, EXACT]
    # proximity: linking is a no-op, so the repeat of B is an approximate hit again; A itself is equality
    prox = config.AkerSettings(mode="proximity", pool_size=40, global_thresh=1.0)
    assert run(tmp_path, "proximity", [0.0, 0.9, 0.9, 0.0], [0, 1, 1, 0], prox) == [MISS, APPROX, APPROX, EXACT]
