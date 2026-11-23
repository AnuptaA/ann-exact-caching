import json

import numpy as np

from recal import robustness, sim
from recal.backend import ExactBackend
from recal.ksets import k_sequence
from tests.recal.test_system import make_stream, make_world


def test_burst_mask_marks_queries_repeated_five_times_in_a_block():
    keys = np.array([1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 1, 1, 1, 1])
    mask = robustness.burst_mask(keys, block=7, min_count=5)
    assert mask[:5].all() and not mask[5:7].any()
    assert not mask[7] and mask[8:].all()
    again = robustness.burst_mask(np.repeat(np.arange(4), 5), block=20, min_count=5)
    assert again.all() and not robustness.burst_mask(np.arange(20), block=20).any()


def test_burst_block_boundaries_are_independent():
    keys = np.array([7] * 4 + [7] * 4)
    assert not robustness.burst_mask(keys, block=4, min_count=5).any()
    assert robustness.burst_mask(keys, block=8, min_count=5).all()


def test_grid_lines_cover_the_budgets_histories_and_seeds():
    lines = robustness.grid_lines("prep", "random", "w")
    assert len(lines) == (6 + 2) * 2
    assert any("--budget 0.005 --seed 0" in line and "--history" not in line for line in lines)
    assert any("--budget 0.05 --history 10000 --history-frac 0.1" in line for line in lines)


def test_k_and_shift_lines():
    k = robustness.k_lines("prep", "random", "w")
    assert len(k) == 5 * 2 and any("--k 10 --k-mixed" in line for line in k)
    sh = robustness.shift_lines("shifts", "w")
    assert len(sh) == 3 * 2 * 5 and sum("--no-rebuild-signals" in line for line in sh) == 15
    assert any("--prep shifts/w_abrupt_seed3 --stream abrupt" in line for line in sh)


def test_k_sequence_fixed_and_mixed():
    assert set(k_sequence(50, 10).tolist()) == {10}
    mixed = k_sequence(2000, 10, True)
    assert set(mixed.tolist()) == {5, 10, 50, 100} and np.array_equal(mixed, k_sequence(2000, 10, True))


def test_mixed_k_run_records_each_requests_k(tmp_path):
    rng, data, centers = make_world(seed=5)
    stream = make_stream(rng, centers, 3200, repeat_share=0.2)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    np.save(tmp_path / "db.npy", data)
    np.save(tmp_path / "q.npy", uniq.astype(np.float32))
    np.save(tmp_path / "top500.npy", ExactBackend(data).search(uniq.astype(np.float32), 100)[0].astype(np.int32))
    np.savez(tmp_path / "streams.npz", random=inverse.astype(np.int64).ravel())
    (tmp_path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(tmp_path / "db.npy"),
                                                    "q": str(tmp_path / "q.npy")}))
    args = sim.parser().parse_args(["--prep", str(tmp_path), "--stream", "random", "--k", "5", "--k-mixed", "--k-set",
                                    "5", "10", "--backend", "exact", "--history", "3000", "--history-frac", "0.3",
                                    "--min-ess", "30", "--budget", "0.6", "--out", str(tmp_path / "o")])
    from recal.records import Records
    rec = Records()
    out = sim.run(args, rec)
    ks = rec.arrays()["K"]
    assert set(ks.tolist()) == {5, 10} and len(ks) == out["n_scored"] and out["constructions"] >= 1


def test_burst_lines_cover_budgets_with_and_without_the_mini_cache():
    b = robustness.burst_lines("p", "s", "w", budgets=(0.01, 0.05, 0.1))
    assert len(b) == 3 * 2 * 2 and sum("--no-mini-cache" in x for x in b) == 6


def test_make_shift_streams_script_on_small_arrays(tmp_path):
    from scripts import make_shift_streams
    rng = np.random.default_rng(0)
    np.save(tmp_path / "official.npy", rng.normal(size=(300, 4)).astype(np.float32))
    np.save(tmp_path / "base.npy", rng.normal(size=(2000, 4)).astype(np.float32))
    make_shift_streams.main(["--official", str(tmp_path / "official.npy"), "--base", str(tmp_path / "base.npy"),
                             "--n", "600", "--start", "200", "--span", "200", "--groups", "64", "--group-size", "4",
                             "--sample", "500", "--held-out", "0.2", "--seeds", "0", "--allow-numpy-kmeans", "--out-dir", str(tmp_path / "o")])
    names = sorted(p.name for p in (tmp_path / "o").iterdir())
    assert names == ["shift_abrupt_seed0.npz", "shift_correlated_inserts_seed0.npz", "shift_correlated_seed0.npz",
                     "shift_gradual_seed0.npz"]
    ins = np.load(tmp_path / "o" / "shift_correlated_inserts_seed0.npz")
    assert ins["t"].min() >= 200 and (~ins["initial"]).sum() == 400 and ins["insert"].all()
