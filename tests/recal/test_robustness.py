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


def test_k_sequence_fixed():
    assert set(k_sequence(50, 10).tolist()) == {10}


def test_mixed_k_is_tied_to_the_query_vector():
    from recal.ksets import K_SET, k_by_query, k_by_vectors, k_for_vector
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(3000, 6)).astype(np.float32)
    ks = k_by_vectors(vectors)
    assert set(ks.tolist()) == set(K_SET)
    counts = np.bincount(np.searchsorted(K_SET, ks), minlength=len(K_SET)) / len(ks)
    assert np.all(np.abs(counts - 1 / len(K_SET)) < 0.04)
    assert k_for_vector(vectors[7]) == k_for_vector(vectors[7].copy()) == ks[7]
    perturbed = vectors + 1e-3
    assert (k_by_vectors(perturbed) != ks).mean() > 0.6
    idx = np.array([5, 9, 5, 9, 5])
    per_request = k_by_query(vectors, idx)
    assert per_request[0] == per_request[2] == per_request[4] == ks[5] and per_request[1] == per_request[3] == ks[9]
    assert not np.array_equal(k_by_vectors(vectors, seed=1), ks)


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
    assert set(ks.tolist()) <= {5, 10} and len(ks) == out["n_scored"] and out["constructions"] >= 1
    qk = {}
    for pos, k in zip(rec.arrays()["index"], ks):
        qk.setdefault(int(np.load(tmp_path / "streams.npz")["random"][pos]), set()).add(int(k))
    assert all(len(v) == 1 for v in qk.values())


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
