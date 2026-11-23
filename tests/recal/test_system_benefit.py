import numpy as np
import pytest

from recal.system_benefit import nodecache, split


def test_node_bytes_use_stored_precision_and_links():
    assert nodecache.node_bytes(100, 1, 64) == 100 + 256 + 12
    assert nodecache.node_bytes(100, 4, 64) == 400 + 256 + 12
    assert nodecache.nodes_for_bytes(10_000, 100, 1, 64) == int(10_000 // 368)
    assert nodecache.nodes_from_index(1000, 300) == 3


def test_pick_nodes_takes_the_most_visited_with_deterministic_ties():
    visits = [[1, 2, 3], [2, 3], [3], [9]]
    assert nodecache.pick_nodes(visits, 2).tolist() == [3, 2]
    assert nodecache.pick_nodes(visits, 0).size == 0
    assert nodecache.pick_nodes(visits, 10).tolist() == [3, 2, 1, 9]


def test_arm_budgets_sum_to_the_total():
    for s in split.SPLITS:
        a = split.arm_budgets(1000.0, s)
        assert a["recal_bytes"] + a["node_cache_bytes"] == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        split.arm_budgets(1.0, 1.5)


def test_reads_curve_interpolates_and_refuses_outside():
    c = split.ReadsCurve([0, 100, 200], [10.0, 8.0, 7.0])
    assert c.at(50) == pytest.approx(9.0)
    with pytest.raises(ValueError):
        c.at(300)


def test_background_reads_count_non_miss_searches_and_rereads():
    bg = {"miss": {"searches": 100, "depth_sum": 1000}, "audit": {"searches": 10, "depth_sum": 100},
          "candidate": {"searches": 4, "depth_sum": 280}}
    reads = split.background_reads_per_request(bg, 50, lambda d: 2.0 * d, rereads=30)
    assert reads == pytest.approx((30 + 10 * 20.0 + 4 * 140.0) / 50)


def test_choose_split_trades_hits_against_the_node_cache():
    curve = split.ReadsCurve([0, 1000], [100.0, 50.0])
    hits = {0.0: 0.0, 0.10: 0.1, 0.25: 0.3, 0.50: 0.6, 0.75: 0.7, 0.90: 0.72, 1.0: 0.73}
    out = split.choose_split(hits, curve, 1000.0, {s: 1.0 for s in hits})
    assert len(out["table"]) == 7
    best = min(out["table"], key=lambda r: r["expected_reads"])
    assert out["best_split"] == best["split"] and out["best_split"] > 0.0
    flat = split.choose_split({s: 0.0 for s in hits}, curve, 1000.0, {})
    assert flat["best_split"] == 0.0


def test_node_cache_curve_with_a_mock_index_and_io_counter():
    from recal.system_benefit import curves

    class Index:
        def __init__(self, nodes):
            self.nodes = nodes

        def search(self, q, k_neighbors, complexity, beam_width):
            counter["bytes"] += 4096 * (10 - self.nodes // 100)

    counter = {"bytes": 0}
    table = curves.measure_node_cache_curve(Index, [np.zeros(2)] * 5, [0, 400], node_bytes=300.0, k=10, L=64,
                                            n_warm=2, io_counter=lambda: counter["bytes"])
    reads = [r["reads_per_search"] for r in table["rows"]]
    assert table["rows"][1]["cache_bytes"] == 120000.0
    assert reads == [pytest.approx(10.0), pytest.approx(6.0)]


def test_depth_table_format_loads_into_depth_latency(tmp_path):
    from recal.system_benefit import curves
    from simulations.approximate_hits.timing_env import DepthLatency
    import time
    table = curves.measure_depth_latency(lambda q, depth: time.sleep(depth * 1e-5), [0, 0], curves.table_depths([5, 10]))
    path = tmp_path / "t.json"
    curves.save(path, table)
    dl = DepthLatency.load(path)
    assert table["depth_latency"]["depths"] == [5, 10, 35, 70] and dl.ratio(70, 10) > 1.0


def test_job_lines_cover_splits_clients_and_depths():
    from recal.system_benefit import jobs
    cells = jobs.split_cell_lines("prep", "random", 0.05, "w")
    assert len(cells) == 6 and "recal_s10" in cells[0] and "--budget 0.005" in cells[0]
    lines = jobs.throughput_lines("spacev", "w", {"none": "none", "recal_s50": "run.json"}, {"none": 500, "recal_s50": 100})
    assert "--arm none" in lines[0] and "--clients 1 2 4 8 16 32 64" in lines[0] and "--nodes 500" in lines[0]
    assert "--arm recal_s50=run.json" in lines[1]
    assert "--clients 1 " in jobs.single_client_lines("spacev", "w", {"none": "none"})[0]
    assert "--depths 5 10 35 70" in jobs.depth_table_lines("p", "idx", "w", [5, 10, 35, 70])[0]
    assert "--nodes 0 100" in jobs.node_cache_curve_lines("idx", "q.npy", "w", [0, 100], 368.0)[0]


def test_history_runs_give_hit_rates_and_background(tmp_path):
    import json
    from recal import sim
    from recal.backend import ExactBackend
    from recal.system_benefit import history_runs
    from tests.recal.test_system import make_stream, make_world
    rng, data, centers = make_world(seed=5)
    stream = make_stream(rng, centers, 4200, repeat_share=0.2)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    np.save(tmp_path / "db.npy", data)
    np.save(tmp_path / "q.npy", uniq.astype(np.float32))
    np.save(tmp_path / "top500.npy", ExactBackend(data).search(uniq.astype(np.float32), 100)[0].astype(np.int32))
    np.savez(tmp_path / "streams.npz", random=inverse.astype(np.int64).ravel())
    (tmp_path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(tmp_path / "db.npy"), "q": str(tmp_path / "q.npy")}))
    args = sim.parser().parse_args(["--prep", str(tmp_path), "--stream", "random", "--k", "5", "--backend", "exact",
                                    "--history", "3000", "--min-ess", "30", "--out", str(tmp_path / "o")])
    runs = history_runs.split_runs(args, 0.6, 3000, (0.0, 0.5, 1.0))
    assert runs[0.0]["hit_rate"] == 0.0 and runs[0.5]["n_scored"] == 900
    assert runs[1.0]["hit_rate"] >= runs[0.5]["hit_rate"] and runs[1.0]["within_M"]
    assert "candidate" in runs[1.0]["background"]


def test_choose_split_from_history_runs_and_read_back(tmp_path, capsys):
    import json
    from recal.system_benefit import choose
    from tests.recal.test_system import make_stream, make_world
    from recal.backend import ExactBackend
    rng, data, centers = make_world(seed=5)
    stream = make_stream(rng, centers, 4200, repeat_share=0.2)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    np.save(tmp_path / "db.npy", data)
    np.save(tmp_path / "q.npy", uniq.astype(np.float32))
    np.save(tmp_path / "top500.npy", ExactBackend(data).search(uniq.astype(np.float32), 100)[0].astype(np.int32))
    np.savez(tmp_path / "streams.npz", random=inverse.astype(np.int64).ravel())
    (tmp_path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(tmp_path / "db.npy"), "q": str(tmp_path / "q.npy")}))
    total = 0.6 * data.size * 4
    curve = {"rows": [{"cache_bytes": 0.0, "reads_per_search": 100.0}, {"cache_bytes": total, "reads_per_search": 60.0}]}
    (tmp_path / "curve.json").write_text(json.dumps(curve))
    out = tmp_path / "chosen.json"
    choose.main(["run", "--prep", str(tmp_path), "--stream", "random", "--k", "5", "--backend", "exact", "--history", "3000",
                 "--min-ess", "30", "--total-fraction", "0.6", "--curve", str(tmp_path / "curve.json"),
                 "--n-vectors", str(len(data)), "--dim", str(data.shape[1]), "--degree", "16", "--coord-bytes", "4",
                 "--out", str(out)])
    chosen = json.loads(out.read_text())
    assert chosen["best_split"] in split.SPLITS and len(chosen["table"]) == 7
    assert chosen["recal_budget_fraction"] == pytest.approx(chosen["best_split"] * 0.6)
    assert chosen["node_cache_nodes"] >= 0 and chosen["history_len"] == int(0.3 * 4200)
    choose.main(["get", str(out), "recal_budget_fraction"])
    assert float(capsys.readouterr().out) == pytest.approx(chosen["recal_budget_fraction"])
