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


