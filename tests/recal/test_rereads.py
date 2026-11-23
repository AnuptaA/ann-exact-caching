import numpy as np

from recal.backend import ExactBackend
from recal.params import Params
from recal.rereads import RereadSchedule
from recal.system import ReCal
from tests.recal.test_system import make_stream, make_world


def test_schedule_spreads_the_total_evenly_and_exactly():
    s = RereadSchedule(10)
    s.add(95)
    taken = [s.take() for _ in range(15)]
    assert sum(taken) == 95 and taken[10:] == [0] * 5 and max(taken) - min(taken[:10]) <= 1


def test_schedule_without_spread_charges_the_next_request():
    s = RereadSchedule(0)
    s.add(40)
    assert [s.take(), s.take()] == [40, 0]


def build_system(tmp_path, spread):
    rng, data, centers = make_world(seed=11)
    params = Params(K=5, k_max=20, H=3000, cold_start=1500, min_ess=30.0, seed=1, reread_spread=spread)
    system = ReCal(ExactBackend(data), params, 0.6 * data.size * 4, tmp_path)
    return system, make_stream(rng, centers, 2200, repeat_share=0.2)


def run_all(system, stream):
    return [system.serve(q, 5).rereads for q in stream]


def test_first_construction_reads_every_pool_and_shell_vector(tmp_path):
    system, stream = build_system(tmp_path, spread=0)
    rereads = run_all(system, stream)
    first = system.construction_log[0]
    assert first["new_pool_vectors"] > 0
    assert first["new_shell_vectors"] > 0
    total = sum(c["new_pool_vectors"] + c["new_shell_vectors"] for c in system.construction_log)
    assert sum(rereads) == total
    assert rereads[system.construction_log[0]["request"] - 1] == total


def test_spread_reads_come_after_the_construction_and_add_up(tmp_path):
    system, stream = build_system(tmp_path, spread=300)
    rereads = np.array(run_all(system, stream))
    at = system.construction_log[0]["request"] - 1
    assert rereads[:at].sum() == 0 and (rereads[at:at + 300] > 0).all()
    total = sum(c["new_pool_vectors"] + c["new_shell_vectors"] for c in system.construction_log)
    assert rereads.sum() == total


def test_later_construction_counts_only_new_vectors(tmp_path):
    system, stream = build_system(tmp_path, spread=0)
    run_all(system, stream)
    system.construct()
    second = system.construction_log[-1]
    first = system.construction_log[0]
    assert second["new_pool_vectors"] < first["new_pool_vectors"]
