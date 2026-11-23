import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from recal import accounting, qvcache_driver as qd
from scripts import measure_qvcache_memory as mm
from scripts import validate_qvcache as v
from tests.recal.test_qvcache import FakeQVCache, args_for, fake_module, prep  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


def synthetic(per_point=990.0, fixed=5.0e6, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    points = np.array([25_000, 50_000, 100_000, 200_000])
    return points, per_point * points + fixed + rng.normal(scale=noise, size=len(points))


def test_fit_recovers_slope_and_intercept_from_noisy_measurements():
    points, extra = synthetic(noise=2e5)
    fit = mm.fit_linear(points, extra)
    assert fit["per_point"] == pytest.approx(990.0, rel=0.01) and fit["fixed"] == pytest.approx(5.0e6, rel=0.2)
    assert fit["r2"] > 0.999
    exact = mm.fit_linear(*synthetic())
    assert exact["per_point"] == pytest.approx(990.0) and exact["fixed"] == pytest.approx(5.0e6, rel=1e-6)


def test_five_percent_rule():
    model = accounting.qvcache_unit_costs(128, 4, 64)["per_point"]
    assert model == 128 * 4 + 4 + 256 == 772
    assert mm.compare({"per_point": 990.0}, model)["replace"]
    assert not mm.compare({"per_point": 800.0}, model)["replace"]
    assert mm.compare({"per_point": 740.0}, model)["relative_difference"] < 0
    assert mm.compare({"per_point": 772.0 * 1.06}, model)["replace"] and not mm.compare({"per_point": 772.0 * 1.04}, model)["replace"]


def test_measured_constants_replace_the_model_and_results_record_the_source(tmp_path, monkeypatch):
    path = tmp_path / "qvcache_memory.json"
    path.write_text("{}\n")
    monkeypatch.setattr(accounting, "QVCACHE_MEMORY_FILE", path)
    assert accounting.qvcache_unit_costs(128, 4, 64)["source"] == "model"
    capacity_model = accounting.qvcache_capacity_for_budget(1e8, 128, 4, 64, 16, 100)
    fit = mm.fit_linear(*synthetic())
    mm.update_constants(path, 128, 4, 64, fit, "measured on test")
    costs = accounting.qvcache_unit_costs(128, 4, 64)
    assert costs["per_point"] == pytest.approx(990.0) and costs["source"] == "measured:128x4x64"
    assert accounting.qvcache_capacity_for_budget(1e8, 128, 4, 64, 16, 100) < capacity_model
    full = accounting.qvcache_full_capacity_bytes(accounting.qvcache_capacity_for_budget(1e8, 128, 4, 64, 16, 100),
                                                  128, 4, 64, 16, 100)
    assert full <= 1e8 and full > 1e8 - 4 * 990 - 1
    used = accounting.qvcache_bytes_used(1000, 10, 128, 4, 64, 16)
    assert set(used) == {"points", "fixed", "regions", "total"}
    assert accounting.qvcache_unit_costs(64, 4, 64)["source"] == "model"
    model = accounting.qvcache_memory_model(128, 4, 64, 16)
    assert model["source"] == "measured:128x4x64" and model["per_point"] == pytest.approx(990.0)


def test_cell_result_records_the_memory_model(prep, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(accounting, "QVCACHE_MEMORY_FILE", tmp_path / "none.json")
    result, _ = qd.run_cell(args_for(prep, tmp_path), tmp_path / "o", fake_module())
    assert result["memory_model"]["source"] == "model" and result["memory_model"]["per_point"] > 0


def test_measure_one_reports_resident_memory_above_the_empty_cache(tmp_path):
    base = np.random.default_rng(0).normal(size=(600, 6)).astype(np.float32)
    FakeQVCache.instances.clear()
    rss = lambda: 1_000_000 + 800 * sum(len(c.vectors) for c in FakeQVCache.instances)  # noqa: E731
    out = mm.measure_one(fake_module(), base, 200, qd.settings(), 0.075, tmp_path, rss=rss, settle_s=0.0)
    assert out["points"] >= 0.98 * 200 and out["rss_empty"] == 1_000_000
    assert out["extra"] == 800 * out["points"]


def test_report_fits_the_sizes_and_compares_with_the_model():
    def runner(size):
        return {"size": size, "points": size, "rss_empty": 0, "rss_filled": 0, "extra": 990 * size + 3_000_000}
    results = mm.measure_sizes([25_000, 50_000, 100_000], runner)
    rep = mm.report(results, 128, 4.0, 64)
    assert rep["fit"]["per_point"] == pytest.approx(990.0) and rep["comparison"]["replace"]
    assert rep["comparison"]["model_per_point"] == 772


def test_rss_reads_a_positive_number():
    assert mm.rss_bytes() > 0


def test_validation_data_fetch_script_lists_its_commands_and_says_it_is_not_an_experiment():
    out = subprocess.run(["bash", str(ROOT / "scripts/fetch_qvcache_validation_data.sh"), "--dry-run"], capture_output=True,
                         text=True, check=True).stdout
    assert "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" in out
    assert out.count("fvecs_to_bin float") == 2 and "ivecs_to_bin" in out and "generate_noisy_queries.sh" in out
    assert "not an experiment dataset" in out
    assert "http://corpus-texmex.irisa.fr/siftsmall.tar.gz" in (ROOT / "scripts/fetch_qvcache_validation_data.sh").read_text()


def test_validation_stops_with_a_clear_message_when_siftsmall_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_qvcache_validation_data.sh"):
        v.ensure_data(tmp_path)
    (tmp_path / "data" / "siftsmall").mkdir(parents=True)
    (tmp_path / "data" / "siftsmall" / "siftsmall_base.bin").write_bytes(b"x")
    v.ensure_data(tmp_path)
