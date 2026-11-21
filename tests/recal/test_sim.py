import json

import numpy as np
import pytest

from recal import sim
from tests.recal.test_system import make_stream, make_world


@pytest.fixture
def prep(tmp_path):
    rng, data, centers = make_world(seed=5)
    stream = make_stream(rng, centers, 4200, repeat_share=0.2)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    np.save(tmp_path / "db.npy", data)
    np.save(tmp_path / "q.npy", uniq.astype(np.float32))
    from recal.backend import ExactBackend
    top = ExactBackend(data).search(uniq.astype(np.float32), 100)[0].astype(np.int32)
    np.save(tmp_path / "top500.npy", top)
    np.savez(tmp_path / "streams.npz", random=inverse.astype(np.int64).ravel())
    (tmp_path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(tmp_path / "db.npy"),
                                                    "q": str(tmp_path / "q.npy")}))
    return tmp_path


@pytest.mark.parametrize("backend", ["exact", "stored"])
def test_cell_writes_the_result_shape_and_bytes(prep, tmp_path, backend):
    args = sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--budget", "0.6", "--k", "5",
                                    "--backend", backend, "--batch", "500", "--timing", "--log-served",
                                    "--cold-start", "1500", "--history", "3000", "--min-ess", "30",
                                    "--out", str(tmp_path / "out")])
    r = sim.run(args)
    assert r["method"] == "recal" and r["within_M"] and r["bytes"]["total"] <= r["M"]
    assert "bytes_first_construction" in r
    assert set(r["log"][0]) >= {"t", "n", "served", "served_retention_sum", "audited_served", "lambda", "memory",
                                "cached", "entries"}
    assert sum(b["n"] for b in r["log"]) == 4000 or sum(b["n"] for b in r["log"]) > 3000
    assert r["timing"]["n_timed"] == len(r["timing"]["hit"]) == len(r["timing"]["dt_ms"]) == 4200
    assert r["constructions"] >= 1 and r["args"]["warmup"] > 0
    assert {"miss", "audit", "candidate", "label"} >= set(r["background"]) - {"eval", "serve"}
    assert len(r["audit_pos"]) == r["counts"]["audits"]
    assert sum(b["served"] for b in r["log"]) > 0


def test_main_writes_json_atomically(prep, tmp_path):
    from recal import cells
    import sys
    out = tmp_path / "res"
    sys.argv = ["x", "--method", "recal", "--prep", str(prep), "--budget", "0.6", "--k", "5", "--backend", "exact", "--cold-start", "1500", "--history", "3000", "--min-ess", "30",
                "--out", str(out)]
    cells.main()
    data = json.loads((out / "recal.json").read_text())
    assert data["bytes"]["total"] <= data["M"] and not list(out.glob(".*tmp"))


def test_proximity_is_flagged_as_cluster_only(prep, tmp_path):
    from recal import cells
    import sys
    sys.argv = ["x", "--method", "proximity", "--prep", str(prep), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit) as e:
        cells.main()
    assert "cluster" in str(e.value)


def test_next_fraction_stays_inside_the_bracket():
    from recal.cells import next_fraction
    assert next_fraction(0.05, 1.2e6, 1.0e6, None, 0.05) < 0.05
    f = next_fraction(0.05, 1.2e6, 1.0e6, 0.045, 0.05)
    assert 0.045 < f < 0.05
    assert 0.05 < next_fraction(0.05, 0.5e6, 1.0e6, 0.05, 0.08) < 0.08
