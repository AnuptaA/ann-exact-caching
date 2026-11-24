import json
import shlex
from pathlib import Path

import numpy as np
import pytest

from recal import matrix, modes, sim
from recal.backend import ExactBackend
from tests.recal.test_system import make_stream, make_world

CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "matrix.example.json"


def cfg():
    return json.loads(CONFIG.read_text())


def recal_cells(lines):
    for line in lines:
        tokens = shlex.split(line.split("# ")[0].replace("{", " ").replace("}", " "))
        if "recal.cells" in tokens and tokens[tokens.index("--method") + 1] == "recal":
            yield line, tokens


def backend_of(tokens):
    return tokens[tokens.index("--backend") + 1] if "--backend" in tokens else None


def test_check_mode_backend_pairs_deployable_with_diskann_and_idealized_with_exact():
    modes.check_mode_backend("deployable", "file")
    modes.check_mode_backend("deployable", "pgvector")
    modes.check_mode_backend("idealized", "exact")
    modes.check_mode_backend("idealized", "stored")
    for mode, backend in (("deployable", "exact"), ("deployable", "stored"), ("idealized", "file"), ("idealized", "pgvector")):
        with pytest.raises(ValueError):
            modes.check_mode_backend(mode, backend)
    assert modes.recal_backend("deployable") == "diskann" and modes.recal_backend("idealized") == "exact"


def test_generator_never_pairs_a_mode_with_the_wrong_backend():
    all_lines = matrix.all_lines(cfg())
    seen = {"deployable": set(), "idealized": set()}
    for exp, lines in all_lines.items():
        for line, tokens in recal_cells(lines):
            mode = "idealized" if "--idealized" in tokens else "deployable"
            backend = backend_of(tokens)
            modes.check_mode_backend(mode, backend)
            assert "--strict-modes" in tokens, (exp, line)
            seen[mode].add(backend)
    assert seen["deployable"] == {"file", "pgvector"} and seen["idealized"] == {"exact"}


def test_baselines_in_q1_run_on_both_backends():
    labels = [s[0] for s in matrix.q1_systems(cfg(), cfg()["workloads"]["spacev-0.3"])]
    for system in ("aker", "proximity-a", "proximity-b", "qvcache"):
        assert f"{system}_exact" in labels and f"{system}_diskann" in labels


@pytest.fixture
def prep(tmp_path):
    rng, data, centers = make_world(seed=5)
    stream = make_stream(rng, centers, 3000, repeat_share=0.3)
    uniq, inverse = np.unique(np.stack(stream), axis=0, return_inverse=True)
    np.save(tmp_path / "db.npy", data)
    np.save(tmp_path / "q.npy", uniq.astype(np.float32))
    np.save(tmp_path / "top500.npy", ExactBackend(data).search(uniq.astype(np.float32), 100)[0].astype(np.int32))
    np.savez(tmp_path / "streams.npz", random=inverse.astype(np.int64).ravel())
    (tmp_path / "meta.json").write_text(json.dumps({"kind": "files", "db": str(tmp_path / "db.npy"), "q": str(tmp_path / "q.npy")}))
    return tmp_path


def run_args(prep, tmp_path, *extra):
    return sim.parser().parse_args(["--prep", str(prep), "--stream", "random", "--k", "5", "--budget", "0.6",
                                    "--history", "3000", "--min-ess", "30", "--out", str(tmp_path / "o"), *extra])


def test_result_records_mode_and_backend(prep, tmp_path):
    ideal = sim.run(run_args(prep, tmp_path, "--backend", "exact", "--idealized", "--strict-modes"))
    assert ideal["mode"] == "idealized" and ideal["backend"] == "exact"
    plain = sim.run(run_args(prep, tmp_path, "--backend", "exact"))
    assert plain["mode"] == "deployable" and plain["backend"] == "exact"


def test_strict_modes_refuses_a_mismatched_run(prep, tmp_path):
    with pytest.raises(SystemExit, match="deployable ReCal runs on diskann or pgvector"):
        sim.run(run_args(prep, tmp_path, "--backend", "exact", "--strict-modes"))
    with pytest.raises(SystemExit, match="idealized ReCal runs on exact"):
        sim.run(run_args(prep, tmp_path, "--backend", "pgvector", "--pg-dsn", "x", "--idealized", "--strict-modes"))
