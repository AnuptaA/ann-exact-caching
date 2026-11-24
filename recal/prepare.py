"""Prepare stages of a dataset: the commands that build its shared artifacts, and a tiny in-process dataset.

Each stage names the artifact it makes, the paths it writes and the commands it runs. `scripts/prepare.py` runs
them in order, writes the manifest (path, size, sha256, command, code commit of every artifact) and pushes the
artifacts to the store. Commands for the real datasets need the cluster (data, DiskANN, PostgreSQL).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recal import artifacts

PY = ".venv-core/bin/python"
ANN = ".venv-ann/bin/python"
K_MAX = 100
SEARCH_LIST_FACTOR = 7
NPROC = "$(getconf _NPROCESSORS_ONLN)"
# derived from TripClick: not uploaded unless asked for
RESTRICTED = ("tripclick",)


@dataclass
class Stage:
    name: str
    artifact: str | None
    commands: list[str]
    paths: list[str] = field(default_factory=list)


def settings(cfg: dict, dataset: str) -> dict:
    return cfg["prepare"][dataset]


def fetch_stage(cfg: dict, dataset: str) -> Stage:
    return Stage("fetch", None, [f"sh scripts/fetch_data.sh {settings(cfg, dataset)['fetch']}"])


def prep_stage(cfg: dict, dataset: str) -> Stage:
    """Request streams and exact answers to depth 7 x K_max per workload, then exact ground truth to depth 200."""
    depth = SEARCH_LIST_FACTOR * K_MAX
    cmds = []
    for w in artifacts.workloads_of(cfg, dataset).values():
        cmds.append(f"PYTHONPATH=. {PY} simulations/approximate_hits/online_prepare.py --dataset trace "
                    f"--trace {w['replay_workload']} --out {w['prep']} --R {depth}")
        cmds.append(f"PYTHONPATH=. {PY} -m scripts.precompute_answers --backend exact "
                    f"--queries {w['prep']}/q.npy --base {settings(cfg, dataset)['base']} --kmax {K_MAX} "
                    f"--gt-out {w['prep']}/gt")
    return Stage("prep", "prep", cmds, artifacts.local_paths(cfg, dataset)["prep"])


def diskann_stages(cfg: dict, dataset: str) -> list[Stage]:
    """DiskANN index at recall@10 0.95 with a search list of at least 7 x K_max, then its answer tables."""
    ws, listsize = artifacts.workloads_of(cfg, dataset), SEARCH_LIST_FACTOR * K_MAX
    first = next(iter(ws.values()))
    index_dir = str(Path(first["diskann_file"]).parent / "diskann")
    build = (f"PYTHONPATH=. {ANN} simulations/approximate_hits/eval_real_backend.py --prep {first['prep']} "
             f"--index diskann --targets 0.95 --fixed 0.95:{listsize} --R {listsize} --answer-depth {K_MAX} "
             f"--threads {NPROC} --index-dir {index_dir} --build-only")
    tables = [f"PYTHONPATH=. {ANN} simulations/approximate_hits/eval_real_backend.py --prep {w['prep']} "
              f"--index diskann --targets 0.95 --fixed 0.95:{listsize} --R {listsize} --answer-depth {K_MAX} "
              f"--threads {NPROC} --index-dir {index_dir}" for w in ws.values()]
    paths = artifacts.local_paths(cfg, dataset)
    return [Stage("diskann_index", "diskann_index", [build], paths["diskann_index"]),
            Stage("diskann_answers", "diskann_answers", tables, paths["diskann_answers"])]


def pgvector_stage(cfg: dict, dataset: str) -> Stage | None:
    """The database update jobs clone: loaded and indexed in parallel, maintenance_work_mem sized by the harness."""
    if not any(n in cfg["q4"]["update_workloads"] for n in artifacts.workloads_of(cfg, dataset)):
        return None
    cmds = [f"make q1-stage DATASET={dataset} STAGE={stage} Q1_PG_WORKERS={NPROC}"
            for stage in ("pg_load", "pg_index")]
    return Stage("pgvector_db", "pgvector_db", cmds)


def shift_stage(cfg: dict, dataset: str) -> Stage | None:
    """k-means (seed 0) once per dataset, the shift streams of every seed, then one prep folder per stream."""
    q4, ws = cfg["q4"], artifacts.workloads_of(cfg, dataset)
    names = [n for n in q4["shift_workloads"] if n in ws]
    if not names:
        return None
    s = settings(cfg, dataset)
    cmds = []
    for n in names:
        prep = ws[n]["prep"]
        out = f"{q4['shift_prep_dir']}/{n}_streams"
        seeds = " ".join(map(str, cfg["shift_seeds"]))
        cmds.append(f"PYTHONPATH=. {PY} -m scripts.make_shift_streams --official {prep}/q.npy --base {s['base']} "
                    f"--seeds {seeds} --out-dir {out}")
        for kind in ("abrupt", "gradual", "correlated"):
            for seed in cfg["shift_seeds"]:
                cmds.append(f"PYTHONPATH=. {PY} -m scripts.convert_streams shift --stream {out}/shift_{kind}_seed{seed}.npz "
                            f"--kind {kind} --official {prep}/q.npy --base {s['base']} "
                            f"--out {q4['shift_prep_dir']}/{n}_{kind}_seed{seed}{s.get('shift_flags', '')}")
    return Stage("shift_preps", "shift_preps", cmds, artifacts.local_paths(cfg, dataset)["shift_preps"])


def update_stage(cfg: dict, dataset: str) -> Stage | None:
    q4, ws = cfg["q4"], artifacts.workloads_of(cfg, dataset)
    names = [n for n in q4["update_workloads"] if n in ws]
    if not names:
        return None
    s = settings(cfg, dataset)
    cmds = []
    for n in names:
        prep = ws[n]["prep"]
        cmds.append(f"{PY} -c \"import numpy as np; np.save('{prep}/stream_ids.npy', "
                    f"np.load('{prep}/streams.npz')['{ws[n]['stream']}'])\"")
        for mode in q4["update_modes"]:
            for rate in q4["update_rates"]:
                tag = f"{mode}{round(100 * rate):d}"
                schedule = f"{q4['update_prep_dir']}/{n}_{tag}_schedule.npz"
                cmds.append(f"PYTHONPATH=. {PY} -m scripts.make_update_schedule --base {s['base']} "
                            f"--queries {prep}/q.npy --stream-ids {prep}/stream_ids.npy --fraction {rate:g} "
                            f"--mode {mode} --first 30000 --last 100000 --out {schedule}")
                cmds.append(f"PYTHONPATH=. {PY} -m scripts.convert_streams updates --prep {prep} "
                            f"--schedule {schedule} --base {s['base']} --out {q4['update_prep_dir']}/{n}_{tag}")
    return Stage("update_preps", "update_preps", cmds, artifacts.local_paths(cfg, dataset)["update_preps"])


def medrag_stages(cfg: dict) -> list[Stage]:
    w = artifacts.workloads_of(cfg, "medrag")["medrag"]
    cmd = (f"PYTHONPATH=. {PY} -m scripts.medrag_working_set --prep {w['prep']} --stream {w['stream']} "
           f"--backend-file {w['diskann_file']} --coord-bytes {w['coord_bytes']:g} --out {cfg['medrag_budgets_file']}")
    return [Stage("medrag_budgets", "medrag_budgets", [cmd], artifacts.local_paths(cfg, "medrag")["medrag_budgets"])]


def qvcache_stage(cfg: dict, dataset: str) -> Stage | None:
    s = settings(cfg, dataset)
    if "qvcache_memory" not in s:
        return None
    w = next(iter(artifacts.workloads_of(cfg, dataset).values()))
    cmd = (f"PYTHONPATH=. {PY} -m scripts.measure_qvcache_memory --base {s['base']} --coord-bytes {w['coord_bytes']:g} "
           f"--sizes {s['qvcache_memory']} --out results/qvcache_memory/{dataset}.json --update-constants")
    return Stage("qvcache_memory", None, [cmd])


def stages(cfg: dict, dataset: str) -> list[Stage]:
    """All stages of a dataset in dependency order."""
    out = [fetch_stage(cfg, dataset), prep_stage(cfg, dataset), *diskann_stages(cfg, dataset)]
    if dataset == "medrag":
        out += medrag_stages(cfg)
    out += [s for s in (pgvector_stage(cfg, dataset), shift_stage(cfg, dataset), update_stage(cfg, dataset),
                        qvcache_stage(cfg, dataset)) if s is not None]
    return out


# ---- tiny dataset: every stage in process, for tests ------------------------------------------------------------

def tiny_config(names=("tiny-a", "tiny-b"), n_vectors: int = 20000, dim: int = 12) -> dict:
    """A configuration whose Q1 and Q2 cells run on small synthetic prep folders (see make_tiny)."""
    workloads = {n: {"dataset": "tiny", "prep": f"results/online/{n}", "stream": "trace", "coord_bytes": 4,
                     "n_vectors": n_vectors, "dim": dim, "degree": 64, "budgets": [0.6], "standin": 0.6,
                     "diskann_file": f"results/backends/{n}/diskann.npz", "replay_workload": n} for n in names}
    return {"seeds": [0], "shift_seeds": [0], "q1": {"targets": [0.9], "backends": ["exact", "diskann"]},
            "q2": {"target": 0.9},
            "q3": {"target": 0.9, "systems": ["recal"], "node_shares": [50], "saturation_clients": 4,
                   "clients": [1, 2], "repetitions": 1},
            "q4": {"grid_workloads": [], "grid_budgets": [], "histories": [], "shift_workloads": [],
                   "shift_prep_dir": "results/shifts/preps", "k_workloads": [], "fixed_k": [],
                   "update_workloads": [], "update_modes": [], "update_rates": [],
                   "update_prep_dir": "results/updates/preps"},
            "workloads": workloads, "prepare": {"tiny": {"fetch": "--list", "base": "results/tiny/base.npy"}}}


def make_tiny(cfg: dict, root: Path, seed: int = 0, n_requests: int = 3000) -> dict:
    """Write the prep folder and the answer file of every tiny workload under root; returns the artifact entries."""
    from recal.backend import ExactBackend
    entries = {}
    for i, (name, w) in enumerate(cfg["workloads"].items()):
        rng = np.random.default_rng(seed + i)
        centers = rng.normal(scale=3.0, size=(30, w["dim"]))
        data = (centers[rng.integers(0, 30, w["n_vectors"])] + rng.normal(size=(w["n_vectors"], w["dim"])))
        data = data.astype(np.float32)
        pool = (centers[rng.integers(0, 30, 600)] + rng.normal(size=(600, w["dim"]))).astype(np.float32)
        stream = rng.zipf(1.3, n_requests) % len(pool)
        prep = root / w["prep"]
        prep.mkdir(parents=True, exist_ok=True)
        np.save(prep / "db.npy", data)
        np.save(prep / "q.npy", pool)
        top = ExactBackend(data).search(pool, 100)[0].astype(np.int32)
        np.save(prep / "top500.npy", top)
        np.savez(prep / "streams.npz", trace=stream.astype(np.int64))
        (prep / "meta.json").write_text(json.dumps({"kind": "files", "db": f"{w['prep']}/db.npy",
                                                    "q": f"{w['prep']}/q.npy"}))
        answers = root / w["diskann_file"]
        answers.parent.mkdir(parents=True, exist_ok=True)
        np.savez(answers, **{"answer_0.95": top[:, :10], "deep_0.95": top})
        entries[name] = {"prep": w["prep"], "diskann_answers": w["diskann_file"]}
    return entries
