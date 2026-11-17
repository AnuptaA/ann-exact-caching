#!/usr/bin/env python3
"""Workload registry (doc 9ah): every workload of the headline and timing experiments in one place.

A workload = data (results/online/<prep>: database, queries, streams), the stream used, the cache sizes,
and its DiskANN backend (results/backends/<backend>: diskann.json / diskann.npz with the answers at the
recall target, choice.json, node_cache.json). Backend-recall variants are workloads of their own that share
the data and point at their own backend folder (no symlinked data folders). Results of workload <name> go
to results/<experiment>/<name>/.

  from simulations.approximate_hits.workloads import WORKLOADS, get
  w = get("esci1m@0.97"); w.answers_file, w.target, w.backend_info()["L"]
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

RES = Path("simulations/approximate_hits/results")
MAIN_BUDGETS = (0.003, 0.01, 0.03, 0.1)


@dataclass(frozen=True)
class Workload:
    name: str                  # registry key and results folder name
    title: str
    prep: str                  # results/online/<prep>
    stream: str = "trace"
    backend: str = ""          # results/backends/<backend> (default: prep)
    target: str = "0.95"       # recall target (key of answer_<target> / deep_<target> in diskann.npz)
    budgets: tuple = MAIN_BUDGETS
    real_log: bool = False     # real query log: drift-handling forms only (no static ours), decided 11-14
    paper_settings: bool = False   # also Aker / QVCache at the papers' own settings
    group: str = "main"        # main | sensitivity
    notes: str = field(default="", compare=False)

    @property
    def prep_dir(self):
        return RES / "online" / self.prep

    @property
    def backend_dir(self):
        return RES / "backends" / (self.backend or self.prep)

    @property
    def answers_file(self):
        return self.backend_dir / "diskann.npz"

    def backend_info(self):
        """DiskANN index, search list L and beam W at the target (diskann.json)."""
        d = json.loads((self.backend_dir / "diskann.json").read_text())
        t = d["targets"][self.target]
        return {"index_dir": d.get("index_dir") or str(self.backend_dir / "diskann"), "L": int(t["effort"]),
                "W": int(t.get("beam", d.get("beam", 2))), "recall": float(t["recall_answer_at_k"])}

    def node_cache(self):
        """Base node cache (node_cache.json of the backend folder, else of a workload on the same index)."""
        own = self.backend_dir / "node_cache.json"
        if own.exists():
            return int(json.loads(own.read_text())["chosen"]["nodes"])
        idx = self.backend_info()["index_dir"]
        for w in WORKLOADS.values():
            f = w.backend_dir / "node_cache.json"
            if w.name != self.name and f.exists() and w.backend_info()["index_dir"] == idx:
                return int(json.loads(f.read_text())["chosen"]["nodes"])
        raise KeyError(f"no node cache for {self.name}")


_W = [
    Workload("spacev-aker-0.3", "SPACEV, Zipf 0.3", "spacev-aker-0.3", paper_settings=True),
    Workload("spacev-aker-0.6", "SPACEV, Zipf 0.6", "spacev-aker-0.6", paper_settings=True),
    Workload("spacev-aker-0.99", "SPACEV, Zipf 0.99", "spacev-aker-0.99", paper_settings=True),
    Workload("esci1m", "ESCI (no repeated queries)", "esci1m", stream="random", paper_settings=True),
    Workload("triviaqa-ip-0.3", "TriviaQA, Zipf 0.3", "triviaqa-ip-0.3", budgets=(0.01, 0.1)),
    Workload("triviaqa-ip-0.6", "TriviaQA, Zipf 0.6", "triviaqa-ip-0.6", budgets=(0.01, 0.1)),
    Workload("triviaqa-ip-0.99", "TriviaQA, Zipf 0.99", "triviaqa-ip-0.99", budgets=(0.01, 0.1)),
    # main numbers at recall 0.95 (decided 11-17; the 0.9 backend is results/backends/spacev-real)
    Workload("spacev-real", "SPACEV real query log", "spacev-real", backend="spacev-real95",
             budgets=(0.01, 0.1), real_log=True),
    Workload("tripclick-ip", "TripClick real health-search log", "tripclick-ip", real_log=True),
    # 0.3% of the 23.9M corpus already holds the working set (doc 9y)
    Workload("medrag-ip", "MedRAG-Zipf (Proximity)", "medrag-ip", budgets=(0.0001, 0.0003, 0.001, 0.003)),
    # backend-recall sensitivity (doc 9ae d)
    Workload("esci1m@0.97", "ESCI, DiskANN recall 0.97", "esci1m", stream="random", backend="esci1m-r97",
             target="0.97", budgets=(0.01, 0.1), group="sensitivity"),
    Workload("spacev-aker-0.6@0.97", "SPACEV Zipf 0.6, DiskANN recall 0.97", "spacev-aker-0.6",
             backend="spacev-aker-0.6-r97", target="0.97", budgets=(0.01, 0.1), group="sensitivity"),
]
WORKLOADS = {w.name: w for w in _W}


def get(name):
    return WORKLOADS[name]


if __name__ == "__main__":
    for w in _W:
        try:
            b = w.backend_info()
            nc = w.node_cache()
        except Exception as e:                       # missing backend files
            b, nc = {"error": repr(e)}, None
        print(f"{w.name:22s} {w.group:11s} prep {w.prep:17s} stream {w.stream:6s} backend {w.backend_dir.name:20s} "
              f"target {w.target} L {b.get('L')} W {b.get('W')} recall {b.get('recall', 0):.3f} node cache {nc} "
              f"budgets {w.budgets}")
