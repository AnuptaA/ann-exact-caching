"""The ReCal serving loop (Q1, Q2, H2, H9, H10): mini cache, main cache, audits, history and recalibration."""

from dataclasses import dataclass, field
from pathlib import Path

import time

import numpy as np

from recal import build as builder
from recal import policy
from recal.core import MAIN_HIT, MINI_HIT, MISS_K_MAX, Core
from recal.history import AUDITED, HIT, MISS, HistoryLog
from recal.params import Params
from recal.signals import Signals
from recal.updates import Updater


@dataclass
class Response:
    ids: np.ndarray
    dists: np.ndarray
    source: str                    # "mini", "main" or "backend"
    score: float | None = None
    path: int = 0
    background: list = field(default_factory=list)   # backend work caused by the request: (kind, depth)
    audited: bool = False
    cache_ms: float = 0.0              # cache-path time of the request (hash, mini lookup, probe, score)


class ReCal:
    def __init__(self, backend, params: Params, M_bytes: float, workdir: str | Path, signals=None):
        self.backend, self.p, self.M = backend, params, float(M_bytes)
        self.rng = np.random.default_rng(params.seed)
        width = params.k_max
        self.log = HistoryLog(Path(workdir) / "log", backend.dim, params.H, width)
        self.t = 0                      # main-path requests so far (R1)
        self.requests = 0
        self.constructions = 0
        self.built: builder.Built | None = None
        self.core = self._bare_core()
        self.lam: float | None = None
        self.window: policy.CalibrationWindow | None = None
        self.status = policy.Status(int(params.drop_window * params.H))
        self.certified = policy.Certified(None)
        self.h0 = 0.2
        self.last_construction = 0      # main-path request counter at the last construction
        self.counts = {"mini_hits": 0, "main_hits": 0, "misses": 0, "audits": 0, "k_max_misses": 0}
        self.updater = Updater(self)
        self.signals = Signals(params, self.updater) if signals is None else signals
        self.rebuild_requested: str | None = None

    def _bare_core(self) -> Core:
        core = Core(self.backend.dim, self.p.k_max)
        core.set_budget(self.M)
        core.finish_build()
        entry = 40.0 + 8.0 * self.p.K
        core.mini_set_capacity(max(self.p.mini_floor, int(self.p.mini_cap * self.M // entry)))   # B3: at its cap
        return core

    # ---- serving ---------------------------------------------------------------------------------

    def serve(self, q: np.ndarray, K: int) -> Response:
        q = np.ascontiguousarray(q, dtype=np.float32)
        self.requests += 1
        self.log.append_hash(self.core.hash_vector(q))
        m = 0 if self.p.idealized else (self.built.m if self.built else 0)
        t0 = time.perf_counter()
        res = self.core.lookup(q, K, self.lam, m, True)
        cache_ms = (time.perf_counter() - t0) * 1e3
        if res.path == MINI_HIT:                      # Q2: touches nothing else
            self.counts["mini_hits"] += 1
            return Response(res.ids, res.dists, "mini", path=res.path, cache_ms=cache_ms)
        if res.path == MISS_K_MAX:
            self.counts["k_max_misses"] += 1
            ids, dists = self.backend.search(q, K, kind="miss")
            return Response(ids, dists, "backend", path=res.path, background=[("miss", K)], cache_ms=cache_ms)
        self.t += 1
        self.window and self.window.expire(self.t)
        response = self._main_path(q, K, res)
        response.cache_ms = cache_ms
        self.status.record(self.lam)
        self.signals.after_request(self, response.score if response.score is not None and res.has_score else None)
        if self.rebuild_requested:
            self.rebuild_requested = None
            self.construct()
        self._maybe_construct()
        return response

    def _main_path(self, q, K, res) -> Response:
        if res.path == MAIN_HIT:
            self.counts["main_hits"] += 1
            if self.rng.random() < self.p.a:
                return self._audit(q, K, res)
            self.log.append(q, K, HIT, version=self.backend.version)
            return Response(res.ids, res.dists, "main", float(res.score), res.path)
        self.counts["misses"] += 1
        ids, dists = self.backend.search(q, K, kind="miss")
        self.log.append(q, K, MISS, ids, dists, self.backend.version)
        self.core.mini_put(q, ids, dists)
        if res.has_score and len(res.ids) == K and self.window is not None:
            y = float(np.isin(ids, res.ids).mean())
            self._label(res.score, y, 1.0)
        return Response(ids, dists, "backend", float(res.score) if res.has_score else None, res.path,
                        [("miss", K)])

    def _audit(self, q, K, res) -> Response:
        self.counts["audits"] += 1
        ids, dists = self.backend.search(q, K, kind="audit")
        self.log.append(q, K, AUDITED, ids, dists, self.backend.version)
        self.core.mini_put(q, ids, dists)
        y = float(np.isin(ids, res.ids).mean())
        self._label(res.score, y, 1.0 / self.p.a)
        return Response(res.ids, res.dists, "main", float(res.score), res.path, [("audit", K)], audited=True)

    def _label(self, score: float, y: float, weight: float) -> None:
        """H2, H9: a labeled query enters the window and lambda is recomputed."""
        if self.window is None:
            return
        self.window.add(self.t, score, y, weight)
        self.certified = self.window.certify()
        self.lam = self.certified.lam

    # ---- construction ----------------------------------------------------------------------------

    def _maybe_construct(self) -> None:
        if self.built is None and len(self.log) >= self.p.cold_start:
            self.construct()

    def construct(self) -> builder.Built:
        """B4 to B12: label, build, then swap everything at once."""
        snap = self.log.snapshot()
        labels = builder.fetch_labels(snap, self.log, self.backend, self.backend.version)
        old = self.built
        built = builder.build(
            snap, self.backend, self.p, self.M, np.random.default_rng(self.p.seed + 1 + self.constructions),
            self.t, None if old is None else old.centers, None if old is None else old.center_K,
            self.h0, self.log.hash_snapshot())
        built.info["labels"] = labels
        # B12: the new core takes over the mini cache at the size R, then everything is swapped
        built.core.mini_set_capacity(built.R)
        for h, ids, dists in self.core.mini_dump():
            built.core.mini_put_hash(h, ids, dists)
        self.core, self.built = built.core, built
        self.window, self.lam, self.certified, self.h0 = built.window, built.lam, built.certified, max(built.h0, 1e-3)
        self.status = policy.Status(int(self.p.drop_window * self.p.H))
        self.status.margin_unreachable = built.margin_unreachable
        self.constructions += 1
        self.last_construction = self.t
        self.signals.after_construction(self)
        return built

    # ---- reports ---------------------------------------------------------------------------------

    def status_report(self) -> dict:
        """H10, H11: lambda, why it is undefined, the audited recall of the window."""
        cert = self.certified
        report = self.status.report(self.lam, cert.n_labeled, cert.ess, self.p.min_ess)
        if self.window is not None:
            recall, n = self.window.audited_recall()
            report.update({"audited_recall": recall, "audited_in_window": n, "target": self.p.T})
        return report

    def bytes_by_structure(self) -> dict:
        """Bytes per structure; inserts consume the headroom, so the total does not grow past M."""
        b = dict(self.core.bytes_by_structure())
        if self.built is not None:
            grown = self.updater._structure_bytes() - self.updater.base_bytes
            b["headroom"] = max(0.0, self.p.headroom * self.M - grown)
            b["total"] = sum(v for k, v in b.items() if k not in ("total", "budget")) + max(0.0, grown - self.p.headroom * self.M)
        return b

    # ---- data updates ----------------------------------------------------------------------------

    def delete(self, ids) -> int:
        return self.updater.delete(ids)

    def insert(self, ids, vectors) -> None:
        self.updater.insert(ids, vectors)
