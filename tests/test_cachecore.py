"""Unit tests of the C++ cache core (doc 9ah) against a NumPy reference of the same definitions.
Run with .venv-core: .venv-core/bin/python -m pytest tests/test_cachecore.py"""
import numpy as np
import pytest

cc = pytest.importorskip("simulations.approximate_hits.cachecore")

D, K, N_DB = 16, 10, 3000
EPS = 1e-12


def make_world(seed=0):
    rng = np.random.default_rng(seed)
    db = rng.normal(size=(N_DB, D)).astype(np.float32)
    q = rng.normal(size=(400, D)).astype(np.float32)
    return rng, db, q


def ref_entries(db, q, rng, n_entries=40, ball=30, shell=60):
    """Entries as the simulator builds them: center = a query, ball = its `ball` nearest database points,
    radius = distance to the last, shell = the next `shell` points (all sampled here)."""
    ents = []
    for c in rng.choice(len(q), n_entries, replace=False):
        d2 = ((db - q[c]) ** 2).sum(axis=1)
        order = np.argsort(d2)
        b, s = order[:ball], order[ball:ball + shell]
        ents.append((int(c), b, s, float(np.sqrt(d2[b[-1]]))))
    return ents


def ref_lookup(x, db, q, ents, model, lam, shell_balls):
    """NumPy reference of the hit decision (FastHitPath with exact kNN, shell over shell_balls entries)."""
    mu, sd, coef, b = model
    cached = np.unique(np.concatenate([e[1] for e in ents]))
    d2 = ((db[cached] - x) ** 2).sum(axis=1)
    part = np.argsort(d2)[:K]
    nr = float(np.sqrt(max(d2[part].max(), 0.0))) + EPS
    cen = np.array([e[0] for e in ents])
    rad = np.array([e[3] for e in ents])
    de = np.sqrt(((q[cen] - x) ** 2).sum(axis=1))
    fit = (nr + de) / rad
    order = np.argsort(fit, kind="stable")
    f = np.full(7, 10.0)
    for j in range(min(3, len(ents))):
        e = order[j]
        f[2 * j], f[2 * j + 1] = de[e] / rad[e], nr / rad[e]
    sel = order[:shell_balls] if shell_balls > 0 else np.arange(len(ents))
    samp = np.unique(np.concatenate([ents[e][2] for e in sel]))
    samp = samp[~np.isin(samp, cached)]
    cnt = int((((db[samp] - x) ** 2).sum(axis=1) < nr * nr).sum())
    f[6] = np.log1p(cnt)
    s = float(((f - mu) / sd) @ coef + b)
    return s >= lam, s, cached[part], cnt


def build(db, q, ents, shell_balls, model, ef_cap=1 << 12):
    core = cc.Core(D, K, shell_balls, capacity=ef_cap)
    core.set_model(*[list(map(float, m)) for m in model[:3]], float(model[3]))
    for c, b, s, r in ents:
        core.add_entry(c, q[c], r, b.astype(np.int64), db[b], s.astype(np.int64), db[s])
    return core


@pytest.fixture
def world():
    rng, db, q = make_world()
    ents = ref_entries(db, q, rng)
    model = (rng.normal(size=7), np.abs(rng.normal(size=7)) + 0.5, rng.normal(size=7), 0.3)
    return rng, db, q, ents, model


@pytest.mark.parametrize("shell_balls", [0, 3, 10])
def test_exact_lookup_matches_reference(world, shell_balls):
    rng, db, q, ents, model = world
    core = build(db, q, ents, shell_balls, model)
    for i in range(60):
        x = q[i]
        h, s, near, cnt = ref_lookup(x, db, q, ents, model, 0.0, shell_balls)
        r = core.lookup(x, 0.0, 0)
        assert r.intruders == cnt
        assert abs(r.score - s) < 1e-4
        assert r.hit == h
        assert sorted(r.near) == sorted(near.tolist())


def test_counts_and_add_remove_roundtrip(world):
    rng, db, q, ents, model = world
    core = build(db, q, ents, 10, model)
    cached = np.unique(np.concatenate([e[1] for e in ents]))
    samp = np.unique(np.concatenate([e[2] for e in ents]))
    assert core.n_cached() == len(cached)
    assert core.n_samples() == len(np.setdiff1d(samp, cached))
    before = [core.lookup(q[i], 0.0, 0) for i in range(30)]
    c, b, s, r = ref_entries(db, q, np.random.default_rng(9), n_entries=1)[0]
    if not core.has_entry(c):
        core.add_entry(c, q[c], r, b.astype(np.int64), db[b], s.astype(np.int64), db[s])
        core.remove_entry(c)
    after = [core.lookup(q[i], 0.0, 0) for i in range(30)]
    assert core.n_cached() == len(cached)
    assert core.n_samples() == len(np.setdiff1d(samp, cached))
    for x, y in zip(before, after):
        assert x.score == pytest.approx(y.score) and x.intruders == y.intruders and sorted(x.near) == sorted(y.near)


def test_remove_all_entries_empties(world):
    rng, db, q, ents, model = world
    core = build(db, q, ents, 10, model)
    for c, *_ in ents:
        core.remove_entry(c)
    assert core.n_cached() == 0 and core.n_samples() == 0 and core.n_entries() == 0
    assert not core.lookup(q[0], -1e9, 0).hit


def test_graph_matches_exact_at_high_ef_and_after_rebuild(world):
    rng, db, q, ents, model = world
    core = build(db, q, ents, 10, model)
    for e in ents[::3]:                      # evictions leave deleted graph nodes
        core.remove_entry(e[0])
    kept = ents[1::3] + ents[2::3]
    agree = 0
    for i in range(60):
        h, s, near, cnt = ref_lookup(q[i], db, q, kept, model, 0.0, 10)
        r = core.lookup(q[i], 0.0, 512)
        agree += sorted(r.near) == sorted(near.tolist())
    assert agree >= 58
    core.rebuild_index(4)
    assert core.graph_deleted() == 0 and core.graph_elements() == core.n_cached()
    for i in range(30):
        h, s, near, cnt = ref_lookup(q[i], db, q, kept, model, 0.0, 10)
        assert abs(core.lookup(q[i], 0.0, 0).score - s) < 1e-4


def test_batch_equals_sequential(world):
    rng, db, q, ents, model = world
    for sb in (0, 10):
        core = build(db, q, ents, sb, model)
        core.set_threads(4)
        hit, score, near, nr, feat = core.lookup_batch(q[:100], 0.0, 0)
        core.set_threads(1)
        for i in range(100):
            r = core.lookup(q[i], 0.0, 0)
            assert r.score == pytest.approx(score[i]) and r.hit == hit[i] and sorted(r.near) == sorted(near[i].tolist())


def test_lru_order_and_mark_used(world):
    rng, db, q, ents, model = world
    core = build(db, q, ents, 10, model)
    for t, (c, b, *_rest) in enumerate(ents):
        core.set_used(c, float(t))
    c0, b0 = ents[0][0], ents[0][1]
    core.mark_used(b0[:1].astype(np.int64), 1000.0)
    order = core.lru_order()
    assert order[-1] == c0 or core.last_used(c0) == 1000.0
    assert core.last_used(c0) == 1000.0


class RefQV:
    """NumPy reference of QVCache (online_baselines.QVCache logic, per-request inserts, exact search)."""

    def __init__(self, db, k, n_mini, cap, Dm, alpha, mu, P, lo, hi, nb):
        self.db, self.k, self.n, self.cap, self.Dm, self.alpha = db, k, n_mini, cap, Dm, alpha
        self.mu, self.P, self.lo, self.hi, self.nb = mu, P, lo, hi, nb
        self.mini = [[] for _ in range(n_mini)]
        self.used = [0.0] * n_mini
        self.owner, self.theta, self.fill = {}, {}, 0

    def region(self, x):
        z = (x.astype(np.float64) - self.mu) @ self.P
        b = np.floor((z - self.lo) / np.maximum(self.hi - self.lo, 1e-12) * self.nb)
        return tuple(np.clip(b, 0, self.nb - 1).astype(int))

    def lookup(self, x, t):
        ids = np.array([i for m in self.mini for i in m], dtype=np.int64)
        if len(ids) < self.k:
            return False, ids
        d2 = ((self.db[ids] - x) ** 2).sum(axis=1)
        o = np.argsort(d2, kind="stable")[:self.k]
        dk = float(np.sqrt(d2[o[-1]]))
        th = self.theta.get(self.region(x))
        hit = th is not None and dk <= (1 + self.Dm) * th
        if hit:
            for v in ids[o].tolist():
                self.used[self.owner[v]] = t
        return hit, ids[o]

    def on_miss(self, x, ans, t):
        dk = float(np.sqrt(((self.db[ans[-1]] - x) ** 2).sum()))
        key = self.region(x)
        th = self.theta.get(key)
        self.theta[key] = dk if th is None else (1 - self.alpha) * th + self.alpha * dk
        new = [v for v in ans.tolist() if v not in self.owner]
        if len(self.mini[self.fill]) + len(new) > self.cap:
            empty = [i for i in range(self.n) if not self.mini[i]]
            v = empty[0] if empty else min(range(self.n), key=lambda i: self.used[i])
            for i in self.mini[v]:
                if self.owner.get(i) == v:
                    del self.owner[i]
            self.mini[v], self.fill = [], v
        for v in new:
            self.owner[v] = self.fill
            self.mini[self.fill].append(v)
        self.used[self.fill] = t


def test_qvcache_exact_matches_reference():
    rng, db, q = make_world(3)
    stream = np.concatenate([rng.choice(len(q), 600), rng.choice(60, 600)])   # some repetition
    w = q[stream[:200]].astype(np.float64)
    mu = w.mean(axis=0)
    _, _, vt = np.linalg.svd(w - mu, full_matrices=False)
    P = vt[:4].T
    z = (w - mu) @ P
    lo, hi = z.min(axis=0), z.max(axis=0)
    args = (K, 4, 120, 0.15, 0.9, mu, P, lo, hi, 4)
    ref = RefQV(db, *args)
    core = cc.QVCore(D, K, 4, 120, 0.15, 0.9, mu, np.ascontiguousarray(P), lo, hi, 4, False)
    n_hit = 0
    for t, qi in enumerate(stream.tolist()):
        x = q[qi]
        h_r, near_r = ref.lookup(x, float(t))
        h_c, near_c, _ = core.lookup(x, float(t))
        assert h_r == h_c and sorted(near_r.tolist()) == sorted(near_c)
        n_hit += h_c
        if not h_c:
            ans = np.argsort(((db - x) ** 2).sum(axis=1))[:K].astype(np.int64)
            ref.on_miss(x, ans, float(t))
            core.on_miss(x, ans, db[ans], float(t))
        assert core.n_cached() == len(ref.owner)
    assert n_hit > 20


def quantize4(x, lo, hi):
    step = np.maximum(hi - lo, 1e-12) / 15
    return (lo + np.round((np.clip(x, lo, hi) - lo) / step) * step).astype(np.float32)


@pytest.mark.parametrize("dim", [16, 17])
def test_packed_4bit_codes_match_float_samples(dim):
    rng = np.random.default_rng(5)
    db = rng.normal(size=(N_DB, dim)).astype(np.float32)
    q = rng.normal(size=(300, dim)).astype(np.float32)
    lo, hi = db.min(axis=0).astype(np.float64), db.max(axis=0).astype(np.float64)
    dbq = quantize4(db, lo, hi)
    ents = []
    for c in rng.choice(len(q), 30, replace=False):
        d2 = ((db - q[c]) ** 2).sum(axis=1)
        o = np.argsort(d2)
        ents.append((int(c), o[:25], o[25:120], float(np.sqrt(d2[o[24]]))))
    model = (rng.normal(size=7), np.abs(rng.normal(size=7)) + 0.5, rng.normal(size=7), 0.1)
    cores = []
    for packed in (False, True):
        core = cc.Core(dim, K, 10, capacity=4096)
        core.set_model(*[list(map(float, m)) for m in model[:3]], float(model[3]))
        if packed:
            core.set_quantizer(list(lo), list(hi), 4)
        for c, b, s, r in ents:
            core.add_entry(c, q[c], r, b.astype(np.int64), db[b], s.astype(np.int64), dbq[s])
        cores.append(core)
    for i in range(200):
        a, b = cores[0].lookup(q[i], 0.0, 0), cores[1].lookup(q[i], 0.0, 0)
        assert a.intruders == b.intruders and abs(a.score - b.score) < 1e-6


def test_ball_mode_1_full_m_equals_scan_and_mode_2_nearest(world):
    rng, db, q, ents, model = world
    scan = build(db, q, ents, 10, model)
    m1 = build(db, q, ents, 10, model)
    m1.set_ball_mode(1, len(ents), 4 * len(ents))
    m2 = build(db, q, ents, 10, model)
    m2.set_ball_mode(2, 10, 200)
    cen = np.array([e[0] for e in ents]); rad = np.array([e[3] for e in ents])
    for i in range(60):
        a, b = scan.lookup(q[i], 0.0, 0), m1.lookup(q[i], 0.0, 0)
        assert a.intruders == b.intruders and abs(a.score - b.score) < 1e-6
        # mode 2: features from the 3 nearest centers
        r = m2.lookup(q[i], 0.0, 0)
        de = np.sqrt(((q[cen] - q[i]) ** 2).sum(axis=1))
        o = np.argsort(de, kind="stable")[:3]
        assert np.allclose(r.features[0:6:2], de[o] / rad[o], atol=1e-5)
