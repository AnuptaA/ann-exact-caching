"""Data updates: eager deletes with a sweep, lazy batched inserts with headroom and displacement.

Entries are never inserted or evicted individually. Shells and the mini cache are not touched.
"""

import numpy as np


class Updater:
    def __init__(self, system):
        self.s = system
        self.buffer_ids: list = []
        self.buffer_vecs: list = []
        self.frozen = 0              # inserts refused since the last construction
        self.admitted = 0
        self.ignored = 0
        self.base_bytes = 0.0        # structure bytes at the last construction

    def after_construction(self) -> None:
        self.frozen = self.admitted = self.ignored = 0
        self.base_bytes = self._structure_bytes()

    def _structure_bytes(self) -> float:
        b = self.s.core.bytes_by_structure()
        return float(b["data_pool"] + b["reverse_map"] + b["entries"] + b["shell_pool"] + b["center_index"])

    def headroom_left(self) -> float:
        """2% of M minus what inserts have added since the construction."""
        grown = self._structure_bytes() - self.base_bytes
        return max(0.0, self.s.p.headroom * self.s.M - grown)

    # ---- updates --------------------------------------------------------------------------------------

    def delete(self, ids) -> int:
        """Delete from the database and from the cache (eager); sweep at 5% dead slots."""
        ids = np.asarray(ids, dtype=np.int64)
        self.s.backend.delete(ids)
        hit = sum(1 for i in ids.tolist() if self.s.core.delete_id(int(i)))
        core = self.s.core
        total = core.n_data_slots() + core.n_dead_slots()
        if total and core.n_dead_slots() >= self.s.p.sweep * total:
            core.sweep()
        return hit

    # ---- updates --------------------------------------------------------------------------------

    def insert(self, ids, vectors) -> None:
        """Buffer inserts into the database; process them in batches of max(100, 0.1% N)."""
        ids = np.asarray(ids, dtype=np.int64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.s.backend.insert(ids, vectors)
        self.buffer_ids.extend(ids.tolist())
        self.buffer_vecs.extend(list(vectors))
        batch = max(self.s.p.insert_batch_min, int(self.s.p.insert_batch_frac * len(self.s.backend)))
        while len(self.buffer_ids) >= batch:
            self.process(batch)

    def process(self, n: int | None = None) -> dict:
        n = len(self.buffer_ids) if n is None else n
        if n == 0:
            return {"admitted": 0, "frozen": 0, "ignored": 0}
        ids = np.array(self.buffer_ids[:n], dtype=np.int64)
        vecs = np.stack(self.buffer_vecs[:n])
        del self.buffer_ids[:n], self.buffer_vecs[:n]
        admitted, frozen, ignored, _ = self.s.core.insert_batch(
            ids, vecs, self.headroom_left(), self.s.p.insert_rounds, self.s.p.insert_centers)
        self.admitted += admitted
        self.frozen += frozen
        self.ignored += ignored
        return {"admitted": admitted, "frozen": frozen, "ignored": ignored}
