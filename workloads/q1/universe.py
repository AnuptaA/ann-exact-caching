"""Universe of query groups: 10K groups of 50 interpolated members per dataset.

Request ids: members [0, G*S), base queries [G*S, G*S+G), then the official
pool for no-locality streams.
"""

from dataclasses import dataclass

import numpy as np

N_GROUPS = 10_000
GROUP_SIZE = 50
LAMBDA_MAX = 0.5


@dataclass(frozen=True)
class Universe:
    base_idx: np.ndarray   # (G,) indices into the official query set
    partner_idx: np.ndarray  # (G, S) indices into the official query set
    lam: np.ndarray        # (G, S) interpolation weights
    n_official: int

    @property
    def n_groups(self) -> int:
        return len(self.base_idx)

    @property
    def group_size(self) -> int:
        return self.partner_idx.shape[1]

    @property
    def n_members(self) -> int:
        return self.n_groups * self.group_size

    def member_id(self, group: int, member: int) -> int:
        return group * self.group_size + member

    def base_id(self, group: int) -> int:
        return self.n_members + group

    def pool_id(self, index: int) -> int:
        return self.n_members + self.n_groups + index

    @property
    def n_ids(self) -> int:
        return self.n_members + self.n_groups + self.n_official


def build_universe(
    n_official: int,
    n_groups: int = N_GROUPS,
    group_size: int = GROUP_SIZE,
    seed: int = 0,
) -> Universe:
    if n_groups > n_official:
        raise ValueError(f"{n_groups} groups from only {n_official} official queries")
    if n_official < 2:
        raise ValueError("need at least 2 official queries to draw partners")
    rng = np.random.default_rng([seed, 1])
    base_idx = rng.choice(n_official, size=n_groups, replace=False)
    # partner uniform over the official set minus the base itself
    draw = rng.integers(0, n_official - 1, size=(n_groups, group_size))
    partner_idx = draw + (draw >= base_idx[:, None])
    lam = rng.uniform(0.0, LAMBDA_MAX, size=(n_groups, group_size)).astype(np.float32)
    return Universe(base_idx, partner_idx.astype(np.int64), lam, n_official)


def member_vectors(uni: Universe, official: np.ndarray, groups: slice) -> np.ndarray:
    """Float32 member vectors of the selected groups, shape (groups*S, d)."""
    official = np.asarray(official)
    base = np.asarray(official[uni.base_idx[groups]], dtype=np.float32)
    partner = np.asarray(official[uni.partner_idx[groups].ravel()], dtype=np.float32)
    lam = uni.lam[groups].reshape(-1, 1)
    base_rep = np.repeat(base, uni.group_size, axis=0)
    return ((1.0 - lam) * base_rep + lam * partner).astype(np.float32)


def request_vectors(uni: Universe, official: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Vectors for arbitrary request ids (members, bases, official pool)."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) and (ids.min() < 0 or ids.max() >= uni.n_ids):
        raise ValueError("request id out of range")
    out = np.empty((len(ids), official.shape[1]), dtype=np.float32)
    is_member = ids < uni.n_members
    is_base = (ids >= uni.n_members) & (ids < uni.n_members + uni.n_groups)
    is_pool = ~(is_member | is_base)
    if is_member.any():
        m = ids[is_member]
        g, j = m // uni.group_size, m % uni.group_size
        lam = uni.lam[g, j].reshape(-1, 1)
        base = np.asarray(official[uni.base_idx[g]], dtype=np.float32)
        partner = np.asarray(official[uni.partner_idx[g, j]], dtype=np.float32)
        out[is_member] = (1.0 - lam) * base + lam * partner
    if is_base.any():
        out[is_base] = official[uni.base_idx[ids[is_base] - uni.n_members]]
    if is_pool.any():
        out[is_pool] = official[ids[is_pool] - uni.n_members - uni.n_groups]
    return out
