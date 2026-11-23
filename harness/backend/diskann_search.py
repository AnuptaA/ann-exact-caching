"""DiskANN search backend for answer tables (ids equal base row positions)."""

import numpy as np


class DiskannBackend:
    name = "diskann"

    def __init__(self, index, dtype, beam_width: int = 2):
        """index: an opened disk index with search(vector, k_neighbors, complexity, beam_width)."""
        self.index, self.dtype, self.beam_width = index, np.dtype(dtype), beam_width

    @classmethod
    def open(cls, index_dir: str, dtype, threads: int = 8, prefix: str = "ann") -> "DiskannBackend":
        from simulations.approximate_hits.diskann_aio import open_disk_index
        index = open_disk_index(max_wait=1800, index_directory=str(index_dir), num_threads=threads,
                                num_nodes_to_cache=0, cache_mechanism=0, index_prefix=prefix)
        return cls(index, dtype)

    def search(self, queries: np.ndarray, list_size: int, ef_search: int) -> np.ndarray:
        """ef_search is DiskANN's search list size (complexity); it must be at least list_size."""
        if ef_search < list_size:
            raise ValueError(f"search list {ef_search} is shorter than the {list_size} answers asked")
        out = np.full((len(queries), list_size), -1, dtype=np.int64)
        for i, q in enumerate(np.asarray(queries).astype(self.dtype)):
            ids, _ = self.index.search(q, k_neighbors=list_size, complexity=ef_search,
                                       beam_width=self.beam_width)
            ids = np.asarray(ids, dtype=np.int64)
            out[i, :len(ids)] = ids
        return out
