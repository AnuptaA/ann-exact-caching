"""ReCal's two modes and the backend each one runs on.

deployable: labels are the backend's answers (DiskANN, or the live pgvector of the update runs) and m entries are
probed per query. idealized: labels are the exact top-K and every entry is read, so it runs on the exact backend.
"""

BACKEND_KIND = {"exact": "exact", "stored": "exact", "file": "diskann", "diskann": "diskann", "pgvector": "pgvector"}
ALLOWED = {"deployable": ("diskann", "pgvector"), "idealized": ("exact",)}


def mode_of(idealized: bool) -> str:
    return "idealized" if idealized else "deployable"


def backend_kind(name: str) -> str:
    return BACKEND_KIND[name]


def recal_backend(mode: str) -> str:
    """The backend name a generated ReCal cell uses for its mode."""
    return {"deployable": "diskann", "idealized": "exact"}[mode]


def check_mode_backend(mode: str, backend: str) -> None:
    if mode not in ALLOWED:
        raise ValueError(f"unknown ReCal mode {mode}")
    if backend_kind(backend) not in ALLOWED[mode]:
        raise ValueError(f"{mode} ReCal runs on {' or '.join(ALLOWED[mode])}, not on {backend_kind(backend)}")
