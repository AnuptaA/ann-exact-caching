"""Job lines (one shell command each, skipping itself when its result exists) for the Q3 experiments."""

from recal.system_benefit.split import SPLITS

CLIENTS = (1, 2, 4, 8, 16, 32, 64)
PY = ".venv-core/bin/python"
ANN = ".venv-ann/bin/python"
REPLAY = "simulations/approximate_hits/throughput_replay.py"
PROBE = "simulations/approximate_hits/latency_probe.py"


def pct(s: float) -> str:
    return f"{round(100 * s):d}"


def split_cell_lines(prep: str, stream: str, total_fraction: float, name: str, splits=SPLITS,
                     extra: str = "") -> list[str]:
    """ReCal cells at s x B with per-request timing, for each split s > 0."""
    out = []
    for s in splits:
        if s == 0:
            continue
        d = f"simulations/approximate_hits/results/q3/{name}/recal_s{pct(s)}"
        out.append(f"[ -f {d}/recal.json ] || {{ mkdir -p {d} && PYTHONPATH=. {PY} -u -m recal.cells --method recal "
                   f"--prep {prep} --stream {stream} --budget {s * total_fraction:.6g} --history-frac 0.3 --timing "
                   f"{extra} --out {d} > {d}/log.txt 2>&1; }}  # recal.cells")
    return out


def throughput_lines(workload: str, name: str, arms: dict, node_cache_nodes: dict, clients=CLIENTS,
                     extra: str = "") -> list[str]:
    """Closed-loop replay per arm. arms: label -> 'none' or a timed run json; node_cache_nodes: label -> nodes."""
    out = []
    for label, run in arms.items():
        d = f"simulations/approximate_hits/results/q3/{name}/throughput_{label}"
        spec = "none" if run == "none" else f"{label}={run}"
        nodes = node_cache_nodes.get(label)
        node_flag = f"--nodes {int(nodes)} " if nodes is not None else ""
        out.append(f"[ -f {d}/replay.json ] || {{ mkdir -p {d} && PYTHONPATH=. {ANN} -u {REPLAY} --workload "
                   f"{workload} --arm {spec} --clients {' '.join(map(str, clients))} {node_flag}--io-complete {extra} "
                   f"--out {d}/replay.json > {d}/log.txt 2>&1; }}  # throughput_replay.py")
    return out


def single_client_lines(workload: str, name: str, arms: dict, extra: str = "") -> list[str]:
    """Single-client composed latency for the appendix: the same replay at one client."""
    return throughput_lines(workload, f"{name}_single", arms, {}, clients=(1,), extra=extra)


def depth_table_lines(prep: str, index_dir: str, name: str, depths: list[int], search_list: int = 200) -> list[str]:
    d = f"simulations/approximate_hits/results/q3/{name}/depth_latency"
    return [f"[ -f {d}/diskann.json ] || {{ mkdir -p {d} && PYTHONPATH=. {ANN} -u {PROBE} --prep {prep} --index "
            f"{index_dir} --depths {' '.join(map(str, depths))} --search-list {search_list} --out {d}/diskann.json "
            f"> {d}/log.txt 2>&1; }}  # latency_probe.py"]


def node_cache_curve_lines(index_dir: str, queries: str, name: str, nodes: list[int], node_bytes: float,
                           k: int = 10, search_list: int = 128) -> list[str]:
    d = f"simulations/approximate_hits/results/q3/{name}/node_cache_curve"
    return [f"[ -f {d}/curve.json ] || {{ mkdir -p {d} && PYTHONPATH=. {ANN} -u scripts/node_cache_curve.py "
            f"--index-dir {index_dir} --queries {queries} --nodes {' '.join(map(str, nodes))} --node-bytes "
            f"{node_bytes:g} --k {k} --search-list {search_list} --out {d}/curve.json > {d}/log.txt 2>&1; }}"]
