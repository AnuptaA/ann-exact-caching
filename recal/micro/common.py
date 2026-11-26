"""Result files of the micro-experiments: settings, results, environment and code commit, plus an optional CSV."""

import csv
import hashlib
import json
from pathlib import Path

from recal.measure import environment, jobs_util

RESULTS_ROOT = Path("results/micro")
BUDGETS_DEFAULT = (0.01, 0.05, 0.1)


def config_hash(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]


def out_dir(experiment: str, workload: str, root: Path = RESULTS_ROOT) -> Path:
    return Path(root) / experiment / workload


def result_path(directory: Path, experiment: str) -> Path:
    return Path(directory) / f"{experiment.lower()}.json"


def is_fresh(directory: Path, experiment: str, settings: dict) -> bool:
    """A result exists with the same config hash (settings and code commit are in the hash)."""
    path = result_path(directory, experiment)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("config_hash") == config_hash(settings)
    except ValueError:
        return False


def write_result(directory: Path, experiment: str, settings: dict, results, rows: list[dict] | None = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = {"experiment": experiment, "settings": settings, "config_hash": config_hash(settings),
            "results": results, "environment": environment.describe(), "code_commit": jobs_util.code_commit()}
    path = result_path(directory, experiment)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=1, default=_plain))
    tmp.replace(path)
    if rows:
        write_csv(path.with_suffix(".csv"), rows)
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)
