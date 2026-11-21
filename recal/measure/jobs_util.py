"""Helpers for job runners: atomic result writes, config hash, freshness, retry class.

CLI (used by the shell runners):
  python -m recal.measure.jobs_util check <json> <jobline> [--trust-existing]
      exit 0 fresh, 1 missing, 2 stale / partial / unreadable

Hash scheme 2: the hash covers every argument after defaults are applied (minus the timing-noise keys) and the
code commit. A job script registers its parsed arguments with register_args(); `<script> <args>
--print-config-hash` prints the hash the script would store, which is how a job line is checked without a run.
A result without a scheme 2 hash is stale (--trust-existing restores the existence check).
  python -m recal.measure.jobs_util classify <exit code> <logfile>
      prints ok, retry or fatal
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

EXCLUDED_KEYS = {
    "out", "tag", "log", "log_file", "logfile", "log_level", "verbose", "progress",
    "threads", "n_threads", "cores", "cpus", "taskset", "wait_quiet",
}
EXCLUDED_PREFIXES = ("quiet", "timing_noise")
JOB_MARKERS = ("online_sim.py", "hit_table", "recal.cells", "recal.sim")
SCRIPT_MARKERS = (
    "online_sim.py", "eval_aker_real.py", "recal.cells", "recal.sim",
)
SHELL_STOPS = {"|", "||", "&&", ";", "&"}
TRANSIENT_CODES = {137, -9}
TRANSIENT_MESSAGES = ("machine not quiet",)

FRESH, MISSING, STALE = 0, 1, 2
HASH_SCHEME = 2
HASH_FLAG = "--print-config-hash"
_registered: dict = {}


def atomic_write_json(path: Any, obj: Any, **dump_kwargs: Any) -> None:
    """Write JSON through a temp file in the same directory, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, **dump_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _is_excluded(key: str) -> bool:
    return key in EXCLUDED_KEYS or key.startswith(EXCLUDED_PREFIXES)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(f"{value:.12g}")
    try:
        return float(f"{float(value):.12g}")
    except (TypeError, ValueError):
        return str(value)


def argv_to_dict(argv: list) -> dict:
    """Flag -> value dict from command-line tokens ('--x v', '--x=v', '--flag')."""
    result: dict = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--") or len(tok) == 2:
            i += 1
            continue
        key, eq, value = tok[2:].partition("=")
        key = key.replace("-", "_")
        if eq:
            result[key] = value
        elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            result[key] = argv[i + 1]
            i += 1
        else:
            result[key] = True
        i += 1
    return result


def normalize_args(args: dict) -> dict:
    return {
        k.replace("-", "_"): _normalize_value(v)
        for k, v in args.items()
        if not _is_excluded(k.replace("-", "_"))
    }


def config_hash(args: dict) -> str:
    text = json.dumps(normalize_args(args), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def code_commit(cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd or Path(__file__).parent,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def resolved_hash(resolved: dict, commit: Optional[str] = None) -> str:
    """Hash of the resolved settings plus the code commit (scheme 2)."""
    args = {k: v for k, v in resolved.items() if k != "print_config_hash"}
    text = json.dumps(
        {"args": normalize_args(args), "code_commit": commit or code_commit()}, sort_keys=True, default=str
    )
    return hashlib.sha256(text.encode()).hexdigest()


def add_hash_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(HASH_FLAG, action="store_true", help="print the config hash and exit")


def register_args(args: argparse.Namespace) -> None:
    """Call right after parsing and after defaults are resolved: remember the settings, honour the hash flag."""
    resolved = dict(vars(args))
    if resolved.get("print_config_hash"):
        print(resolved_hash(resolved))
        raise SystemExit(0)
    _registered.clear()
    _registered.update(resolved)


def stamp_result(result: dict, argv: Optional[list] = None) -> dict:
    """Add config_hash and code_commit: scheme 2 when the script registered its arguments."""
    if _registered:
        result["config_hash"] = resolved_hash(_registered)
        result["hash_scheme"] = HASH_SCHEME
    else:
        argv = sys.argv[1:] if argv is None else argv
        result["config_hash"] = config_hash(argv_to_dict(argv))
        result["hash_scheme"] = 1
    result["code_commit"] = code_commit()
    return result


def jobline_args(line: str) -> dict:
    """Argument dict of the first known script invocation in a job line."""
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()
    start = None
    for i, tok in enumerate(tokens):
        if any(tok.endswith(m) for m in SCRIPT_MARKERS):
            start = i + 1
            break
    if start is None:
        return {}
    args = []
    for tok in tokens[start:]:
        if tok in SHELL_STOPS or re.match(r"^\d*>", tok):
            break
        args.append(tok)
    return argv_to_dict(args)


def jobline_json(line: str) -> Optional[str]:
    m = re.search(r"\[ -f (\S+\.json) \]", line)
    return m.group(1) if m else None


def is_job_line(line: str) -> bool:
    return any(m in line for m in JOB_MARKERS)


def jobline_hash(line: str, repo: Optional[Path] = None) -> Optional[str]:
    """Hash the script of a job line would store, from `<interpreter> ... --print-config-hash`."""
    try:
        tokens = shlex.split(line)
    except ValueError:
        return None
    marker = next((i for i, t in enumerate(tokens) if any(t.endswith(m) for m in SCRIPT_MARKERS)), None)
    if marker is None:
        return None
    start = next((i for i in range(marker, -1, -1) if re.search(r"python[\d.]*$", tokens[i])), None)
    if start is None:
        return None
    cmd = []
    for tok in tokens[start:]:
        if tok in SHELL_STOPS or re.match(r"^\d*>", tok):
            break
        cmd.append(tok)
    repo = repo or Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        out = subprocess.run(cmd + [HASH_FLAG], cwd=repo, env=env, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.stdout.split()
    return lines[-1] if out.returncode == 0 and lines else None


def check_result(path: Any, line: str, strict_hash: bool = False,
                 trust_existing: bool = False, hasher: Any = None) -> int:
    """FRESH (0), MISSING (1) or STALE (2) for the result of a job line.

    strict_hash is accepted for old callers: a result without a scheme 2 hash is always stale now.
    """
    path = Path(path)
    if not path.is_file():
        return MISSING
    if trust_existing:
        return FRESH
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return STALE
    if not isinstance(data, dict):
        return STALE
    stored = data.get("config_hash")
    if stored is None or data.get("hash_scheme") != HASH_SCHEME:
        return STALE
    expected = (hasher or jobline_hash)(line)
    return FRESH if expected is not None and stored == expected else STALE


def classify_exit(code: int, log_text: str) -> str:
    """ok, retry (known transient failure) or fatal."""
    if code == 0:
        return "ok"
    if "Traceback" in log_text:
        return "fatal"
    if any(msg in log_text for msg in TRANSIENT_MESSAGES):
        return "retry"
    if code in TRANSIENT_CODES:
        return "retry"
    return "fatal"


def _read_log(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    chk = sub.add_parser("check")
    chk.add_argument("json")
    chk.add_argument("jobline")
    chk.add_argument("--strict-hash", action="store_true")
    chk.add_argument("--trust-existing", action="store_true")
    cls = sub.add_parser("classify")
    cls.add_argument("code", type=int)
    cls.add_argument("logfile")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return check_result(a.json, a.jobline, a.strict_hash, a.trust_existing)
    print(classify_exit(a.code, _read_log(a.logfile)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
