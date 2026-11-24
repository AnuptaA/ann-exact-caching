"""Identity of an experiment cell: the hash of the run its job line starts.

Lines of recal.cells are parsed with the cell parser, defaults resolved, and hashed with the options that do not
change the run left out; other lines hash their arguments without the output path. The code commit is replaced by
a fixed marker so the hash does not change between commits.

Left out of the hash: out (where the result goes), tag (a label), and the keys jobs_util never hashes for the same
reason (log, log_level, verbose, progress, threads, cores, taskset, quiet*, timing_noise*), plus records (an extra
output file). Everything else counts: system, workload, stream, budget, mode, backend, T, seed, history, options.
"""

import hashlib
import re
import shlex
from pathlib import Path

from recal.measure import jobs_util

CODE_MARKER = "matrix"
EXTRA_EXCLUDED = ("records",)
SUBSTITUTION = re.compile(r"\$\(([^)]*)\)")
SHELL_STOPS = {"|", "||", "&&", ";", "&"}
REPO = str(Path(__file__).resolve().parents[1])


def command_tokens(line: str) -> tuple[list[str], list[str]]:
    """The python command of a job line (after the interpreter) and the shell substitutions it contains."""
    substitutions: list[str] = []

    def keep(match):
        substitutions.append(match.group(1))
        return f"__SUBST{len(substitutions) - 1}__"

    tokens = shlex.split(SUBSTITUTION.sub(keep, line.split("# ")[0]).replace("{", " ").replace("}", " "))
    start = next((i for i, t in enumerate(tokens) if re.search(r"python[\d.]*$", t)), None)
    if start is None:
        raise ValueError(f"no python command in: {line[:80]}")
    command = []
    for tok in tokens[start + 1:]:
        if tok in SHELL_STOPS or re.match(r"^\d*>", tok):
            break
        command.append(tok)
    return command, substitutions


def parse_cells(argv: list[str], substitutions: list[str]):
    from recal import cells
    resolved = [("0" if re.fullmatch(r"__SUBST\d+__", t) else t) for t in argv]
    ns = cells.make_parser().parse_args(resolved)
    values = {k: v for k, v in vars(ns).items() if k not in EXTRA_EXCLUDED}
    values["_substitutions"] = substitutions
    return values


def cell_hash(line: str) -> str:
    command, substitutions = command_tokens(line)
    if "-m" in command and command[command.index("-m") + 1:command.index("-m") + 2] == ["recal.cells"]:
        values = parse_cells(command[command.index("-m") + 2:], substitutions)
        return jobs_util.resolved_hash(values, CODE_MARKER)
    kept, skip = [], False
    for tok in command:
        if skip:
            skip = False
        elif tok == "--out":
            skip = True
        else:
            kept.append(tok)
    # Job lines carry absolute paths; hash them relative to the repository so clones agree.
    text = " ".join(kept + substitutions).replace(REPO, ".")
    return hashlib.sha256(text.encode()).hexdigest()


def out_dir(line: str) -> str:
    found = re.search(r"--out (\S+)", line)
    if found:
        return found.group(1)
    found = re.search(r"mkdir -p (\S+)", line)
    return found.group(1) if found else ""
