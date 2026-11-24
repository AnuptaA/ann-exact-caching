#!/usr/bin/env python3
"""Push, pull and list objects of the artifact and result store (RECAL_STORE, see recal/store.py).

  python -m scripts.store push <local file or directory> <key or prefix>
  python -m scripts.store pull <key or prefix> <local path>
  python -m scripts.store ls [prefix]
"""

import argparse
import json
from pathlib import Path

from recal import store


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, help="s3://bucket/prefix or a directory (default: RECAL_STORE)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    push = sub.add_parser("push")
    push.add_argument("local", type=Path)
    push.add_argument("key")
    pull = sub.add_parser("pull")
    pull.add_argument("key")
    pull.add_argument("local", type=Path)
    ls = sub.add_parser("ls")
    ls.add_argument("prefix", nargs="?", default="")
    a = ap.parse_args(argv)
    s = store.open_store(a.store)
    if a.cmd == "push":
        n = s.push_tree(a.local, a.key) if a.local.is_dir() else int(s.push_file(a.key, a.local))
        print(f"pushed {n} object(s)")
    elif a.cmd == "pull":
        keys = s.keys(a.key)
        if keys == [a.key]:
            n = int(s.pull_file(a.key, a.local))
        else:
            n = s.pull_prefix(a.key, a.local)
        print(f"pulled {n} object(s)")
    else:
        for row in s.ls(a.prefix):
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
