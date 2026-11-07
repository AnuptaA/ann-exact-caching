"""Command line: python -m workloads.q1 <command> ..."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from workloads.q1 import groundtruth, stats, streams, universe
from workloads.q1.datasets import esci, sift10m, sphere, spacev, triviaqa

logger = logging.getLogger(__name__)


def _fetch(args):
    out = Path(args.out_dir)
    {"sift10m": sift10m.fetch, "spacev": spacev.fetch, "sphere": sphere.fetch}[args.dataset](out)


def _esci(args):
    print(json.dumps(esci.build(Path(args.raw_dir), Path(args.out_dir))))


def _triviaqa_questions(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    from workloads.q1.datasets import common
    for split, url in triviaqa.SPLIT_URLS.items():
        paths[split] = common.download_file(url, out / f"triviaqa_{split}.parquet")
    questions, report = triviaqa.load_questions(paths)
    np.save(out / "triviaqa_questions.npy", np.array(questions, dtype=str))
    print(json.dumps(report))


def _triviaqa_encode(args):
    questions = np.load(args.questions, allow_pickle=False).tolist()
    encode = triviaqa.dpr_query_encoder(Path(args.encoder_dir), device=args.device)
    np.save(args.out, triviaqa.encode_questions(questions, encode))


def _universe(args):
    official = np.load(args.official, mmap_mode="r")
    uni = universe.build_universe(len(official), args.n_groups, args.group_size, args.seed)
    np.savez(args.out, base_idx=uni.base_idx, partner_idx=uni.partner_idx, lam=uni.lam,
             n_official=uni.n_official)


def load_universe(path) -> universe.Universe:
    z = np.load(path)
    return universe.Universe(z["base_idx"], z["partner_idx"], z["lam"], int(z["n_official"]))


def _streams(args):
    uni = load_universe(args.universe)
    out = {}
    for s in args.zipf:
        out[f"zipf_{s:.2f}"] = streams.zipf_stream(uni, args.n_requests, s, args.seed)
    out["uniform"] = streams.uniform_stream(uni, args.n_requests, args.seed)
    if args.no_locality:
        out["no_locality"] = streams.no_locality_stream(uni, args.seed)
    np.savez(args.out, history_fraction=args.history_fraction, **out)
    for name, ids in out.items():
        s = float(name.split("_")[1]) if name.startswith("zipf") else None
        print(stats.format_stats(name, stats.stream_stats(uni, ids, s)))
        print(f"  history {streams.history_split(len(ids), args.history_fraction)} requests")


def _groundtruth(args):
    base = np.load(args.base, mmap_mode="r")
    queries = np.load(args.queries, mmap_mode="r")
    groundtruth.build_ground_truth(queries, base, Path(args.out_dir), args.k, args.device)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m workloads.q1")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download a dataset (sift10m, spacev, sphere)")
    p.add_argument("dataset", choices=["sift10m", "spacev", "sphere"])
    p.add_argument("--out-dir", required=True)
    p.set_defaults(run=_fetch)

    p = sub.add_parser("esci", help="embed all ESCI products and distinct queries")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(run=_esci)

    p = sub.add_parser("triviaqa-questions", help="download and de-duplicate TriviaQA questions")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(run=_triviaqa_questions)

    p = sub.add_parser("triviaqa-encode", help="encode questions with the DPR query encoder")
    p.add_argument("--questions", required=True)
    p.add_argument("--encoder-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(run=_triviaqa_encode)

    p = sub.add_parser("universe", help="10K groups of 50 members from the official queries")
    p.add_argument("--official", required=True, help=".npy of the official query vectors")
    p.add_argument("--out", required=True)
    p.add_argument("--n-groups", type=int, default=universe.N_GROUPS)
    p.add_argument("--group-size", type=int, default=universe.GROUP_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=_universe)

    p = sub.add_parser("streams", help="zipf, uniform and optional no-locality streams")
    p.add_argument("--universe", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--zipf", type=float, nargs="+", default=list(streams.ZIPF_EXPONENTS))
    p.add_argument("--n-requests", type=int, default=streams.N_REQUESTS)
    p.add_argument("--history-fraction", type=float, default=streams.HISTORY_FRACTION)
    p.add_argument("--no-locality", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=_streams)

    p = sub.add_parser("groundtruth", help="exact top-k ground truth (resumable)")
    p.add_argument("--base", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--k", type=int, default=groundtruth.TOP_K)
    p.add_argument("--device", default="numpy", help="numpy, cuda or mps")
    p.set_defaults(run=_groundtruth)
    return ap


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.run(args)
