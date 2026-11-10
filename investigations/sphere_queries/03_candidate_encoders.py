"""Part A step 4: encode Hermes passages with candidate encoders, compare to sphere-10m rows.

usage: [ROW_OFFSET=k] 03_candidate_encoders.py PARQUET N CANDIDATE [CANDIDATE ...]
rows k..k+N-1 are used
candidate: kind:model:fmt:max_len[:pool], kind in {st, dpr_ctx}, fmt in
{raw, pair, pair_raw, title_text}, pool fp16 casts the DPR model to half precision, optional _tt0 suffix = token_type_ids all zero, pool in {cls, mean}. model is a hub id or a local dir.
"""

import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer, DPRContextEncoder

from common import cosine_rows, relative_l2

BATCH = 16


def load_rows(path: str, n: int) -> tuple[np.ndarray, list[str], list[str]]:
    table = pq.ParquetFile(path).read(columns=["title", "raw", "vector"])
    table = table.slice(int(os.environ.get("ROW_OFFSET", "0")), n)
    vec = np.array(table["vector"].to_pylist(), dtype=np.float64)
    return vec, table["title"].to_pylist(), table["raw"].to_pylist()


def strip_title(title: str, raw: str) -> str:
    return raw[len(title):].lstrip() if raw.startswith(title) else raw


def make_inputs(fmt: str, titles: list[str], raws: list[str]):
    if fmt == "raw":
        return raws, None
    if fmt == "pair":
        return titles, [strip_title(t, r) for t, r in zip(titles, raws)]
    if fmt == "pair_raw":
        return titles, raws
    if fmt == "title_text":
        return [f"{t}. {strip_title(t, r)}" for t, r in zip(titles, raws)], None
    raise ValueError(fmt)


def encode(kind: str, model_id: str, fmt: str, max_len: int, pool: str,
           titles: list[str], raws: list[str]) -> np.ndarray:
    tok_id = "bert-base-uncased" if kind == "dpr_ctx" else model_id
    tok = AutoTokenizer.from_pretrained(tok_id)
    if kind == "dpr_ctx":
        model = DPRContextEncoder.from_pretrained(model_id).eval()
        if pool == "fp16":
            model = model.half()
    else:
        model = AutoModel.from_pretrained(model_id).eval()
    zero_types = fmt.endswith("_tt0")
    first, second = make_inputs(fmt.removesuffix("_tt0"), titles, raws)
    order = np.argsort([len(a) for a in first])
    out = np.zeros((len(raws), 768), dtype=np.float32)
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        args = [[first[i] for i in idx]]
        if second is not None:
            args.append([second[i] for i in idx])
        batch = tok(*args, padding=True, truncation="longest_first",
                    max_length=max_len, return_tensors="pt")
        if zero_types:
            batch["token_type_ids"] = torch.zeros_like(batch["token_type_ids"])
        with torch.no_grad():
            if kind == "dpr_ctx":
                vec = model(**batch).pooler_output.float()
            else:
                hidden = model(**batch).last_hidden_state
                if pool == "cls":
                    vec = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vec = (hidden * mask).sum(1) / mask.sum(1)
        out[idx] = vec.numpy()
    return out


def main() -> None:
    torch.set_num_threads(8)
    path, n = sys.argv[1], int(sys.argv[2])
    vec, titles, raws = load_rows(path, n)
    print(f"rows {n}, torch threads {torch.get_num_threads()}")
    for spec in sys.argv[3:]:
        parts = spec.split(":")
        kind, model_id, fmt, max_len = parts[0], parts[1], parts[2], int(parts[3])
        pool = parts[4] if len(parts) > 4 else "cls"
        t0 = time.time()
        got = encode(kind, model_id, fmt, max_len, pool, titles, raws)
        cos = cosine_rows(got, vec)
        rel = relative_l2(got, vec)
        pct = np.percentile(cos, [5, 50, 95])
        words = np.array([len(r.split()) for r in raws])
        short, long_ = cos[words <= np.median(words)], cos[words > np.median(words)]
        print(f"  median cos short/long half: {np.median(short):.4f}/{np.median(long_):.4f};"
              f" min {cos.min():.4f}")
        print(f"{spec}: cos p5/med/p95 {pct[0]:.4f}/{pct[1]:.4f}/{pct[2]:.4f} "
              f"relL2 med {np.median(rel):.3f} norm {np.linalg.norm(got, axis=1).mean():.2f} "
              f"({time.time() - t0:.0f}s, {n / (time.time() - t0):.1f} rows/s)", flush=True)


if __name__ == "__main__":
    main()
