"""Part B: query/data mismatch metrics on stratified slices.

usage: 07_partb_metrics.py PARTB_DIR HERMES_PARQUET QUESTION_ENCODER_DIR OUT_JSON [N_QUERIES]
Needs the scratch venv (torch, transformers, pyarrow). Repo code is imported read-only
via PYTHONPATH.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer, DPRQuestionEncoder

import metrics
from common import AKER, npy_header, read_npy_rows
from workloads.q1.datasets.triviaqa_evidence import MIN_ALIAS_LEN, answer_lift

partb, hermes, q_enc_dir, out_json = (Path(a) for a in sys.argv[1:5])
n_queries = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
torch.set_num_threads(8)
T0 = time.time()
RESULTS: dict = {"n_queries": n_queries}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def evenly(total: int, n: int) -> np.ndarray:
    return np.unique(np.linspace(0, total - 1, n).astype(np.int64))


def dedupe(vectors: np.ndarray, rows: np.ndarray) -> np.ndarray:
    _, first = np.unique(np.asarray(vectors[rows]), axis=0, return_index=True)
    return rows[np.sort(first)]


def encode_dpr(texts: list[str], max_length: int = 64, name: str | None = None) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = DPRQuestionEncoder.from_pretrained(name or str(q_enc_dir)).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 768), dtype=np.float32)
    for s in range(0, len(texts), 64):
        idx = order[s:s + 64]
        batch = tok([texts[i] for i in idx], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            out[idx] = model(**batch).pooler_output.numpy()
    return out


def encode_bge(texts: list[str], max_length: int = 64) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").eval()
    out = []
    for s in range(0, len(texts), 64):
        batch = tok(texts[s:s + 64], padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt")
        with torch.no_grad():
            out.append(model(**batch).last_hidden_state[:, 0].numpy())
    return np.vstack(out)


def evaluate(name: str, queries: np.ndarray, base: np.ndarray) -> dict:
    ids, kth, med = metrics.topk_and_median(queries, base)
    row = {"queries": len(queries), "base_rows": len(base),
           "relative_contrast": metrics.relative_contrast(kth, med),
           "kth_nn_dist_median": float(np.median(kth)), "median_dist": float(np.median(med)),
           "nn_cosine_median": metrics.nn_cosine(queries, base, ids),
           **{f"hub_{k}": v for k, v in metrics.hubness(ids, len(base)).items()},
           **metrics.norm_stats(queries, base)}
    RESULTS[name] = row
    log(f"{name}: RC {row['relative_contrast']:.3f} hub skew {row['hub_skew']:.2f} "
        f"top100 {row['hub_top_share']:.3f} nn_cos {row['nn_cosine_median']:.3f}")
    return row


# --- slices
sphere = np.load(partb / "sphere_slice.npy")
spacev = np.load(partb / "spacev_slice.npy")
log(f"slices: sphere {sphere.shape}, spacev {spacev.shape}")

# --- Aker TriviaQA queries and text
lines = (partb / "triviaqa_head.txt").read_text().splitlines()[:100_000]
aker_q = np.load(partb / "aker_trivia_0.6.npy", mmap_mode="r")
RESULTS["aker_trivia_check"] = {"lines": len(lines), "npy": list(aker_q.shape), "dtype": str(aker_q.dtype)}
rows = evenly(len(lines), n_queries * 2)
rows = dedupe(aker_q, rows)[:n_queries]
aker_vecs = np.asarray(aker_q[rows], dtype=np.float32)
texts = [lines[i] for i in rows]
log(f"aker queries {aker_vecs.shape}")

# --- matched encoder queries
SKIP_METRICS = os.environ.get("SKIP_METRICS") == "1"
if not SKIP_METRICS:
    t = time.time()
    dpr_vecs = encode_dpr(texts)
    RESULTS["dpr_web_question_throughput_per_s"] = len(texts) / (time.time() - t)
    log(f"dpr_web question encode {len(texts)} in {time.time() - t:.0f}s")

# --- held-out in-space passages (Hermes rows not in the base slice)
table = pq.ParquetFile(hermes).read(columns=["raw", "vector"])
herm_vec = np.asarray(table["vector"].combine_chunks().values, dtype=np.float32).reshape(-1, 768)
herm_raw = table["raw"].to_pylist()
held = herm_vec[10_000:10_000 + n_queries]

# --- SPACEV queries
spv_hdr = npy_header(AKER + "/spacev-10m.npy")
spv_stream = np.load(partb / "aker_spacev_0.6.npy")
sp_rows = dedupe(spv_stream, evenly(len(spv_stream), n_queries * 2))[:n_queries]
sp_queries = spv_stream[sp_rows].astype(np.float32)
sp_held = read_npy_rows(AKER + "/spacev-10m.npy", 10_000, n_queries, spv_hdr["shape"][1],
                        spv_hdr["dtype"], spv_hdr["header"])

if not SKIP_METRICS:
    evaluate("sphere / aker_trivia_bge", aker_vecs, sphere)
    evaluate("sphere / dpr_web_matched", dpr_vecs, sphere)
    evaluate("sphere / held_out_passages", held, sphere)
    evaluate("spacev / aker_stream", sp_queries, spacev)
    evaluate("spacev / held_out_vectors", sp_held, spacev)

# --- answer lift against the 50K Hermes passages (original TriviaQA questions)
qa = pd.read_parquet(partb / "tqa_validation.parquet", columns=["question", "answer"])
chosen, answers = [], []
for question, answer in zip(qa["question"], qa["answer"]):
    aliases = list(answer["aliases"])
    if any(len(a) >= MIN_ALIAS_LEN for a in aliases):
        chosen.append(question)
        answers.append(aliases)
    if len(chosen) == 300:
        break
RESULTS["lift"] = {"questions": len(chosen), "passages": len(herm_raw)}
# Aker's text lines are generated paraphrases, so they carry no answer aliases
check = slice(0, 200)
bge_check = encode_bge(texts[check])
cos = (aker_vecs[check] * bge_check).sum(1) / (
    np.linalg.norm(aker_vecs[check], axis=1) * np.linalg.norm(bge_check, axis=1))
RESULTS["aker_vs_bge_cosine"] = {"lines": 200, "median": float(np.median(cos)), "min": float(cos.min())}
log(f"aker vs bge-base-en-v1.5 cosine median {np.median(cos):.5f} min {cos.min():.5f}")
encoders = {
    "bge-base-en-v1.5 (raw CLS)": encode_bge(chosen),
    "dpr_web question encoder": encode_dpr(chosen),
    "dpr single-nq question encoder": encode_dpr(
        chosen, name="facebook/dpr-question_encoder-single-nq-base"),
}
for name, vecs in encoders.items():
    for metric in ("l2", "ip"):
        res = answer_lift(vecs, herm_vec, herm_raw, answers, metric=metric)
        RESULTS["lift"][f"{name} | {metric}"] = res
        log(f"lift {name} {metric}: precision {res['precision']:.4f} lift {res['lift']:.1f}x "
            f"any {res['any_hit']:.3f} (q={res['questions']})")

# --- released stream skew by exact vector identity
RESULTS["stream_skew"] = {"triviaqa 0.6": metrics.stream_skew(np.asarray(aker_q))}
for skew_name in ("0.3", "0.6", "0.99"):
    RESULTS["stream_skew"][f"spacev {skew_name}"] = metrics.stream_skew(
        np.load(partb / f"aker_spacev_{skew_name}.npy"))
log("stream skew done")
out_json.write_text(json.dumps(RESULTS, indent=1))
log("done")
