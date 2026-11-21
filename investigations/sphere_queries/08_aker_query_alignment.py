"""Check whether Aker triviaqa-sim stream row i corresponds to triviaqa.txt line i.

usage: 08_aker_query_alignment.py PARTB_DIR [N_LINES]
Encodes the first N lines with bge-base-en-v1.5 raw CLS and compares to the stream rows.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

partb = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
torch.set_num_threads(8)
lines = (partb / "triviaqa_head.txt").read_text().splitlines()[:n]
stream = np.load(partb / "aker_trivia_0.6.npy").astype(np.float32)
tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").eval()
batch = tok(lines, padding=True, truncation=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    mine = model(**batch).last_hidden_state[:, 0].numpy()


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


sims = unit(mine) @ unit(stream).T
best = sims.argmax(1)
print("diagonal cosine median:", float(np.median(np.diag(sims))))
print("best-match cosine median over all rows:", float(np.median(sims.max(1))))
print("best row == line index:", float(np.mean(best == np.arange(n))))
print("best rows (first 15):", best[:15].tolist())
print("best cos min:", float(sims.max(1).min()))
