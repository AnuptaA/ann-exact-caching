"""Part C: CPU question-encoding throughput for bge-base-en-v1.5 and the DPR_web question encoder.

usage: 09_encoder_throughput.py PARTB_DIR DPR_WEB_QUESTION_DIR [N_TEXTS]
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, DPRQuestionEncoder

partb, dpr_dir = Path(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
torch.set_num_threads(8)
texts = (partb / "triviaqa_head.txt").read_text().splitlines()[:n]
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
models = {"bge-base-en-v1.5 raw CLS": (bge_tok, AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").eval(), "cls"),
          "dpr_web question encoder": (tok, DPRQuestionEncoder.from_pretrained(dpr_dir).eval(), "pooler")}
for name, (tokenizer, model, pool) in models.items():
    for batch_size in (64,):
        for start in range(0, 64, 64):
            batch = tokenizer(texts[:64], padding=True, truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                model(**batch)
        t0 = time.time()
        for start in range(0, n, batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=64, return_tensors="pt")
            with torch.no_grad():
                out = model(**batch)
                _ = out.pooler_output if pool == "pooler" else out.last_hidden_state[:, 0]
        rate = n / (time.time() - t0)
        print(f"{name}: {rate:.0f} texts/s (batch {batch_size}, {n} texts, 8 threads); "
              f"96,005 questions ~ {96005 / rate / 60:.1f} min")
