"""Evidence log for the TriviaQA encoder choice: answer-passage lift.

For each question encoder, retrieve the top-20 passages by Euclidean distance and
measure how often they contain an answer alias, relative to the rate at which a
random passage does. A log only; it does not change the encoder.
"""

from pathlib import Path

import numpy as np

TOP = 20
MIN_ALIAS_LEN = 4


def answer_lift(question_vecs, passage_vecs, passage_texts, answers, top: int = TOP,
                n_base: int = 2000, seed: int = 0, metric: str = "l2") -> dict:
    """top-`top` precision, lift over the random-passage rate, and any-hit share.

    metric: "l2" (squared Euclidean distance, smaller is closer) or "ip" (inner product,
    larger is closer; the metric DPR was trained with).
    """
    if metric not in ("l2", "ip"):
        raise ValueError(f"metric must be 'l2' or 'ip', got {metric!r}")
    texts = [t.lower() for t in passage_texts]
    answers = [[a.lower() for a in al if len(a) >= MIN_ALIAS_LEN] for al in answers]
    keep = [i for i, al in enumerate(answers) if al]
    if not keep:
        raise ValueError("no question has a usable answer alias")
    qv = np.asarray(question_vecs, dtype=np.float32)[keep]
    answers = [answers[i] for i in keep]
    pv = np.asarray(passage_vecs, dtype=np.float32)
    p_sq = np.einsum("ij,ij->i", pv, pv)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(texts), min(n_base, len(texts)), replace=False)
    base = float(np.mean([np.mean([any(a in texts[j] for a in al) for j in sample])
                          for al in answers]))
    hits = []
    for start in range(0, len(qv), 256):
        q = qv[start:start + 256]
        d = (np.einsum("ij,ij->i", q, q)[:, None] + p_sq[None, :] - 2.0 * (q @ pv.T)
             if metric == "l2" else -(q @ pv.T))
        top_ids = np.argpartition(d, top - 1, axis=1)[:, :top]
        for row, ids in enumerate(top_ids):
            al = answers[start + row]
            hits.append([any(a in texts[j] for a in al) for j in ids])
    hits = np.array(hits)
    precision = float(hits.mean())
    return {"questions": len(keep), "precision": precision, "base_rate": base,
            "lift": precision / base if base > 0 else float("inf"),
            "any_hit": float(hits.any(axis=1).mean())}


def format_evidence(results: dict, meta: dict | None = None) -> str:
    """results: encoder name -> lift dict, or metric name -> (encoder name -> lift dict)."""
    nested = isinstance(next(iter(results.values())), dict) and "precision" not in next(iter(results.values()))
    tables = results if nested else {"": results}
    lines = ["# TriviaQA encoder evidence (informational)", ""]
    lines += [f"- {k}: {v}" for k, v in (meta or {}).items()]
    for metric, table in tables.items():
        lines += ["", f"## {metric}" if metric else "", "| encoder | questions | top-20 precision | lift | any hit |",
                  "|---|---|---|---|---|"]
        for name, r in table.items():
            lines.append(f"| {name} | {r['questions']} | {r['precision']:.3f} | "
                         f"{r['lift']:.1f}x | {r['any_hit']:.3f} |")
        lines += ["", f"random-passage rate: {next(iter(table.values()))['base_rate']:.4f}"]
    return "\n".join(lines) + "\n"


def hub_cls_encoder(name: str, max_length: int = 64):
    """Raw CLS embedding (not normalized) of a Hugging Face encoder; needs the network."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()

    def encode(texts):
        batch = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            return model(**batch).last_hidden_state[:, 0].numpy()
    return encode


def hub_dpr_encoder(name: str, max_length: int = 64):
    """A public DPR question encoder from the hub; needs the network."""
    import torch
    from transformers import AutoTokenizer, DPRQuestionEncoder
    tok, model = AutoTokenizer.from_pretrained("bert-base-uncased"), DPRQuestionEncoder.from_pretrained(name).eval()

    def encode(texts):
        batch = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            return model(**batch).pooler_output.numpy()
    return encode


def run_evidence(sphere_parquet: str, questions_parquet: str, encoders: dict, out_path: str,
                 n_questions: int = 300) -> dict:
    """Encode questions with each encoder, score against sphere passages, write the log.

    encoders maps a name to a callable list[str] -> (n, 768) array. Run it in the
    isolated environment; sphere_parquet is a Hermes SPHERE_100K file (vector, raw)
    and questions_parquet a TriviaQA rc.nocontext split (question, answer).
    Both Euclidean and inner product are reported, on the raw (unnormalized) vectors.
    """
    import pandas as pd
    passages = pd.read_parquet(sphere_parquet, columns=["vector", "raw"])
    pv = np.stack(passages["vector"].to_numpy()).astype(np.float32)
    qa = pd.read_parquet(questions_parquet, columns=["question", "answer"]).head(n_questions * 3)
    answers = [list(a["aliases"]) for a in qa["answer"]]
    keep = [i for i, al in enumerate(answers) if any(len(a) >= MIN_ALIAS_LEN for a in al)][:n_questions]
    questions = [qa["question"].iloc[i] for i in keep]
    results = {"Euclidean distance on raw vectors (the Q1 metric)": {},
               "Inner product on raw vectors (DPR's training metric)": {}}
    for name, encode in encoders.items():
        vecs = np.vstack([encode(questions[i:i + 64]) for i in range(0, len(questions), 64)])
        for label, metric in zip(results, ("l2", "ip")):
            results[label][name] = answer_lift(vecs, pv, passages["raw"].tolist(),
                                               [answers[i] for i in keep], metric=metric)
    meta = {"passages": f"{len(pv)} Hermes SPHERE_100K passages ({Path(sphere_parquet).name})",
            "questions": f"{len(keep)} TriviaQA questions with answer aliases ({Path(questions_parquet).name})",
            "vectors": "raw, unnormalized", "top": TOP,
            "lift": "top-20 answer-passage precision divided by the rate for random passages"}
    with open(out_path, "w") as f:
        f.write(format_evidence(results, meta))
    return results
