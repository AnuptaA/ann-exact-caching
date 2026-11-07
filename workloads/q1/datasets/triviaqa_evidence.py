"""Evidence log for the TriviaQA encoder choice: answer-passage lift.

For each question encoder, retrieve the top-20 passages by Euclidean distance and
measure how often they contain an answer alias, relative to the rate at which a
random passage does. A log only; it does not change the encoder.
"""

import numpy as np

TOP = 20
MIN_ALIAS_LEN = 4


def answer_lift(question_vecs, passage_vecs, passage_texts, answers, top: int = TOP,
                n_base: int = 2000, seed: int = 0) -> dict:
    """top-`top` precision, lift over the random-passage rate, and any-hit share."""
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
        d = np.einsum("ij,ij->i", q, q)[:, None] + p_sq[None, :] - 2.0 * (q @ pv.T)
        top_ids = np.argpartition(d, top - 1, axis=1)[:, :top]
        for row, ids in enumerate(top_ids):
            al = answers[start + row]
            hits.append([any(a in texts[j] for a in al) for j in ids])
    hits = np.array(hits)
    precision = float(hits.mean())
    return {"questions": len(keep), "precision": precision, "base_rate": base,
            "lift": precision / base if base > 0 else float("inf"),
            "any_hit": float(hits.any(axis=1).mean())}


def format_evidence(results: dict) -> str:
    lines = ["# TriviaQA encoder evidence (informational)", "",
             "| encoder | questions | top-20 precision | lift | any hit |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['questions']} | {r['precision']:.3f} | "
                     f"{r['lift']:.1f}x | {r['any_hit']:.3f} |")
    lines += ["", f"random-passage rate: {next(iter(results.values()))['base_rate']:.4f}"]
    return "\n".join(lines) + "\n"


def run_evidence(sphere_parquet: str, questions_parquet: str, encoders: dict, out_path: str,
                 n_questions: int = 300) -> dict:
    """Encode questions with each encoder, score against sphere passages, write the log.

    encoders maps a name to a callable list[str] -> (n, 768) array. Run it in the
    isolated environment; sphere_parquet is a Hermes SPHERE_100K file (vector, raw)
    and questions_parquet a TriviaQA rc.nocontext split (question, answer).
    """
    import pandas as pd
    passages = pd.read_parquet(sphere_parquet, columns=["vector", "raw"])
    pv = np.stack(passages["vector"].to_numpy()).astype(np.float32)
    qa = pd.read_parquet(questions_parquet, columns=["question", "answer"]).head(n_questions * 3)
    answers = [list(a["aliases"]) for a in qa["answer"]]
    keep = [i for i, al in enumerate(answers) if any(len(a) >= MIN_ALIAS_LEN for a in al)][:n_questions]
    questions = [qa["question"].iloc[i] for i in keep]
    results = {}
    for name, encode in encoders.items():
        vecs = np.vstack([encode(questions[i:i + 64]) for i in range(0, len(questions), 64)])
        results[name] = answer_lift(vecs, pv, passages["raw"].tolist(), [answers[i] for i in keep])
    with open(out_path, "w") as f:
        f.write(format_evidence(results))
    return results
