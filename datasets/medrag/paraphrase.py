#!/usr/bin/env python3
"""
Rewordings for MedRAG-Zipf (context doc 9y), the local-LLM stand-in for Proximity's GPT-4o step:
"each time a given query appears, we rephrase it ... syntactically different but semantically
equivalent ... each generated version of the question is unique across the entire dataset".

Needs: datasets/medrag/zipf_draw.json (build_workload.py draw) and a llama.cpp server on loopback
(llama-server --host 127.0.0.1, Qwen2.5-7B-Instruct Q8_0; /tmp/mx/medrag.sh starts it).
For every question with c occurrences, c accepted rewordings (every occurrence is a rewording;
the original wording is never used):
  1. generate: sampled (temperature 1, top-p 0.95) with one of a few wording hints (all asking to
     keep the meaning exactly); once a question has many accepted versions, a sample of them is
     shown with "write one different from these" (the most frequent question needs ~700);
  2. unique: normalized text (lowercase, alphanumerics only) differs from the original and from
     every version accepted for any question;
  3. verify: the same model, greedy, judges whether the rewording asks exactly the same thing
     (same correct yes / no / maybe answer); only "Yes" is accepted (Proximity instead checked that
     its RAG system answered every rewording the same).
Output datasets/medrag/paraphrases.json {question index: [rewordings]} (saved after every round,
resumable) and paraphrase_stats.json (generated / duplicate / rejected counts).

  .venv/bin/python datasets/medrag/paraphrase.py --server http://127.0.0.1:18080
"""

import re
import json
import time
import random
import logging
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
D = Path("datasets/medrag")
HINTS = ["Use different wording.", "Change the sentence structure.", "Use synonyms where they are exact.",
         "Reorder the parts of the question.", "Make it slightly more concise.",
         "Make it slightly more formal.", "Phrase it in plain language, keeping every medical term's meaning.",
         "Start the question with a different word.", "Phrase it as a clinician asking a colleague.",
         "Phrase it as a researcher asking about the evidence."]
GEN = ("Rephrase the medical question below so that it is worded differently but asks exactly the same "
       "thing (the correct yes / no / maybe answer must not change). {hint} Reply with the rephrased "
       "question only.{avoid}\n\nQuestion: {q}")
VER = ("Question A: {a}\nQuestion B: {b}\n\nDo A and B ask exactly the same thing, so that the correct "
       "yes / no / maybe answer to A is always the correct answer to B? Reply with only Yes or No.")


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def chat(server, prompt, temperature, max_tokens, seed):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                       "top_p": 0.95, "max_tokens": max_tokens, "seed": seed, "cache_prompt": True}).encode()
    for attempt in range(100):
        try:
            req = urllib.request.Request(server + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:                       # server busy / restarting / suspended: retry
            if attempt % 10 == 9:
                logger.warning(f"request failed {attempt + 1}x: {e!r}")
            time.sleep(min(300, 5 * (attempt + 1)))
    raise RuntimeError("server unreachable")


def clean(text, q):
    t = text.strip().split("\n")[0].strip().strip('"').strip("“”").strip()
    t = re.sub(r"^(rephrased question|question|rephrased)\s*:\s*", "", t, flags=re.I).strip().strip('"')
    if not t or len(t) > 3 * len(q) + 40 or len(t) < 8:
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    draw = json.loads((D / "zipf_draw.json").read_text())
    qs, counts = draw["questions"], draw["counts"]
    out_f, stats_f = D / "paraphrases.json", D / "paraphrase_stats.json"
    acc = {int(k): v for k, v in json.loads(out_f.read_text()).items()} if out_f.exists() else {}
    stats = json.loads(stats_f.read_text()) if stats_f.exists() else {"generated": 0, "bad": 0, "duplicate": 0,
                                                                     "rejected": 0, "accepted": 0}
    for i in range(len(qs)):
        acc.setdefault(i, [])
    seen = {norm(q) for q in qs} | {norm(p) for v in acc.values() for p in v}
    rng = random.Random(args.seed + sum(len(v) for v in acc.values()))
    pool = ThreadPoolExecutor(args.workers)
    for rnd in range(args.rounds):
        need = {i: counts[i] - len(acc[i]) for i in range(len(qs)) if counts[i] > len(acc[i])}
        if not need:
            break
        logger.info(f"round {rnd}: {len(need)} questions need {sum(need.values())} rewordings")
        tasks = []
        for i, n in need.items():
            for _ in range(max(2, int(1.5 * n) + 1)):
                avoid = ""
                if len(acc[i]) >= 20:
                    ex = rng.sample(acc[i], min(15, len(acc[i])))
                    avoid = " It must be worded differently from all of these earlier versions:\n" + \
                            "\n".join(f"- {e}" for e in ex)
                tasks.append((i, GEN.format(hint=rng.choice(HINTS), avoid=avoid, q=qs[i]), rng.randrange(1 << 30)))
        rng.shuffle(tasks)
        gens = list(pool.map(lambda t: (t[0], chat(args.server, t[1], 1.0, 96, t[2])), tasks))
        stats["generated"] += len(gens)
        cand = []
        for i, g in gens:
            c = clean(g, qs[i])
            if c is None:
                stats["bad"] += 1
                continue
            k = norm(c)
            if k in seen:
                stats["duplicate"] += 1
                continue
            seen.add(k)                      # also blocks duplicates among this round's candidates
            cand.append((i, c))
        verdicts = list(pool.map(lambda t: chat(args.server, VER.format(a=qs[t[0]], b=t[1]), 0.0, 3, 0), cand))
        for (i, c), v in zip(cand, verdicts):
            if not v.strip().lower().startswith("yes"):
                stats["rejected"] += 1      # stays in seen: not proposed again
                continue
            if len(acc[i]) < counts[i]:
                acc[i].append(c)
                stats["accepted"] += 1
            else:
                seen.discard(norm(c))
        out_f.write_text(json.dumps(acc))
        stats_f.write_text(json.dumps(stats))
        logger.info(f"round {rnd}: {stats}")
    short = {i: counts[i] - len(acc[i]) for i in range(len(qs)) if counts[i] > len(acc[i])}
    stats["short"] = short
    stats_f.write_text(json.dumps(stats))
    if short:
        raise SystemExit(f"{len(short)} questions still short of rewordings: {short}")
    logger.info("all rewordings done")


if __name__ == "__main__":
    main()
