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
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import os as _os
TAG = _os.environ.get("MEDRAG_TAG", "")      # e.g. "_100k": a larger workload beside the 10k one (11-23)
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
GEN_MULTI = ("Write {m} rephrasings of the medical question below. Each must be worded differently from the "
             "question and from the other rephrasings, but ask exactly the same thing (the correct yes / no / maybe "
             "answer must not change). {hint} Reply with the {m} rephrased questions only, one per line, without "
             "numbering.{avoid}\n\nQuestion: {q}")
VER = ("Question A: {a}\nQuestion B: {b}\n\nDo A and B ask exactly the same thing, so that the correct "
       "yes / no / maybe answer to A is always the correct answer to B? Reply with only Yes or No.")


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


BEDROCK = {}        # 11-23: {"model": id, "client": boto3 bedrock-runtime} when --bedrock-model is given


def chat_bedrock(prompt, temperature, max_tokens):
    """Amazon Bedrock Converse (on-demand), no seed."""
    c = BEDROCK["client"]
    for attempt in range(100):
        try:
            # Claude Haiku 5.5 accepts only temperature 1 (0 is rejected) and thinks first when no temperature is
            # given: always 1, at least 16 tokens, and the first text block of the reply
            r = c.converse(modelId=BEDROCK["model"], messages=[{"role": "user", "content": [{"text": prompt}]}],
                           inferenceConfig={"maxTokens": max(max_tokens, 16), "temperature": 1.0},
                           # no extended thinking: it otherwise spends the token budget before any text (11-23)
                           additionalModelRequestFields={"thinking": {"type": "disabled"}})
            u = r.get("usage", {})
            with BEDROCK["lock"]:
                BEDROCK["in"] += u.get("inputTokens", 0)
                BEDROCK["out"] += u.get("outputTokens", 0)
            return next((b["text"] for b in r["output"]["message"]["content"] if "text" in b), "")
        except Exception as e:
            name = type(e).__name__ + str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
            if "ValidationException" in name:
                logger.warning(f"request rejected: {e!r}")
                return ""
            if attempt % 10 == 9:
                logger.warning(f"request failed {attempt + 1}x: {e!r}")
            time.sleep(min(60, 0.5 * 2 ** min(attempt, 7)))      # throttling: exponential back-off
    raise RuntimeError("Bedrock unreachable")


def chat(server, prompt, temperature, max_tokens, seed):
    if BEDROCK:
        return chat_bedrock(prompt, temperature, max_tokens)
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                       "top_p": 0.95, "max_tokens": max_tokens, "seed": seed, "cache_prompt": True}).encode()
    for attempt in range(100):
        try:
            req = urllib.request.Request(server + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 400:                        # request rejected (e.g. too long): a failed candidate
                logger.warning(f"request rejected: {e!r}")
                return ""
            time.sleep(min(300, 5 * (attempt + 1)))
        except Exception as e:                       # server busy / restarting / suspended: retry
            if attempt % 10 == 9:
                logger.warning(f"request failed {attempt + 1}x: {e!r}")
            time.sleep(min(300, 5 * (attempt + 1)))
    raise RuntimeError("server unreachable")


FOREIGN = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC", "HEBREW", "THAI", "DEVANAGARI")


def foreign(t):
    """A character of another script (11-14: an accepted rewording ended in Chinese characters);
    Latin with accents and Greek letters (alpha, beta, ...) are fine."""
    return any(any(w in unicodedata.name(ch, "") for w in FOREIGN) for ch in t if ord(ch) > 127)


CHATTER = re.compile(r"\(note|\bnote:|please|let me know|translat", re.I)    # the model talking


def clean(text, q):
    t = text.strip().split("\n")[0].strip().strip('"').strip("“”").strip()
    t = re.sub(r"^(rephrased question|question|rephrased)\s*:\s*", "", t, flags=re.I).strip().strip('"')
    if not t or len(t) > 3 * len(q) + 40 or len(t) < 8 or foreign(t) or CHATTER.search(t):
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bedrock-model", default=None,
                    help="generate and verify with this Amazon Bedrock model (inference profile id) instead of the local "
                         "llama.cpp server; credentials from AWS_PROFILE / the environment (11-23)")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--per-call", type=int, default=1, help="rewordings asked for per generation call (11-23)")
    args = ap.parse_args()
    if args.bedrock_model:
        import threading
        import boto3
        from botocore.config import Config
        BEDROCK.update(model=args.bedrock_model, lock=threading.Lock(), **{"in": 0, "out": 0},
                       client=boto3.client("bedrock-runtime", region_name=args.region,
                                           config=Config(max_pool_connections=args.workers + 8,
                                                         retries={"max_attempts": 2, "mode": "standard"})))
    draw = json.loads((D / f"zipf_draw{TAG}.json").read_text())
    qs, counts = draw["questions"], draw["counts"]
    out_f, stats_f = D / f"paraphrases{TAG}.json", D / f"paraphrase_stats{TAG}.json"
    acc = {int(k): v for k, v in json.loads(out_f.read_text()).items()} if out_f.exists() else {}
    stats = json.loads(stats_f.read_text()) if stats_f.exists() else {"generated": 0, "bad": 0, "duplicate": 0,
                                                                     "rejected": 0, "accepted": 0}
    for i in range(len(qs)):
        acc.setdefault(i, [])
        acc[i] = [p for p in acc[i] if not foreign(p) and not CHATTER.search(p)]   # resume: re-filter
    seen = {norm(q) for q in qs} | {norm(p) for v in acc.values() for p in v}
    rng = random.Random(args.seed + sum(len(v) for v in acc.values()))
    pool = ThreadPoolExecutor(args.workers)
    # adaptive diversity (11-14): a question that gained nothing in a round is "stalled"; it then gets
    # its accepted versions shown ("worded differently from these") even below 20 of them, and a
    # temperature 0.1 higher per stalled round (up to 1.5) (a 16-occurrence question ended 2 short
    # after 120 rounds at temperature 1 without the list)
    stall = {i: 0 for i in range(len(qs))}
    for rnd in range(args.rounds):
        before = {i: len(acc[i]) for i in range(len(qs))}
        need = {i: counts[i] - len(acc[i]) for i in range(len(qs)) if counts[i] > len(acc[i])}
        if not need:
            break
        logger.info(f"round {rnd}: {len(need)} questions need {sum(need.values())} rewordings")
        tasks = []
        per = max(1, args.per_call)
        for i, n in need.items():
            for _ in range(max(2 if per == 1 else 1, -(-(int(1.5 * n) + 1) // per))):
                avoid = ""
                if len(acc[i]) >= 20 or (stall[i] and acc[i]):
                    ex = rng.sample(acc[i], min(15 if not stall[i] else 40, len(acc[i])))
                    while len(ex) > 3 and sum(len(e) + 3 for e in ex) > 1800:   # fits a slot's 1024-token context
                        ex.pop()
                    avoid = " It must be worded differently from all of these earlier versions:\n" + \
                            "\n".join(f"- {e}" for e in ex)
                prompt = (GEN.format(hint=rng.choice(HINTS), avoid=avoid, q=qs[i]) if per == 1 else
                          GEN_MULTI.format(m=per, hint=rng.choice(HINTS), avoid=avoid, q=qs[i]))
                tasks.append((i, prompt, rng.randrange(1 << 30), min(1.5, 1.0 + 0.1 * stall[i])))
        rng.shuffle(tasks)
        gens = list(pool.map(lambda t: (t[0], chat(args.server, t[1], t[3], 96 * per, t[2])), tasks))
        if per > 1:                          # one candidate per line of each reply (numbering / bullets dropped)
            gens = [(i, re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", ln)) for i, g in gens for ln in g.split("\n") if ln.strip()]
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
        for i in need:
            stall[i] = stall[i] + 1 if len(acc[i]) == before[i] else 0
        out_f.write_text(json.dumps(acc))
        stats_f.write_text(json.dumps(stats))
        logger.info(f"round {rnd}: {stats}" + (f" tokens in {BEDROCK['in']:,} out {BEDROCK['out']:,}" if BEDROCK else ""))
    short = {i: counts[i] - len(acc[i]) for i in range(len(qs)) if counts[i] > len(acc[i])}
    stats["short"] = short
    stats_f.write_text(json.dumps(stats))
    if short:
        raise SystemExit(f"{len(short)} questions still short of rewordings: {short}")
    logger.info("all rewordings done")


if __name__ == "__main__":
    main()
