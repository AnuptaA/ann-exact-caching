"""TriviaQA questions encoded with Sphere's DPR query encoder, not normalized.

The questions are public TriviaQA (any split), de-duplicated by text. They are
encoded into sphere-10m's space with dpr_web_query_encoder_hf (BERT-base CLS, no
projection). The checkpoint is a pickle from a third party: run encoding in an
isolated environment (own venv or container, no credentials), loading with
torch's safe weights-only path.
"""

import json
import logging
import tarfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HF = "https://huggingface.co/datasets/mandarjoshi/trivia_qa/resolve/refs%2Fconvert%2Fparquet"
SPLIT_URLS = {
    "train": f"{HF}/rc.nocontext/train/0000.parquet",
    "validation": f"{HF}/rc.nocontext/validation/0000.parquet",
    "test": f"{HF}/rc.nocontext/test/0000.parquet",
}
ENCODER_URL = "https://dl.fbaipublicfiles.com/sphere/dpr_web_query_encoder_hf.tar.gz"
VOCAB_URL = "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt"
MIN_TORCH = (2, 6)     # torch.load defaults to weights_only=True from 2.6
DIM = 768
EXPECTED_DISTINCT = 95_000
COUNT_TOLERANCE = 0.1


def dedup_questions(questions) -> list[str]:
    """Distinct question texts, first occurrence kept, surrounding whitespace dropped."""
    seen = set()
    out = []
    for q in questions:
        q = str(q).strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def load_questions(parquet_paths: dict) -> tuple[list[str], dict]:
    """Distinct questions across the given split files and the count report."""
    import pandas as pd
    all_questions = []
    report = {}
    for split, path in parquet_paths.items():
        qs = pd.read_parquet(path, columns=["question"])["question"].tolist()
        report[split] = len(qs)
        all_questions += qs
    distinct = dedup_questions(all_questions)
    report["total"] = len(all_questions)
    report["distinct"] = len(distinct)
    logger.info(f"TriviaQA question counts {report} (expected distinct ~{EXPECTED_DISTINCT})")
    if abs(len(distinct) - EXPECTED_DISTINCT) > COUNT_TOLERANCE * EXPECTED_DISTINCT:
        logger.warning("distinct question count is far from the expected ~95K")
    return distinct, report


def encode_questions(questions: list[str], encode, batch_size: int = 128) -> np.ndarray:
    """Stack encoder outputs as float32, no normalization."""
    parts = [np.asarray(encode(questions[i:i + batch_size]), dtype=np.float32)
             for i in range(0, len(questions), batch_size)]
    out = np.vstack(parts) if parts else np.zeros((0, DIM), dtype=np.float32)
    if out.shape != (len(questions), DIM):
        raise ValueError(f"encoder returned {out.shape}, expected {(len(questions), DIM)}")
    return out


def fetch_tokenizer(dest: Path) -> Path:
    """The BERT uncased vocabulary, fetched up front so the isolated encoding step needs no network."""
    from workloads.q1.datasets import common
    target = Path(dest) / "bert-base-uncased"
    target.mkdir(parents=True, exist_ok=True)
    common.download_file(VOCAB_URL, target / "vocab.txt")
    (target / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": True}))
    return target


def check_safe_torch() -> None:
    """Refuse to load a third-party pickle with a torch that does not default to weights_only."""
    import torch
    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if (major, minor) < MIN_TORCH:
        raise RuntimeError(f"torch {torch.__version__} is too old to load checkpoints safely; "
                           f"need >= {MIN_TORCH[0]}.{MIN_TORCH[1]}")


def dpr_query_encoder(checkpoint_dir: Path, max_length: int = 64, device: str = "cpu",
                      tokenizer_dir: Path | None = None):
    """Callable list[str] -> (n, 768) ndarray using the DPR question encoder (CLS).

    With tokenizer_dir (a folder from fetch_tokenizer) nothing is downloaded.
    """
    import torch
    from transformers import AutoTokenizer, BertTokenizerFast, DPRQuestionEncoder
    check_safe_torch()
    tokenizer = (BertTokenizerFast.from_pretrained(str(tokenizer_dir)) if tokenizer_dir is not None
                 else AutoTokenizer.from_pretrained("bert-base-uncased"))
    model = DPRQuestionEncoder.from_pretrained(str(checkpoint_dir)).to(device).eval()

    def encode(texts):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            return model(**batch).pooler_output.cpu().numpy()
    return encode


def fetch_questions(out_dir: Path) -> dict:
    """Download the three splits, de-duplicate, write triviaqa_questions.npy; returns the counts."""
    from workloads.q1.datasets import common
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {split: common.download_file(url, out / f"triviaqa_{split}.parquet")
             for split, url in SPLIT_URLS.items()}
    questions, report = load_questions(paths)
    np.save(out / "triviaqa_questions.npy", np.array(questions, dtype=str))
    (out / "triviaqa_counts.json").write_text(json.dumps(report))
    return report


def fetch_encoder(dest: Path) -> Path:
    """Download and unpack dpr_web_query_encoder_hf; members must stay inside dest."""
    from workloads.q1.datasets import common
    dest = Path(dest)
    target = dest / "dpr_web_query_encoder_hf"
    if (target / "config.json").exists():
        return target
    tar_path = common.download_file(ENCODER_URL, dest / "dpr_web_query_encoder_hf.tar.gz")
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if not (member.isfile() or member.isdir()) or ".." in parts or member.name.startswith("/"):
                raise ValueError(f"unexpected tar member {member.name!r}")
            if parts[0] != "dpr_web_query_encoder_hf":
                raise ValueError(f"tar member outside dpr_web_query_encoder_hf/: {member.name!r}")
        # filter= exists from Python 3.10.12; members are validated above
        extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(dest, **extra)
    if not (target / "config.json").exists():
        raise ValueError("encoder archive has no dpr_web_query_encoder_hf/config.json")
    return target
