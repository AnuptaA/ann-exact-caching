"""Where text is embedded: the audited sites are the only ones, and released vectors are never re-embedded.

harness/EMBEDDING_SITES.md lists every site. A new one must be added there and here.
"""

import re
from pathlib import Path

import numpy as np

from harness import data, datasets, pipeline

ROOT = Path(__file__).resolve().parents[2]
ENCODER_PATTERN = re.compile(r"sentence_transformers|SentenceTransformer|AutoModel|DPRQuestionEncoder|"
                             r"DPRContextEncoder|BertModel|from transformers|import transformers")
AUDITED_SITES = {
    "workloads/q1/datasets/esci.py",
    "workloads/q1/datasets/triviaqa.py",
    "workloads/q1/datasets/triviaqa_evidence.py",
    "datasets/esci/build_embedding_subsample.py",
    # not run in this repository's pipeline: their vectors are obtained precomputed
    "datasets/tripclick/build_workload.py",
    "datasets/reembed_triviaqa_queries.py",
}
# investigations/ holds exploratory scripts that no pipeline stage runs
SKIPPED_DIRS = {"venv", "stale", "tests", ".git", "build", "third_party", "node_modules", "investigations"}


def _python_files():
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if not SKIPPED_DIRS.intersection(rel.parts) and not rel.parts[0].startswith(".venv"):
            yield rel.as_posix(), path


def test_the_only_encoder_call_sites_are_the_audited_ones():
    found = {rel for rel, path in _python_files() if ENCODER_PATTERN.search(path.read_text())}
    assert found == AUDITED_SITES


def test_released_base_vectors_have_no_encoder_code():
    for name in ("spacev", "sift10m", "sphere"):
        text = (ROOT / "workloads/q1/datasets" / f"{name}.py").read_text()
        assert not ENCODER_PATTERN.search(text), name


def test_only_esci_and_the_sphere_queries_are_embedded_by_the_pipeline():
    embedding = {name for name, spec in datasets.SPECS.items() if spec.embed is not None}
    assert embedding == {"esci", "sphere"}
    # SPHERE: the base is the released file; the embed step writes only the query vectors
    source = (ROOT / "harness/datasets.py").read_text()
    sphere_embed = source[source.index("def _sphere_embed"):source.index("def _tiny_fetch")]
    assert "triviaqa_enc" in sphere_embed and "sphere-10m" not in sphere_embed


def test_to_f32_only_casts_released_integer_and_float64_vectors(tmp_path, monkeypatch):
    import dataclasses
    rng = np.random.default_rng(0)
    cases = {"spacev": rng.integers(-128, 127, size=(40, 6), dtype=np.int8),
             "sift10m": rng.integers(0, 255, size=(40, 6), dtype=np.uint8),
             "sphere": rng.normal(scale=14.0, size=(40, 6))}
    for name, base in cases.items():
        ctx = pipeline.make_context(name, tmp_path / name, device="cpu")
        spec = dataclasses.replace(ctx.spec, n_base=40, dim=6, base=lambda raw, b=base: b,
                                   official=lambda raw, b=base: b[:5])
        ctx.spec = spec
        ctx.root.mkdir(parents=True)
        pipeline.stage_to_f32(ctx)
        assert np.array_equal(data.read_f32bin(ctx.root / "base.f32bin"), base.astype(np.float32)), name
