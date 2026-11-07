"""Pipeline stage markers, ordering, force, and the tiny dataset through every stage."""

import os
from pathlib import Path

import numpy as np
import pytest

from harness import datasets, pipeline


def _ctx(tmp_path, **kw):
    return pipeline.make_context("tiny", tmp_path, device="numpy", **kw)


def test_stage_order_skip_and_force(tmp_path):
    ctx = _ctx(tmp_path)
    calls = []
    runners = {s: (lambda c, s=s: calls.append(s) or {"stage": s}) for s in pipeline.STAGES}
    assert pipeline.run_pipeline(ctx, runners=runners) == list(pipeline.STAGES)
    assert calls == list(pipeline.STAGES)
    assert pipeline.run_pipeline(ctx, runners=runners) == []            # everything done
    ran = pipeline.run_pipeline(ctx, force="groundtruth", runners=runners)
    assert ran == list(pipeline.STAGES[pipeline.STAGES.index("groundtruth"):])
    assert pipeline.run_pipeline(ctx, stages=["fetch"], runners=runners) == []
    with pytest.raises(ValueError):
        pipeline.run_pipeline(ctx, stages=["nope"], runners=runners)
    with pytest.raises(ValueError):
        datasets.get_spec("nope")


def test_failed_stage_leaves_no_marker_and_resumes(tmp_path):
    ctx = _ctx(tmp_path)
    state = {"fail": True}

    def flaky(c):
        if state["fail"]:
            raise RuntimeError("boom")
        return {}
    runners = {s: (lambda c: {}) for s in pipeline.STAGES}
    runners["inputs"] = flaky
    with pytest.raises(RuntimeError):
        pipeline.run_pipeline(ctx, runners=runners)
    assert pipeline.is_done(ctx, "to_f32") and not pipeline.is_done(ctx, "inputs")
    state["fail"] = False
    assert pipeline.run_pipeline(ctx, runners=runners)[0] == "inputs"


def test_data_stages_on_the_tiny_dataset_without_a_database(tmp_path):
    ctx = _ctx(tmp_path)
    ran = pipeline.run_pipeline(ctx, stages=["fetch", "embed", "to_f32", "inputs", "groundtruth"])
    assert ran == ["fetch", "embed", "to_f32", "inputs", "groundtruth"]
    s = ctx.spec
    assert (ctx.root / "base.f32bin").exists() and (ctx.root / "universe.npz").exists()
    z = np.load(ctx.root / "streams_seed0.npz")
    assert len(z["zipf_0.99"]) == s.n_requests and len(z["no_locality"]) == 600
    n_ids = s.n_groups * s.group_size + s.n_groups + 600
    assert sum(len(np.load(f)["ids"]) for f in (ctx.root / "gt").glob("chunk_*.npz")) == n_ids


def test_wrong_base_shape_is_rejected(tmp_path):
    ctx = _ctx(tmp_path)
    pipeline.run_pipeline(ctx, stages=["fetch"])
    np.save(ctx.raw / "base.npy", np.zeros((10, 16), dtype=np.float32))
    with pytest.raises(ValueError):
        pipeline.run_pipeline(ctx, stages=["to_f32"])


@pytest.mark.skipif(not os.environ.get("PGVECTOR_TEST_DSN"), reason="needs PostgreSQL + pgvector")
def test_tiny_dataset_through_every_stage_with_real_pgvector(tmp_path):
    ctx = _ctx(tmp_path, dsn=os.environ["PGVECTOR_TEST_DSN"], pg_workers=2)
    ran = pipeline.run_pipeline(ctx)
    assert ran == list(pipeline.STAGES)
    import json
    tune = json.loads((ctx.root / "tune.json").read_text())
    assert tune["ef_search"] >= tune["list_size"] and (ctx.root / "backend_recall.json").exists()
    rec = json.loads((ctx.root / "backend_recall.json").read_text())
    assert 0.85 <= rec["mean_recall_all_ids"] <= 1.0
    assert pipeline.run_pipeline(ctx) == []                                    # resumable


@pytest.mark.skipif(not os.environ.get("PGVECTOR_TEST_DSN"), reason="needs PostgreSQL + pgvector")
def test_fidelity_run_on_the_tiny_database(tmp_path):
    from harness import data, fidelity
    ctx = _ctx(tmp_path, dsn=os.environ["PGVECTOR_TEST_DSN"], pg_workers=2)
    pipeline.run_pipeline(ctx)
    official = data.read_f32bin(ctx.root / "official.f32bin")
    rng = np.random.default_rng(0)
    payload = np.asarray(official[rng.integers(0, 40, size=800)])        # a stream with repeats
    np.save(tmp_path / "payload.npy", payload)
    driver_dir = Path(__file__).resolve().parents[2] / "build" / "replay"
    if not (driver_dir / "ak_replay_standard").exists():
        pytest.skip("drivers not built")
    out = fidelity.run_fidelity(tmp_path, os.environ["PGVECTOR_TEST_DSN"], driver_dir, tmp_path / "out",
                                tmp_path / "work", ef_search=100, dataset="tiny",
                                payload_path=tmp_path / "payload.npy")
    assert set(out["report"]) == {"1%", "5%"} and Path(out["plot"]).exists()
    assert out["report"]["5%"]["delivered_recall"] >= 0.8 and out["report"]["5%"]["hit_rate"] > 0.3
