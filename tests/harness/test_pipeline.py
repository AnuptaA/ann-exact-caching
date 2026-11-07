"""Pipeline stage markers, ordering, force, and the tiny dataset through every stage."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from harness import data, datasets, pipeline


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
    with pytest.raises(pipeline.StageError, match="tiny base is"):
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


def test_gpu_contexts_get_large_ground_truth_chunks_and_explicit_values_win(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "auto_device", lambda: "cuda")
    gpu = pipeline.make_context("tiny", tmp_path, device="cuda")
    assert (gpu.gt_query_chunk, gpu.gt_base_chunk) == (pipeline.GPU_QUERY_CHUNK, pipeline.GPU_BASE_CHUNK)
    cpu = pipeline.make_context("tiny", tmp_path, device="numpy")
    assert cpu.gt_query_chunk == pipeline.groundtruth.QUERY_CHUNK
    explicit = pipeline.make_context("tiny", tmp_path, device="cuda", gt_query_chunk=64, gt_base_chunk=None)
    assert explicit.gt_query_chunk == 64 and explicit.gt_base_chunk == pipeline.GPU_BASE_CHUNK


def test_failures_are_loud_logged_and_recorded(tmp_path):
    ctx = _ctx(tmp_path)
    pipeline.run_pipeline(ctx, stages=["fetch"])
    np.save(ctx.raw / "base.npy", np.zeros((10, 16), dtype=np.float32))
    with pytest.raises(pipeline.StageError) as err:
        pipeline.run_pipeline(ctx, stages=["to_f32"])
    text = str(err.value)
    assert "[tiny] stage to_f32 failed" in text and ".failed/to_f32.json" in text and "last log lines" in text
    record = json.loads((ctx.root / ".failed" / "to_f32.json").read_text())
    assert record["stage"] == "to_f32" and "Traceback" in record["traceback"]
    assert not pipeline.is_done(ctx, "to_f32")
    log = (ctx.logs / "generate.log").read_text()
    assert "stage fetch done" in log and "FAILED" in log
    assert json.loads((ctx.root / "env.json").read_text())["list_size"] == pipeline.LIST_SIZE
    np.save(ctx.raw / "base.npy", np.random.default_rng(0).normal(size=(4000, 16)).astype(np.float32))
    pipeline.run_pipeline(ctx, stages=["to_f32"])                           # a rerun clears the record
    assert not (ctx.root / ".failed" / "to_f32.json").exists()


def test_numpy_ground_truth_on_a_big_dataset_is_refused_unless_allowed(tmp_path):
    ctx = pipeline.make_context("sphere", tmp_path, device="numpy")
    ctx.root.mkdir(parents=True)
    data.write_f32bin(ctx.root / "official.f32bin", np.zeros((96005, 768), dtype=np.float32)[:10])
    big = ctx.spec
    ctx = pipeline.Context(Path(tmp_path), big, device="numpy")
    with pytest.raises(pipeline.StageError, match="Q1_ALLOW_SLOW"):
        pipeline.check_resources(ctx, "groundtruth")
    ctx.extra["allow_slow"] = True
    pipeline.check_resources(ctx, "groundtruth")            # accepted on purpose (may warn)


def test_pg_stage_resource_checks_use_the_server_data_dir_and_refuse_small_ram(tmp_path, monkeypatch):
    ctx = pipeline.make_context("sphere", tmp_path)
    seen = []
    monkeypatch.setattr("workloads.q1.datasets.common.check_free_disk",
                        lambda path, needed: seen.append((Path(path), needed)))
    monkeypatch.setenv("Q1_PGDATA", str(tmp_path / "pg"))
    monkeypatch.setattr(pipeline, "total_ram_bytes", lambda: 128 * 2 ** 30)
    pipeline.check_resources(ctx, "pg_load")
    pipeline.check_resources(ctx, "pg_index")
    assert [p for p, _ in seen] == [tmp_path / "pg"] * 2 and seen[0][1] > 28 * 2 ** 30
    monkeypatch.setattr(pipeline, "total_ram_bytes", lambda: 16 * 2 ** 30)
    with pytest.raises(pipeline.StageError, match="GiB of RAM"):
        pipeline.check_resources(ctx, "pg_index")
    monkeypatch.delenv("Q1_PGDATA")
    assert pipeline.pg_data_dir(pipeline.make_context("sphere", tmp_path, dsn="host=127.0.0.1 port=1")) is None


@pytest.mark.parametrize("name", ["esci", "sphere"])
def test_cpu_encoding_of_big_text_datasets_is_refused_unless_allowed(tmp_path, name):
    ctx = pipeline.make_context(name, tmp_path, device="numpy")
    with pytest.raises(pipeline.StageError, match="Q1_ALLOW_SLOW"):
        pipeline.stage_embed(ctx)


def test_validation_catches_bad_outputs(tmp_path):
    ctx = _ctx(tmp_path)
    pipeline.run_pipeline(ctx, stages=["fetch", "embed", "to_f32", "inputs", "groundtruth"])
    pipeline.validate_stage(ctx, "to_f32")
    pipeline.validate_stage(ctx, "inputs")
    pipeline.validate_stage(ctx, "groundtruth")
    bad = np.zeros((4000, 16), dtype=np.float32)
    bad[5, 3] = np.nan
    data.write_f32bin(ctx.root / "base.f32bin", np.where(np.isnan(bad), 1.0, bad)[:3999])
    with pytest.raises(ValueError, match="shape"):
        pipeline.validate_stage(ctx, "to_f32")
    with pytest.raises(ValueError, match="NaN"):
        pipeline.check_vectors("x", np.full((100, 4), np.nan, dtype=np.float32), 100, 4)
    with pytest.raises(ValueError, match="zeros"):
        pipeline.check_vectors("x", np.zeros((100, 4), dtype=np.float32), 100, 4)
    first = next((ctx.root / "gt").glob("chunk_*.npz"))
    z = dict(np.load(first))
    z["ids"][0, 0] = 10 ** 9
    np.savez(first, **z)
    with pytest.raises(ValueError, match="out-of-range"):
        pipeline.validate_stage(ctx, "groundtruth")


def test_cuda_request_without_a_gpu_and_cpu_esci_embedding_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "auto_device", lambda: "numpy")
    with pytest.raises(pipeline.StageError, match="cannot see a CUDA device"):
        pipeline.make_context("tiny", tmp_path, device="cuda")
    ctx = pipeline.make_context("esci", tmp_path, device="numpy")
    with pytest.raises(pipeline.StageError, match="many hours"):
        pipeline.stage_embed(ctx)


def test_sphere_embedding_runs_in_a_separate_stripped_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    stub = tmp_path / "encode_python"
    stub.write_text("#!/bin/sh\n"
                    "env | sort > \"$(dirname \"$0\")/seen_env.txt\"\n"
                    "echo \"$@\" > \"$(dirname \"$0\")/seen_args.txt\"\n"
                    "out=\"\"; while [ $# -gt 0 ]; do [ \"$1\" = --out ] && out=\"$2\"; shift; done\n"
                    "echo data > \"$out\"\n")
    stub.chmod(0o755)
    raw = tmp_path / "raw"
    raw.mkdir()
    datasets._sphere_embed(raw, {"device": "cuda", "encode_python": str(stub)})
    env = (tmp_path / "seen_env.txt").read_text()
    assert "AWS_SECRET_ACCESS_KEY" not in env and "TORCH_FORCE_WEIGHTS_ONLY_LOAD=1" in env
    assert "HF_HUB_OFFLINE=1" in env and "TRANSFORMERS_OFFLINE=1" in env
    args = (tmp_path / "seen_args.txt").read_text()
    assert "-m workloads.q1 triviaqa-encode" in args and f"--tokenizer-dir {raw}/bert-base-uncased" in args
    assert "--device cuda" in args and (raw / "triviaqa_enc.npy").exists()
    assert not (raw / "triviaqa_enc.part.npy").exists()


def test_tokenizer_fetch_and_safe_torch_guard(tmp_path, monkeypatch):
    import sys
    import types
    from workloads.q1.datasets import common, triviaqa
    monkeypatch.setattr(common, "download_file", lambda url, dest, **kw: Path(dest).write_text("vocab") or Path(dest))
    out = triviaqa.fetch_tokenizer(tmp_path)
    assert (out / "vocab.txt").exists() and json.loads((out / "tokenizer_config.json").read_text()) == {"do_lower_case": True}
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.5.1+cu121"))
    with pytest.raises(RuntimeError, match="too old"):
        triviaqa.check_safe_torch()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.7.0+cu128"))
    triviaqa.check_safe_torch()
