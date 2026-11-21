"""ESCI embedding in resumable shards: resume, recipe mismatch, proven reuse, atomic assembly."""

import hashlib
import json

import numpy as np
import pytest

from workloads.q1.datasets import esci

DIM = 4


def fake_encoder(calls: list):
    def encode(texts):
        calls.append(len(texts))
        out = np.stack([np.frombuffer(hashlib.sha256(t.encode()).digest()[:DIM * 4], dtype=np.uint32)
                        .astype(np.float32) % 97 + 1.0 for t in texts])
        return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)
    return encode


TEXTS = [f"product {i}" for i in range(250)]


def embed(tmp_path, texts=TEXTS, calls=None, encode=None):
    calls = [] if calls is None else calls
    return esci.embed_sharded(texts, encode or fake_encoder(calls), tmp_path / "p.npy", dim=DIM, shard_texts=100)


def test_full_run_matches_a_single_pass_and_writes_a_manifest(tmp_path):
    path = embed(tmp_path)
    assert np.array_equal(np.load(path), fake_encoder([])(TEXTS))
    manifest = json.loads((tmp_path / "p.npy.manifest.json").read_text())
    assert manifest["n"] == 250 and manifest["recipe"]["model"] == esci.MODEL
    assert manifest["texts_sha256"] == esci.texts_sha256(TEXTS) and "code_commit" in manifest
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.npy", "p.npy.manifest.json"]


def test_a_killed_run_resumes_from_the_finished_shards(tmp_path):
    calls = []
    inner = fake_encoder(calls)

    def dying(texts):
        if len(calls) == 2:
            raise KeyboardInterrupt
        return inner(texts)

    with pytest.raises(KeyboardInterrupt):
        embed(tmp_path, encode=dying)
    assert sorted(p.name for p in (tmp_path / "p.npy.shards").glob("shard_*")) == \
        ["shard_00000.npy", "shard_00001.npy"] and not (tmp_path / "p.npy").exists()
    resumed = []
    path = embed(tmp_path, calls=resumed)
    assert resumed == [50]
    assert np.array_equal(np.load(path), fake_encoder([])(TEXTS))
    assert not (tmp_path / "p.npy.shards").exists()


def test_shards_of_other_texts_are_never_mixed(tmp_path):
    calls = []
    inner = fake_encoder(calls)

    def dying(texts):
        if calls:
            raise KeyboardInterrupt
        return inner(texts)

    with pytest.raises(KeyboardInterrupt):
        embed(tmp_path, encode=dying)
    with pytest.raises(ValueError, match="other texts"):
        embed(tmp_path, texts=TEXTS[:-1])


def test_existing_embeddings_are_reused_only_after_a_sampled_match(tmp_path):
    embed(tmp_path)
    calls = []
    embed(tmp_path, calls=calls)
    assert calls == [64]                                    # only the proof sample was encoded
    np.save(tmp_path / "p.npy", np.load(tmp_path / "p.npy")[::-1].copy())
    with pytest.raises(ValueError, match="do not match"):
        embed(tmp_path)
    with pytest.raises(ValueError, match="other texts"):
        embed(tmp_path, texts=TEXTS[:-1])


def test_a_file_without_a_manifest_is_adopted_if_it_matches_and_refused_if_not(tmp_path):
    np.save(tmp_path / "p.npy", fake_encoder([])(TEXTS))
    embed(tmp_path)
    assert json.loads((tmp_path / "p.npy.manifest.json").read_text())["adopted"] is True
    other = tmp_path / "x"
    other.mkdir()
    np.save(other / "p.npy", fake_encoder([])(TEXTS)[::-1].copy())
    with pytest.raises(ValueError, match="do not match"):
        embed(other)


def test_bad_shards_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="not L2-normalized"):
        esci.embed_sharded(TEXTS, lambda t: np.full((len(t), DIM), 3.0, dtype=np.float32),
                           tmp_path / "p.npy", dim=DIM, shard_texts=100)
    with pytest.raises(ValueError, match="expected"):
        esci.embed_sharded(TEXTS, lambda t: np.ones((len(t) - 1, DIM), dtype=np.float32) / 2,
                           tmp_path / "q.npy", dim=DIM, shard_texts=100)


def test_sub_batches_give_progress_lines_with_an_eta(tmp_path, caplog):
    import logging
    calls = []
    with caplog.at_level(logging.INFO, logger="workloads.q1.progress"):
        esci.embed_sharded(TEXTS, fake_encoder(calls), tmp_path / "p.npy", dim=DIM, shard_texts=100,
                           sub_batch=30, progress_interval=0.0)
    assert calls == [30, 30, 30, 10, 30, 30, 30, 10, 30, 20]
    lines = [r.message for r in caplog.records if r.message.startswith("p.npy:") and "texts/s" in r.message]
    assert len(lines) == len(calls) and "ETA" in lines[0] and "/250 texts" in lines[-1]
    assert np.array_equal(np.load(tmp_path / "p.npy"), fake_encoder([])(TEXTS))


def test_the_shard_size_is_not_part_of_the_compared_recipe(tmp_path):
    embed(tmp_path)
    calls = []
    esci.embed_sharded(TEXTS, fake_encoder(calls), tmp_path / "p.npy", dim=DIM, shard_texts=20_000)
    assert calls == [64]                                    # reused: only the proof sample was encoded
    manifest = json.loads((tmp_path / "p.npy.manifest.json").read_text())
    assert "shard_texts" not in manifest["recipe"] and manifest["made_with"]["shard_texts"] == 100
    old = {**manifest, "recipe": {**manifest["recipe"], "shard_texts": 100000}}
    (tmp_path / "p.npy.manifest.json").write_text(json.dumps(old))
    esci.embed_sharded(TEXTS, fake_encoder([]), tmp_path / "p.npy", dim=DIM, shard_texts=100)


def test_shards_of_another_size_are_never_mixed(tmp_path):
    calls = []
    inner = fake_encoder(calls)

    def dying(texts):
        if calls:
            raise KeyboardInterrupt
        return inner(texts)

    with pytest.raises(KeyboardInterrupt):
        embed(tmp_path, encode=dying)
    with pytest.raises(ValueError, match="holds shards"):
        esci.embed_sharded(TEXTS, fake_encoder([]), tmp_path / "p.npy", dim=DIM, shard_texts=50)


def test_the_manifest_records_how_the_embeddings_were_made(tmp_path):
    encode = fake_encoder([])
    encode.provenance = {"model_revision": "abc", "max_seq_length": 128, "device": "cpu"}
    esci.embed_sharded(TEXTS, encode, tmp_path / "p.npy", dim=DIM, shard_texts=100,
                       provenance={"device_notice": "No GPU found: running embedding on CPU", "threads": 4})
    made = json.loads((tmp_path / "p.npy.manifest.json").read_text())["made_with"]
    assert made["model_revision"] == "abc" and made["threads"] == 4 and made["shard_texts"] == 100
    assert made["device_notice"].startswith("No GPU found")
