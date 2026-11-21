import json

import numpy as np

from scripts import drive_fetch, verify_data


def make_listing():
    return [
        {"path": "tripclick/tripclick-ip-docs.npy", "id": "a", "size": 10},
        {"path": "tripclick/other.bin", "id": "b", "size": 5},
        {"path": "medrag/medrag-ip-docs.npy", "id": "c", "size": 100},
        {"path": "medrag/base.json", "id": "d", "size": 1},
        {"path": "unrelated/secret.npy", "id": "e", "size": 1},
    ]


def test_local_mode_excludes_corpus_and_unrelated():
    names = {item["id"] for item in drive_fetch.select(make_listing(), "local")}
    assert names == {"a", "d"}


def test_cluster_mode_adds_corpus_only():
    names = {item["id"] for item in drive_fetch.select(make_listing(), "cluster")}
    assert names == {"a", "c", "d"}


def test_plan_keeps_same_size_and_replaces_different(tmp_path):
    chosen = drive_fetch.select(make_listing(), "local")
    (tmp_path / "datasets/aker").mkdir(parents=True)
    (tmp_path / "datasets/aker/tripclick-ip-docs.npy").write_bytes(b"x" * 10)
    actions = {a["id"]: a["action"] for a in drive_fetch.plan(chosen, tmp_path)}
    assert actions == {"a": "keep", "d": "download"}
    (tmp_path / "datasets/aker/tripclick-ip-docs.npy").write_bytes(b"x" * 3)
    actions = {a["id"]: a["action"] for a in drive_fetch.plan(chosen, tmp_path)}
    assert actions["a"] == "replace"


def test_space_floor():
    actions = [{"action": "download", "size": 5 * 2**30}]
    assert drive_fetch.enough_space(actions, 20 * 2**30)
    assert not drive_fetch.enough_space(actions, 12 * 2**30)
    assert drive_fetch.enough_space([{"action": "keep", "size": 5 * 2**30}], 11 * 2**30)


def test_ip_transform_check_accepts_valid_and_flags_bad():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 8)).astype(np.float32)
    top = float(np.linalg.norm(base, axis=1).max())
    extra = np.sqrt(top**2 - np.linalg.norm(base, axis=1) ** 2).astype(np.float32)
    docs = np.hstack([base, extra[:, None]])
    queries = np.hstack([rng.normal(size=(20, 8)).astype(np.float32), np.zeros((20, 1), np.float32)])
    assert verify_data.check_ip_transform(docs, queries, top)["problems"] == []
    queries[3, -1] = 0.5
    assert any("last coordinate" in p for p in verify_data.check_ip_transform(docs, queries)["problems"])
    docs[0, -1] += 1.0
    assert any("norms differ" in p for p in verify_data.check_ip_transform(docs, queries)["problems"])


def test_zipf_draw_check():
    seq = list(range(500)) + [0] * 9500
    counts = np.bincount(seq, minlength=500).tolist()
    assert verify_data.check_zipf_draw({"seq": seq, "counts": counts}) == []
    assert verify_data.check_zipf_draw({"seq": seq[:-1], "counts": counts})
    seq2 = [0] * 10_000
    assert verify_data.check_zipf_draw({"seq": seq2, "counts": np.bincount(seq2, minlength=500).tolist()})
    json.dumps({"seq": seq})
