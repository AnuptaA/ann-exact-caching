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


def fetch_listing():
    return [
        {"path": "sphere/sphere-ip-10m.npy", "id": "s", "size": 40},
        {"path": "sphere/sphere-ip-10m.json", "id": "j", "size": 2},
        {"path": "triviaqa/triviaqa-ip-sim-100k-0.3.npy", "id": "t3", "size": 7},
        {"path": "tripclick/tripclick-ip-docs.npy", "id": "a", "size": 10},
        {"path": "ESCI/queries.npz", "id": "q", "size": 3},
        {"path": "ESCI/data_vectors/shard_000.npz", "id": "sh", "size": 4},
    ]


def run_fetch(tmp_path, capsys, *flags):
    from scripts import data_fetch
    listing = tmp_path / "listing.json"
    listing.write_text(json.dumps(fetch_listing()))
    code = data_fetch.main([*flags, "--dry-run", "--root", str(tmp_path),
                            "--listing", str(listing)])
    return code, capsys.readouterr().out


def test_fetch_modes_are_independent(tmp_path, capsys):
    code, out = run_fetch(tmp_path, capsys, "--sphere")
    assert code == 0
    assert "sphere-ip-10m.npy" in out and "triviaqa" not in out and "spacev" not in out


def test_all_excludes_esci_and_esci_mode_maps_paths(tmp_path, capsys):
    _, out = run_fetch(tmp_path, capsys, "--all")
    assert "datasets/esci" not in out and "spacev-10m.npy" in out
    _, out = run_fetch(tmp_path, capsys, "--esci")
    assert "datasets/esci/queries.npz" in out
    assert "datasets/esci/data_vectors/shard_000.npz" in out


def test_missing_drive_file_is_reported(tmp_path, capsys):
    _, out = run_fetch(tmp_path, capsys, "--triviaqa")
    assert out.count("missing") == 2 and "download" in out


def test_no_corpus_skips_large_medrag_files(tmp_path, capsys):
    _, out = run_fetch(tmp_path, capsys, "--medrag", "--no-corpus")
    assert "medrag-ip-docs.npy" not in out and "base.json" in out
    _, out = run_fetch(tmp_path, capsys, "--medrag")
    assert "medrag-ip-docs.npy" in out


def test_existing_file_with_expected_size_is_kept(tmp_path):
    from scripts import data_fetch
    path = tmp_path / "x.npy"
    path.write_bytes(b"x" * 5)
    assert data_fetch.action_for(path, 5, False) == "keep"
    assert data_fetch.action_for(path, 6, False) == "replace"
    assert data_fetch.action_for(path, 5, True) == "replace"
    assert data_fetch.action_for(tmp_path / "y", 5, False) == "download"


def test_space_floor_per_download():
    from scripts import data_fetch
    assert data_fetch.space_ok(5 * 2**30, 20 * 2**30)
    assert not data_fetch.space_ok(15 * 2**30, 20 * 2**30)


def write_npy(path, array):
    np.save(path, array)


def make_ip_files(directory, bad_query=False):
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(60, 7)).astype(np.float32)
    norms = np.linalg.norm(docs, axis=1)
    top = float(norms.max())
    docs = np.hstack([docs, np.sqrt(top**2 - norms**2)[:, None]]).astype(np.float32)
    queries = np.hstack([rng.normal(size=(20, 7)), np.zeros((20, 1))]).astype(np.float32)
    if bad_query:
        queries[:, -1] = 1.0
    write_npy(directory / "sphere-ip-10m.npy", docs)
    write_npy(directory / "triviaqa-ip-sim-100k-0.3.npy", queries)
    (directory / "sphere-ip-10m.json").write_text(json.dumps({"M": top}))


def test_sphere_verifier_flags_wrong_shape_on_small_files(tmp_path, capsys):
    make_ip_files(tmp_path)
    assert verify_data.verify_sphere(tmp_path) == 1
    assert "expected (10000000, 769)" in capsys.readouterr().out


def test_ip_checks_on_small_files_pass_and_fail(tmp_path):
    make_ip_files(tmp_path)
    docs = np.load(tmp_path / "sphere-ip-10m.npy")
    queries = np.load(tmp_path / "triviaqa-ip-sim-100k-0.3.npy")
    meta = json.loads((tmp_path / "sphere-ip-10m.json").read_text())
    assert verify_data.check_ip_transform(docs, queries, meta["M"])["problems"] == []
    queries[:, -1] = 1.0
    assert verify_data.check_ip_transform(docs, queries, meta["M"])["problems"]


def test_spacev_verifier_reports_shape_problem(tmp_path, capsys):
    write_npy(tmp_path / "spacev-sim-100k-0.3.npy", np.zeros((5, 100), np.int8))
    assert verify_data.verify_spacev(tmp_path) == 1
    assert "spacev-sim-100k-0.3.npy" in capsys.readouterr().out
    write_npy(tmp_path / "spacev-sim-100k-0.3.npy", np.zeros((100_000, 100), np.int8))
    assert verify_data.verify_spacev(tmp_path) == 0


def test_manifest_lists_expected_files_not_on_disk(tmp_path):
    out = tmp_path / "MANIFEST.md"
    verify_data.write_manifest(out, [])
    text = out.read_text()
    assert "sphere-ip-10m.npy" in text and "spacev-10m.npy" in text
    assert "triviaqa-ip-sim-100k-0.99.npy" in text
