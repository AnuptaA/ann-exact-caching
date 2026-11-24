import json
from pathlib import Path

import numpy as np
import pytest

from scripts import verify_data

DATA = Path(__file__).parent / "data"


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



def draw_counts():
    """500 counts with the 100K draw's statistics: max 7,602, median 92, min 38, total 100,000."""
    lower = [38] + [92] * 249
    upper = [92] + [7602] + [279] * 247 + [279 + 168]
    counts = np.array(lower + upper)
    assert len(counts) == 500 and counts.sum() == 100_000
    return counts


def make_draw(seed=0, zipf=0.8):
    counts = draw_counts()
    seq = np.random.default_rng(1).permutation(np.repeat(np.arange(500), counts))
    return {"zipf": zipf, "n": 100_000, "seed": seed, "pubmedqa_ids": [str(i) for i in range(500)],
            "questions": [f"q{i}" for i in range(500)], "counts": counts.tolist(), "seq": seq.tolist(),
            "max_count": int(counts.max())}


def test_zipf_draw_100k_checks_the_published_statistics():
    draw = make_draw()
    assert verify_data.check_zipf_draw(draw) == []
    assert any("exponent" in p for p in verify_data.check_zipf_draw(make_draw(zipf=0.9)))
    assert any("seed" in p for p in verify_data.check_zipf_draw(make_draw(seed=1)))
    short = dict(draw, seq=draw["seq"][:-1])
    assert any("99999" in p for p in verify_data.check_zipf_draw(short))
    skew = make_draw()
    skew["seq"] = sorted(skew["seq"])[:-5] + [0] * 5
    assert verify_data.check_zipf_draw(skew)


def test_requests_json_must_follow_the_draw_with_unique_texts():
    draw = make_draw()
    texts = [f"reworded {i}" for i in range(100_000)]
    ok = {"texts": texts, "question": draw["seq"]}
    assert verify_data.check_requests_json(ok, draw) == []
    dup = {"texts": texts[:-1] + [texts[0]], "question": draw["seq"]}
    assert any("repeated" in p for p in verify_data.check_requests_json(dup, draw))
    wrong = {"texts": texts, "question": draw["seq"][::-1]}
    assert any("question sequence" in p for p in verify_data.check_requests_json(wrong, draw))
    assert any("texts" in p for p in verify_data.check_requests_json({"texts": texts[:5], "question": []}, draw))


def test_medrag_verification_stops_on_a_wrong_request_file(tmp_path, capsys):
    aker, med = tmp_path / "aker", tmp_path / "medrag"
    aker.mkdir(), med.mkdir()
    np.save(aker / "medrag-ip_100k-requests.npy", np.zeros((10, 769), np.float32))
    assert verify_data.verify_medrag(med, aker) == 1
    out = capsys.readouterr().out
    assert "(100000, 769)" in out and "zipf_draw_100k.json" in out
    assert verify_data.verify_medrag(med, tmp_path / "nothing") == 1


def test_triviaqa_expected_shapes_are_the_published_counts():
    shapes = {n: v[0] for n, v in verify_data.AKER_EXPECTED.items() if n.startswith("triviaqa")}
    assert shapes["triviaqa-ip-sim-100k-0.3.npy"] == (99_903, 769) and shapes["triviaqa-ip-sim-100k-0.99.npy"] == (99_907, 769)
    assert "medrag-ip-requests.npy" not in verify_data.MEDRAG_EXPECTED


def listing():
    return json.loads((DATA / "drive_listings.json").read_text())


def run_fetch(tmp_path, capsys, *flags, drop=None):
    from scripts import data_fetch
    lst = listing()
    if drop:
        folder, name = drop
        lst[folder] = [e for e in lst[folder] if e["path"] != name]
    path = tmp_path / "listing.json"
    path.write_text(json.dumps(lst))
    code = data_fetch.main([*flags, "--dry-run", "--root", str(tmp_path), "--listing", str(path)])
    return code, capsys.readouterr().out


def test_every_mode_finds_all_its_files_in_the_real_listings(tmp_path, capsys):
    code, out = run_fetch(tmp_path, capsys, "--all")
    assert code == 0 and "missing" not in out.split("needs")[0]
    for name in ("medrag-ip_100k-requests.npy", "triviaqa-ip-sim-100k-0.6.npy", "tripclick-ip-docs.npy", "sphere-10m.npy"):
        assert name in out


def test_medrag_requests_come_only_from_the_100k_folder_and_never_the_10k(tmp_path, capsys):
    from scripts import data_fetch
    _, out = run_fetch(tmp_path, capsys, "--medrag")
    assert "medrag-ip_100k-requests.npy" in out and "medrag-ip-requests.npy" not in out
    assert "zipf_draw.json" not in out and "requests.json" not in out.replace("requests_100k.json", "")
    by = {i.name: i.folder for i in data_fetch.items_for("medrag")}
    assert by["medrag-ip_100k-requests.npy"] == "medrag100k" and by["medrag-ip-docs.npy"] == "medrag"
    assert by["base.json"] == "medrag" and by["zipf_draw_100k.json"] == "medrag100k"
    _, out = run_fetch(tmp_path, capsys, "--medrag", "--medrag-10k")
    assert "datasets/aker/medrag-10k-ip-requests.npy" in out and "datasets/medrag_10k/zipf_draw.json" in out


def test_no_corpus_skips_the_corpus_files(tmp_path, capsys):
    _, out = run_fetch(tmp_path, capsys, "--medrag", "--no-corpus")
    assert "medrag-ip-docs.npy" not in out and "base_pmids.npy" not in out and "base.json" in out


def test_a_missing_expected_file_stops_before_any_download(tmp_path, capsys):
    code, out = run_fetch(tmp_path, capsys, "--medrag", drop=("medrag100k", "requests_100k.json"))
    assert code == 3 and "requests_100k.json is not in the medrag100k folder (RECAL_DRIVE_MEDRAG_100K)" in out
    assert "needs " not in out


def test_sphere_mode_takes_the_hf_base_and_the_published_m_and_builds_the_ip_corpus():
    from scripts import data_fetch
    items = {i.name: i for i in data_fetch.items_for("sphere")}
    assert items["sphere-10m.npy"].source == "hf" and items["sphere-ip-10m.json"].folder == "triviaqa"
    assert "sphere-ip-10m.npy" not in items


def test_esci_is_only_listed_on_request_and_maps_into_datasets_esci(tmp_path, capsys):
    _, out = run_fetch(tmp_path, capsys, "--all")
    assert "datasets/esci" not in out
    _, out = run_fetch(tmp_path, capsys, "--esci")
    assert "datasets/esci/queries.npz" in out and "datasets/esci/data_vectors/shard_000.npz" in out


def test_space_floor():
    from scripts import data_fetch
    assert data_fetch.space_ok(5 * 2**30, 20 * 2**30) and not data_fetch.space_ok(15 * 2**30, 20 * 2**30)


def test_build_sphere_ip_corpus_matches_the_published_m(tmp_path):
    from scripts import build_sphere_ip
    rng = np.random.default_rng(3)
    base = rng.normal(scale=3.0, size=(300, 6))
    np.save(tmp_path / "sphere-10m.npy", base)
    stats = build_sphere_ip.build(np.load(tmp_path / "sphere-10m.npy", mmap_mode="r"), tmp_path / "ip.npy", block=64)
    ip = np.load(tmp_path / "ip.npy")
    assert ip.shape == (300, 7) and ip.dtype == np.float32
    norms = np.linalg.norm(ip.astype(np.float64), axis=1)
    assert np.allclose(norms, stats["M"], rtol=1e-4)
    (tmp_path / "pub.json").write_text(json.dumps({"M": stats["M"]}))
    build_sphere_ip.check_m(stats, tmp_path / "pub.json")
    (tmp_path / "bad.json").write_text(json.dumps({"M": stats["M"] * 1.01}))
    with pytest.raises(SystemExit):
        build_sphere_ip.check_m(stats, tmp_path / "bad.json")


def test_result_stamp_records_request_file_name_and_sha256(tmp_path):
    import hashlib
    from recal.measure import jobs_util as ju
    ju._data_files.clear()
    path = tmp_path / "medrag-ip_100k-requests.npy"
    path.write_bytes(b"abc" * 100)
    ju.register_data_file(path)
    res = ju.stamp_result({})
    entry = res["data_files"][0]
    assert entry["name"] == path.name and entry["sha256"] == hashlib.sha256(b"abc" * 100).hexdigest()
    assert (tmp_path / (path.name + ".sha256")).exists()
    ju.register_data_file(path)
    assert len(ju._data_files) == 1
    ju._data_files.clear()


def clear_drive_env(monkeypatch):
    from scripts import data_fetch
    for var in data_fetch.DRIVE_VARS.values():
        monkeypatch.delenv(var, raising=False)


def test_folder_id_accepts_a_link_or_an_id():
    from scripts import data_fetch
    assert data_fetch.folder_id("https://example.org/x/folders/abc_-12") == "abc_-12"
    assert data_fetch.folder_id("https://example.org/x/folders/abc_-12?usp=sharing") == "abc_-12"
    assert data_fetch.folder_id(" abc_-12 ") == "abc_-12"


def test_a_missing_variable_stops_before_any_listing_or_download(tmp_path, capsys, monkeypatch):
    from scripts import data_fetch
    clear_drive_env(monkeypatch)

    def fail(_):
        raise AssertionError("Drive was asked")
    monkeypatch.setattr(data_fetch, "list_drive", fail)
    code = data_fetch.main(["--medrag", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 4 and "RECAL_DRIVE_MEDRAG" in out and "RECAL_DRIVE_MEDRAG_100K" in out
    assert "RECAL_DRIVE_TRIPCLICK" not in out


def test_env_file_is_read_and_the_environment_wins(tmp_path, monkeypatch):
    from scripts import data_fetch
    clear_drive_env(monkeypatch)
    (tmp_path / ".env.local").write_text(
        "# note\nRECAL_DRIVE_MEDRAG=https://x/folders/fromfile\nexport RECAL_DRIVE_TRIPCLICK='tc'\n")
    monkeypatch.setenv("RECAL_DRIVE_MEDRAG", "fromenv")
    env = data_fetch.drive_env(tmp_path)
    assert env["RECAL_DRIVE_MEDRAG"] == "fromenv" and env["RECAL_DRIVE_TRIPCLICK"] == "tc"
    assert data_fetch.missing_variables(["medrag", "tripclick", "triviaqa"], env) == ["RECAL_DRIVE_TRIVIAQA"]


def test_dry_run_and_list_report_set_and_missing_without_values(tmp_path, capsys, monkeypatch):
    from scripts import data_fetch
    clear_drive_env(monkeypatch)
    monkeypatch.setenv("RECAL_DRIVE_TRIPCLICK", "secretvalue")
    assert data_fetch.main(["--list", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "RECAL_DRIVE_TRIPCLICK set" in out and "RECAL_DRIVE_MEDRAG missing" in out
    monkeypatch.setattr(data_fetch, "list_drive", lambda _: [])
    data_fetch.main(["--tripclick", "--medrag", "--dry-run", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "RECAL_DRIVE_MEDRAG missing" in out and "secretvalue" not in out
