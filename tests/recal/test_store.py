import json

import pytest

from recal import store
from recal.measure import environment, jobs_util
from scripts import store as store_cli


def make_store(tmp_path):
    return store.Store(store.LocalBackend(tmp_path / "bucket"))


def test_push_pull_roundtrip_with_checksums(tmp_path):
    s = make_store(tmp_path)
    src = tmp_path / "a.bin"
    src.write_bytes(b"x" * 1000)
    assert s.push_file("artifacts/d/a.bin", src) is True
    assert s.push_file("artifacts/d/a.bin", src) is False
    dest = tmp_path / "out" / "a.bin"
    assert s.pull_file("artifacts/d/a.bin", dest) is True and dest.read_bytes() == src.read_bytes()
    assert s.pull_file("artifacts/d/a.bin", dest) is False
    rows = s.ls("artifacts/d")
    assert [r["key"] for r in rows] == ["artifacts/d/a.bin"] and rows[0]["size"] == 1000


def test_a_corrupted_object_fails_the_pull_and_leaves_no_file(tmp_path):
    s = make_store(tmp_path)
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc" * 100)
    s.push_file("k/a.bin", src)
    (tmp_path / "bucket" / "k" / "a.bin").write_bytes(b"abc" * 99 + b"abd")
    dest = tmp_path / "out.bin"
    with pytest.raises(store.ChecksumError):
        s.pull_file("k/a.bin", dest)
    assert not dest.exists() and not (tmp_path / ".out.bin.part").exists()


def test_an_object_without_its_checksum_is_not_there(tmp_path):
    s = make_store(tmp_path)
    (tmp_path / "bucket" / "k").mkdir(parents=True)
    (tmp_path / "bucket" / "k" / "a.bin").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        s.pull_file("k/a.bin", tmp_path / "o")


def test_an_interrupted_push_resumes_and_a_changed_file_is_replaced(tmp_path):
    s = make_store(tmp_path)
    d = tmp_path / "res"
    d.mkdir()
    for i in range(5):
        (d / f"f{i}.json").write_text(json.dumps({"i": i}))
    assert s.push_tree(d, "results/q1/abc") == 5
    (tmp_path / "bucket" / "results/q1/abc/f2.json.sha256").unlink()
    (d / "f4.json").write_text("changed")
    assert s.push_tree(d, "results/q1/abc") == 2
    out = tmp_path / "pulled"
    assert s.pull_prefix("results/q1/abc", out) == 5 and (out / "f4.json").read_text() == "changed"
    assert s.pull_prefix("results/q1/abc", out) == 0


def test_cli_push_pull_ls_and_env_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECAL_STORE", str(tmp_path / "b"))
    f = tmp_path / "x.txt"
    f.write_text("hello")
    store_cli.main(["push", str(f), "artifacts/t/x.txt"])
    store_cli.main(["ls", "artifacts"])
    assert "artifacts/t/x.txt" in capsys.readouterr().out
    store_cli.main(["pull", "artifacts/t/x.txt", str(tmp_path / "y.txt")])
    assert (tmp_path / "y.txt").read_text() == "hello"


def test_environment_is_described_and_stamped_on_results():
    env = environment.describe()
    assert {"instance_type", "cpu_model", "cores", "memory_gb", "disk_type"} <= set(env)
    assert env["cores"] and env["instance_type"] == "unknown" or isinstance(env["instance_type"], str)
    result = jobs_util.stamp_result({}, argv=["--x", "1"])
    assert result["environment"]["cores"] == env["cores"] and result["code_commit"]
