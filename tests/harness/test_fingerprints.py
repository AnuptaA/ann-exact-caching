"""Fingerprints: stored at the end of generation, verified at replay start, part of result reuse."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from harness import data, fingerprints
from harness.runner import manifest, replay, results
from tests.harness.test_replay import DRIVERS, ExactBackend, needs_driver
from harness.runner.prepare import prepare_dataset


@pytest.fixture()
def dataset(tmp_path):
    rng = np.random.default_rng(3)
    base = rng.normal(size=(300, 6)).astype(np.float32)
    official = rng.normal(size=(20, 6)).astype(np.float32)
    prepare_dataset(tmp_path, "toy", base, official, ExactBackend(base), ef_search=50, list_size=12,
                    n_groups=10, group_size=3, n_requests=120, zipf=(0.99,))
    return tmp_path


def test_stored_fingerprints_verify_and_cover_every_input(dataset):
    stored = fingerprints.verify(dataset / "toy", 0)
    assert {"base", "official", "universe", "streams", "answers", "answers_meta", "tune",
            "ground_truth"} <= set(stored)
    assert replay.DatasetDir(dataset, "toy").input_id == fingerprints.combine(stored)


def edit_streams(root: Path):
    z = dict(np.load(root / "streams_seed0.npz"))
    z["uniform"] = z["uniform"][::-1].copy()
    np.savez(root / "streams_seed0.npz", **z)


def edit_universe(root: Path):
    z = dict(np.load(root / "universe.npz"))
    z["lam"] = z["lam"] * 0.5
    np.savez(root / "universe.npz", **z)


def edit_base(root: Path):
    base = np.array(data.read_f32bin(root / "base.f32bin"))
    base[0, 0] += 1.0
    data.write_f32bin(root / "base.f32bin", base)


def edit_answers(root: Path):
    block = root / "answers" / "block_0000000.npz"
    z = dict(np.load(block))
    z["ids"] = z["ids"][:, ::-1].copy()
    np.savez(block, **z)


def edit_ground_truth(root: Path):
    next(iter((root / "gt").glob("chunk_*.npz"))).write_bytes(b"x")


@pytest.mark.parametrize("edit,changed", [(edit_streams, "streams"), (edit_universe, "universe"),
                                          (edit_base, "base"), (edit_answers, "answers"),
                                          (edit_ground_truth, "ground_truth")])
def test_any_changed_input_fails_verification_and_replay_start(dataset, edit, changed):
    root = dataset / "toy"
    edit(root)
    with pytest.raises(fingerprints.FingerprintMismatch, match=changed):
        fingerprints.verify(root, 0)
    with pytest.raises(fingerprints.FingerprintMismatch, match=changed):
        replay.DatasetDir(dataset, "toy")


def test_missing_fingerprints_file_is_an_error(dataset):
    (dataset / "toy" / fingerprints.FILE).unlink()
    with pytest.raises(FileNotFoundError, match="make q1-generate"):
        replay.DatasetDir(dataset, "toy")


def test_rebuilt_table_gives_a_new_input_id_and_new_cache_names(dataset):
    before = replay.DatasetDir(dataset, "toy").input_id
    edit_answers(dataset / "toy")
    fingerprints.store(dataset / "toy", 0)
    assert replay.DatasetDir(dataset, "toy").input_id != before


def test_reuse_key_changes_with_inputs_and_code():
    fp = {"base": 1, "official": 2, "universe": 3, "streams": 4, "answers": 5, "answers_meta": 6, "tune": 7}
    code = {"aker_tree": "a", "driver_src_tree": "d"}
    key = fingerprints.reuse_key(fp, code)
    assert key != fingerprints.reuse_key({**fp, "answers": 50}, code)
    assert key != fingerprints.reuse_key({**fp, "tune": 70}, code)
    assert key != fingerprints.reuse_key(fp, {**code, "aker_tree": "b"})
    assert key == fingerprints.reuse_key({**fp, "ground_truth": "ignored"}, code)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_drivers_built_from_other_sources_are_refused(tmp_path):
    repo = tmp_path / "repo"
    for sub in ("third_party/aker", "harness/cpp"):
        (repo / sub).mkdir(parents=True)
        (repo / sub / "f.txt").write_text("v1")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one")
    first = git(repo, "rev-parse", "HEAD")
    drivers = tmp_path / "drivers"
    drivers.mkdir()
    (drivers / "BUILT_FROM_COMMIT").write_text(first)
    assert fingerprints.code_fingerprints(repo, drivers)["aker_tree"]
    (repo / "third_party/aker/f.txt").write_text("patched")
    git(repo, "commit", "-q", "-am", "patch")
    with pytest.raises(fingerprints.FingerprintMismatch, match="rebuild the drivers"):
        fingerprints.code_fingerprints(repo, drivers)
    (repo / "docs.txt").write_text("unrelated")
    assert fingerprints.code_fingerprints(repo)["aker_tree"] == git(repo, "rev-parse", "HEAD:third_party/aker")


@needs_driver
def test_result_header_records_fingerprints_and_a_changed_table_forces_a_rerun(dataset, tmp_path):
    ds = replay.DatasetDir(dataset, "toy")
    cell = manifest.Cell("toy", "uniform", "aker", "paper", 10)
    path = replay.run_cell(cell, ds, DRIVERS, tmp_path / "r", tmp_path / "w")
    header = results.read_result(path)[0]
    code = fingerprints.code_fingerprints(replay.REPO, DRIVERS)
    assert header["fingerprints"] == ds.fingerprints and header["code"] == code
    assert header["reuse_key"] == fingerprints.reuse_key(ds.fingerprints, code)
    assert header["versions"]["numpy"] == np.__version__ and header["list_size"] == 12
    assert {"standard", "proximity"} == set(header["code"]["driver_sha256"])
    assert results.is_done(tmp_path / "r", cell.cell_id, header["commit"], header["reuse_key"])
    edit_answers(dataset / "toy")
    fingerprints.store(dataset / "toy", 0)
    new_key = fingerprints.reuse_key(replay.DatasetDir(dataset, "toy").fingerprints, code)
    assert not results.is_done(tmp_path / "r", cell.cell_id, header["commit"], new_key)
