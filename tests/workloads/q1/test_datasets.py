"""Tests for dataset helpers on tiny fake data and a local range-capable server."""

import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from workloads.q1 import cli
from workloads.q1.datasets import common, esci, sift10m, sphere, spacev, triviaqa, triviaqa_evidence


class _Handler(BaseHTTPRequestHandler):
    files = {}
    honor_ranges = True

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        data = self.files[self.path]
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()

    def do_GET(self):
        data = self.files[self.path]
        rng = self.headers.get("Range")
        if rng and self.honor_ranges:
            start, end = (int(x) for x in rng.split("=")[1].split("-"))
            body = data[start:end + 1]
            self.send_response(206)
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server():
    _Handler.files, _Handler.honor_ranges = {}, True
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield _Handler, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_download_slice_resumes_and_is_atomic(server, tmp_path):
    handler, url = server
    handler.files["/f"] = bytes(range(200))
    dest = tmp_path / "out.bin"
    part = tmp_path / "out.bin.part"
    part.write_bytes(b"HD" + bytes(range(10, 40)))            # interrupted earlier
    common.download_slice(f"{url}/f", dest, 10, 110, header=b"HD", chunk_bytes=16)
    assert dest.read_bytes() == b"HD" + bytes(range(10, 110)) and not part.exists()
    assert common.download_slice(f"{url}/f", dest, 10, 110, header=b"HD") == dest  # no-op


def test_fetch_range_rejects_servers_without_ranges(server):
    handler, url = server
    handler.files["/f"] = b"x" * 50
    handler.honor_ranges = False
    with pytest.raises(OSError):
        common.fetch_range(f"{url}/f", 0, 10)


def test_sift_slice_reads_only_the_first_rows(server, tmp_path):
    handler, url = server
    full = np.arange(20 * 128, dtype=np.uint8).reshape(20, 128)
    handler.files["/base"] = struct.pack("<ii", 20, 128) + full.tobytes()
    handler.files["/query"] = struct.pack("<ii", 3, 128) + full[:3].tobytes()
    base, query = sift10m.fetch(tmp_path, n_base=7, base_url=f"{url}/base", query_url=f"{url}/query")
    b, q = sift10m.load(tmp_path)
    assert b.shape == (7, 128) and np.array_equal(b, full[:7]) and q.shape == (3, 128)
    with pytest.raises(ValueError):
        sift10m.verify(b, q)                    # fake sizes are not the real ones
    with pytest.raises(ValueError):
        sift10m.fetch(tmp_path / "x", n_base=99, base_url=f"{url}/base", query_url=f"{url}/query")


def test_spacev_and_sphere_verify_dtypes_and_shapes(tmp_path):
    base = np.zeros((6, spacev.DIM), dtype=np.int8)
    queries = np.zeros((spacev.N_QUERY, spacev.DIM), dtype=np.int8)
    spacev.verify(base, queries, n_base=6)
    with pytest.raises(ValueError):
        spacev.verify(base.astype(np.float32), queries, n_base=6)
    sphere.verify(np.zeros((4, sphere.DIM)), n_base=4)
    with pytest.raises(ValueError):
        sphere.verify(np.zeros((4, sphere.DIM), dtype=np.float32), n_base=4)
    with pytest.raises(OSError):
        common.check_free_disk(tmp_path, 10 ** 18)


def test_esci_text_queries_and_counts():
    assert esci.product_text("Shirt ", "Soft cotton\nsecond line") == "Shirt. Soft cotton"
    assert esci.product_text("Shirt", None) == "Shirt"
    assert esci.product_text(None, float("nan")) == ""
    frame = pd.DataFrame({"query_id": [3, 1, 2, 1, 4], "query": ["b", "a", "a", "a", "c"]})
    texts, ids = esci.distinct_queries(frame)
    assert texts == ["a", "b", "c"] and ids.tolist() == [1, 3, 4]
    counts = esci.check_counts(100, 50, 48, expected_products=100, expected_query_ids=50)
    assert counts["query_texts"] == 48
    with pytest.raises(ValueError):
        esci.check_counts(10, 50, 48, expected_products=100, expected_query_ids=50)


def test_esci_embedding_match_check():
    texts = [f"t{i}" for i in range(30)]
    vecs = np.random.default_rng(0).normal(size=(30, 4)).astype(np.float32)
    lookup = {t: v for t, v in zip(texts, vecs)}
    encode = lambda batch: np.array([lookup[t] for t in batch])
    assert esci.verify_embeddings_match(vecs, texts, encode, n=10) > 0.9999
    with pytest.raises(ValueError):
        esci.verify_embeddings_match(vecs[::-1].copy(), texts, encode, n=10)
    with pytest.raises(ValueError):
        esci.verify_embeddings_match(vecs[:5], texts, encode)


def test_triviaqa_dedup_encode_and_counts():
    assert triviaqa.dedup_questions([" a ", "a", "b", "", "b"]) == ["a", "b"]
    encode = lambda batch: np.ones((len(batch), triviaqa.DIM)) * 3.0
    out = triviaqa.encode_questions(["q"] * 5, encode, batch_size=2)
    assert out.shape == (5, triviaqa.DIM) and out.dtype == np.float32 and out[0, 0] == 3.0  # raw
    with pytest.raises(ValueError):
        triviaqa.encode_questions(["q"], lambda b: np.ones((len(b), 3)))
    assert triviaqa.encode_questions([], encode).shape == (0, triviaqa.DIM)


def test_answer_lift_is_high_for_a_matching_space():
    rng = np.random.default_rng(0)
    passages = [f"passage about topic_{i % 50:02d}_" for i in range(500)]
    centers = rng.normal(size=(50, 8)).astype(np.float32) * 5
    pv = centers[np.arange(500) % 50] + rng.normal(size=(500, 8)).astype(np.float32) * 0.1
    questions = centers[:20] + rng.normal(size=(20, 8)).astype(np.float32) * 0.1
    answers = [[f"topic_{i:02d}_"] for i in range(20)]
    good = triviaqa_evidence.answer_lift(questions, pv, passages, answers, top=10, n_base=200)
    bad = triviaqa_evidence.answer_lift(rng.normal(size=(20, 8)).astype(np.float32) * 5, pv,
                                        passages, answers, top=10, n_base=200)
    assert good["precision"] > 0.9 and good["lift"] > bad["lift"] * 3
    assert "lift" in triviaqa_evidence.format_evidence({"enc": good})
    with pytest.raises(ValueError):
        triviaqa_evidence.answer_lift(questions, pv, passages, [["ab"]] * 20)


def test_cli_universe_streams_groundtruth(tmp_path, capsys):
    rng = np.random.default_rng(0)
    official = rng.normal(size=(40, 4)).astype(np.float32)
    np.save(tmp_path / "official.npy", official)
    cli.main(["universe", "--official", str(tmp_path / "official.npy"), "--out",
              str(tmp_path / "u.npz"), "--n-groups", "20", "--group-size", "5"])
    cli.main(["streams", "--universe", str(tmp_path / "u.npz"), "--out", str(tmp_path / "s.npz"),
              "--n-requests", "500", "--zipf", "0.6", "--no-locality"])
    out = capsys.readouterr().out
    assert "zipf_0.60" in out and "no_locality" in out and "history 150" in out
    z = np.load(tmp_path / "s.npz")
    assert len(z["no_locality"]) == 40 and len(z["zipf_0.60"]) == 500
    np.save(tmp_path / "base.npy", rng.normal(size=(60, 4)).astype(np.float32))
    cli.main(["groundtruth", "--base", str(tmp_path / "base.npy"), "--queries",
              str(tmp_path / "official.npy"), "--out-dir", str(tmp_path / "gt"), "--k", "10"])
    assert len(list((tmp_path / "gt").glob("chunk_*.npz"))) == 1
    with pytest.raises(SystemExit):
        cli.main(["--help"])


def test_fetch_encoder_extracts_the_archive_layout_and_rejects_escapes(server, tmp_path):
    import io
    import tarfile

    def make(names):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            top = tarfile.TarInfo("dpr_web_query_encoder_hf")
            top.type = tarfile.DIRTYPE
            tar.addfile(top)
            for name in names:
                info = tarfile.TarInfo(name)
                info.size = 2
                tar.addfile(info, io.BytesIO(b"{}"))
        return buf.getvalue()

    handler, url = server
    handler.files["/enc"] = make(["dpr_web_query_encoder_hf/config.json"])
    old = triviaqa.ENCODER_URL
    triviaqa.ENCODER_URL = f"{url}/enc"
    try:
        out = triviaqa.fetch_encoder(tmp_path / "ok")
        assert (out / "config.json").exists() and triviaqa.fetch_encoder(tmp_path / "ok") == out
        handler.files["/enc"] = make(["../escape.txt"])
        with pytest.raises(ValueError):
            triviaqa.fetch_encoder(tmp_path / "bad")
        handler.files["/enc"] = make(["other/config.json"])
        with pytest.raises(ValueError):
            triviaqa.fetch_encoder(tmp_path / "bad2")
    finally:
        triviaqa.ENCODER_URL = old


def test_esci_build_writes_products_and_distinct_queries(tmp_path):
    pytest.importorskip("pyarrow")
    raw = tmp_path / "raw"
    raw.mkdir()
    n_products = 100
    pd.DataFrame({"product_id": [f"p{i}" for i in range(n_products)],
                  "product_title": [f"title {i}" for i in range(n_products)],
                  "product_bullet_point": [f"first {i}\nsecond" if i % 2 else None for i in range(n_products)]}
                 ).to_parquet(raw / esci.RAW_FILES[0])
    pd.DataFrame({"query_id": [3, 1, 2, 1, 4, 5], "query": ["b", "a", "a", "a", "c", "d"]}
                 ).to_parquet(raw / esci.RAW_FILES[1])
    seen = []

    def encode(texts):
        seen.append(list(texts))
        return np.tile(np.arange(len(texts), dtype=np.float32)[:, None], (1, 4))

    old = esci.check_counts
    esci.check_counts = lambda p, q, t, **kw: old(p, q, t, expected_products=n_products, expected_query_ids=5)
    try:
        counts = esci.build(raw, tmp_path / "out", encode)
    finally:
        esci.check_counts = old
    assert counts == {"products": 100, "query_ids": 5, "query_texts": 4}
    assert seen[0] == ["a", "b", "c", "d"] and seen[1][0] == "title 0" and seen[1][1] == "title 1. first 1"
    assert np.load(tmp_path / "out" / "esci_products.npy").shape == (100, 4)
    assert np.load(tmp_path / "out" / "esci_query_ids.npy").tolist() == [1, 3, 4, 5]
