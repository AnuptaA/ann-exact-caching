"""Scale-only behavior without large data: offsets past 2^31, header sizes, dtypes, range reads."""

import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from harness import data
from workloads.q1.datasets import common, sift10m

BIG = 2 ** 31


def pattern(start: int, length: int) -> bytes:
    offsets = np.arange(start, start + length, dtype=np.uint64)
    return ((offsets * np.uint64(7) + np.uint64(3)) % np.uint64(251)).astype(np.uint8).tobytes()


class _VirtualHandler(BaseHTTPRequestHandler):
    """Serves virtual files: path -> (size, header bytes). Records ranges; fails on over-read."""

    files: dict = {}
    ranges: list = []
    over_reads: list = []
    cut_once: set = set()

    def log_message(self, *args):
        pass

    def _content(self, path, start, length):
        size, header = self.files[path]
        raw = bytearray(pattern(start, length))
        for i, b in enumerate(header):
            if start <= i < start + length:
                raw[i - start] = b
        return bytes(raw)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(self.files[self.path][0]))
        self.end_headers()

    def do_GET(self):
        size, _ = self.files[self.path]
        start, end = (int(x) for x in self.headers["Range"].split("=")[1].split("-"))
        self.ranges.append((self.path, start, end + 1))
        if end >= size:
            self.over_reads.append((self.path, start, end))
            self.send_response(416)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = self._content(self.path, start, end - start + 1)
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if (self.path, start) in self.cut_once:
            self.cut_once.discard((self.path, start))
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.close_connection = True
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture()
def virtual_server():
    _VirtualHandler.files, _VirtualHandler.ranges = {}, []
    _VirtualHandler.over_reads, _VirtualHandler.cut_once = [], set()
    httpd = HTTPServer(("127.0.0.1", 0), _VirtualHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield _VirtualHandler, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def covered(ranges, path):
    spans = sorted((s, e) for p, s, e in ranges if p == path)
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def test_download_slice_past_2_gib_reads_exactly_the_requested_bytes(virtual_server, tmp_path):
    handler, url = virtual_server
    handler.files["/big"] = (5 * BIG, b"")
    start, stop = BIG + 5, BIG + 5 + 100
    dest = common.download_slice(f"{url}/big", tmp_path / "s.bin", start, stop, chunk_bytes=32)
    assert dest.read_bytes() == pattern(start, 100)
    assert covered(handler.ranges, "/big") == [[start, stop]] and not handler.over_reads


def test_content_length_beyond_int32(virtual_server):
    handler, url = virtual_server
    handler.files["/huge"] = (3 * BIG + 17, b"")
    assert common.content_length(f"{url}/huge") == 3 * BIG + 17


def test_sift_slice_requests_exactly_header_plus_rows_and_never_over_reads(virtual_server, tmp_path,
                                                                           monkeypatch):
    handler, url = virtual_server
    n_total, dim, n_base = 1_000_000_000, 128, 40
    total = 8 + n_total * dim
    handler.files["/base"] = (total, struct.pack("<ii", n_total, dim))
    handler.files["/query"] = (8 + 3 * dim, struct.pack("<ii", 3, dim))
    monkeypatch.setattr(common.download_slice, "__defaults__", (b"", 1000))
    base, query = sift10m.fetch(tmp_path, n_base=n_base, base_url=f"{url}/base", query_url=f"{url}/query")
    assert not handler.over_reads
    slice_spans = [(s, e) for p, s, e in handler.ranges if p == "/base" and not (s == 0 and e == 8)]
    assert covered([("/base", s, e) for s, e in slice_spans], "/base") == [[8, 8 + n_base * dim]]
    assert covered(handler.ranges, "/base")[0][0] == 0
    mapped = common.memmap_bin(base, np.uint8)
    assert mapped.shape == (n_base, dim)
    expected = np.frombuffer(pattern(8, n_base * dim), dtype=np.uint8).reshape(n_base, dim)
    assert np.array_equal(mapped, expected)


def test_dropped_connection_mid_body_is_retried(virtual_server, tmp_path, monkeypatch):
    handler, url = virtual_server
    handler.files["/f"] = (10_000, b"")
    handler.cut_once.add(("/f", 4000))
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    dest = common.download_slice(f"{url}/f", tmp_path / "o.bin", 0, 10_000, chunk_bytes=4000)
    assert dest.read_bytes() == pattern(0, 10_000)
    assert [r for r in handler.ranges if r[1] == 4000].__len__() == 2


def test_header_beyond_int32_fails_loudly():
    with pytest.raises(struct.error):
        common.bin_header(BIG, 128)
    assert common.bin_header(BIG - 1, 128) == struct.pack("<ii", BIG - 1, 128)


def test_f32bin_size_must_match_header(tmp_path):
    path = tmp_path / "x.f32bin"
    data.write_f32bin(path, np.arange(12, dtype=np.float32).reshape(4, 3))
    assert data.read_f32bin(path).shape == (4, 3)
    with path.open("ab") as f:
        f.write(b"\0\0\0\0")
    with pytest.raises(ValueError, match="header"):
        data.read_f32bin(path)
    short = tmp_path / "short.f32bin"
    short.write_bytes(b"\1\0")
    with pytest.raises(ValueError, match="header"):
        data.read_f32bin(short)


def test_memmap_bin_validates_size_for_each_dtype(tmp_path):
    rows = np.array([[0, 127, 128, 255], [1, 2, 3, 4]], dtype=np.uint8)
    path = tmp_path / "a.u8bin"
    path.write_bytes(common.bin_header(2, 4) + rows.tobytes())
    assert np.array_equal(common.memmap_bin(path, np.uint8), rows)
    path.write_bytes(common.bin_header(2, 4) + rows.tobytes()[:-1])
    with pytest.raises(ValueError, match="header"):
        common.memmap_bin(path, np.uint8)


@pytest.mark.parametrize("dtype,values", [
    (np.uint8, [0, 127, 128, 255]),
    (np.int8, [-128, -1, 0, 127]),
])
def test_f32_conversion_keeps_signed_and_high_unsigned_values(tmp_path, dtype, values):
    rows = np.array([values, values[::-1]] * 3, dtype=dtype)
    src = tmp_path / "s.bin"
    src.write_bytes(common.bin_header(*rows.shape) + rows.tobytes())
    out = data.write_f32bin(tmp_path / "o.f32bin", common.memmap_bin(src, dtype), chunk_rows=4)
    back = data.read_f32bin(out)
    assert back.dtype == np.float32 and np.array_equal(back, rows.astype(np.float32))


def test_offsets_use_64_bit_arithmetic_past_2_gib():
    row, dim = 10_000_000, 768
    offset = 8 + np.int64(row) * dim * 4
    assert offset > BIG and offset == 8 + row * dim * 4
    ids = np.array([BIG + 7], dtype=np.int64)
    assert (ids * dim * 4)[0] == (BIG + 7) * dim * 4
