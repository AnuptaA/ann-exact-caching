"""Download verification: published checksums, trust on first use, sidecars, quarantine."""

import base64
import hashlib
from email.message import Message

import pytest

from workloads.q1.datasets import common

MIB = 2 ** 20


def headers(**items) -> Message:
    msg = Message()
    for key, value in items.items():
        msg[key.replace("_", "-")] = value
    return msg


def test_huggingface_headers_give_sha256_or_git_sha1():
    sha = "ab" * 32
    got = common.checksum_from_headers("huggingface.co", headers(x_linked_etag=f'"{sha}"', x_linked_size="99"))
    assert got == ("sha256", sha, 99)
    blob = "cd" * 20
    assert common.checksum_from_headers("huggingface.co", headers(x_linked_etag=f'"{blob}"'))[:2] == \
        ("git-sha1", blob)
    assert common.checksum_from_headers("huggingface.co", headers(etag='"abc"')) is None


def test_azure_md5_and_s3_etags():
    md5 = base64.b64encode(hashlib.md5(b"x").digest()).decode()
    assert common.checksum_from_headers("c.web.core.windows.net", headers(Content_MD5=md5, Content_Length="5")) \
        == ("md5-b64", md5, 5)
    hexmd5 = hashlib.md5(b"x").hexdigest()
    assert common.checksum_from_headers("dl.fbaipublicfiles.com", headers(ETag=f'"{hexmd5}"'))[:2] == \
        ("md5", hexmd5)
    assert common.checksum_from_headers("dl.fbaipublicfiles.com", headers(ETag=f'"{hexmd5}-49"'))[0] == \
        "s3-multipart"
    assert common.checksum_from_headers("example.org", headers(ETag=f'"{hexmd5}"')) is None


def test_lfs_pointer_parsing():
    text = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 1108857465\n"
    assert common.parse_lfs_pointer(text) == ("0" * 64, 1108857465)
    assert common.parse_lfs_pointer("PAR1 not a pointer") is None


def test_digests_match_independent_computations(tmp_path):
    body = bytes(range(256)) * 100
    path = tmp_path / "f"
    path.write_bytes(body)
    assert common.digest_of(path, "sha256") == hashlib.sha256(body).hexdigest()
    assert common.digest_of(path, "md5") == hashlib.md5(body).hexdigest()
    assert common.digest_of(path, "md5-b64") == base64.b64encode(hashlib.md5(body).digest()).decode()
    assert common.digest_of(path, "git-sha1") == hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


def test_s3_multipart_etag_for_8_mib_parts(tmp_path):
    body = bytes(range(251)) * (17 * MIB // 251 + 1)
    body = body[:17 * MIB]
    path = tmp_path / "f"
    path.write_bytes(body)
    parts = [body[i:i + 8 * MIB] for i in range(0, len(body), 8 * MIB)]
    etag = hashlib.md5(b"".join(hashlib.md5(p).digest() for p in parts)).hexdigest() + f"-{len(parts)}"
    assert common.digest_of(path, "s3-multipart", etag) == etag
    path.write_bytes(body[:-1] + b"\xff")
    assert common.digest_of(path, "s3-multipart", etag) != etag


def test_verified_file_gets_a_sidecar_and_reruns_skip_hashing(tmp_path, monkeypatch):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    good = hashlib.sha256(b"hello").hexdigest()
    monkeypatch.setattr(common, "published_checksum", lambda url: ("sha256", good, 5))
    record = common.ensure_verified(path, "http://x/f.bin")
    assert record["digest"] == good and (tmp_path / "f.bin.verified.json").exists()
    monkeypatch.setattr(common, "published_checksum", lambda url: pytest.fail("network used on rerun"))
    monkeypatch.setattr(common, "digest_of", lambda *a, **k: pytest.fail("hashed again"))
    assert common.ensure_verified(path, "http://x/f.bin") == record


def test_mismatch_is_quarantined_and_raises(tmp_path, monkeypatch):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    monkeypatch.setattr(common, "published_checksum", lambda url: ("sha256", "0" * 64, 5))
    with pytest.raises(common.ChecksumError, match="f.bin.corrupt"):
        common.ensure_verified(path, "http://x/f.bin")
    assert not path.exists() and (tmp_path / "f.bin.corrupt").exists()
    assert not (tmp_path / "f.bin.verified.json").exists()


def test_wrong_published_size_is_an_error_even_with_a_matching_hash(tmp_path, monkeypatch):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    monkeypatch.setattr(common, "published_checksum",
                        lambda url: ("sha256", hashlib.sha256(b"hello").hexdigest(), 6))
    with pytest.raises(common.ChecksumError, match="size"):
        common.ensure_verified(path, "http://x/f.bin")


def test_trust_on_first_use_detects_a_changed_file(tmp_path, monkeypatch):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    monkeypatch.setattr(common, "published_checksum", lambda url: None)
    first = common.ensure_verified(path, "http://x/f.bin")
    assert first["kind"] == "sha256-tofu"
    path.write_bytes(b"jello")
    with pytest.raises(common.ChecksumError, match="sha256-tofu"):
        common.ensure_verified(path, "http://x/f.bin")


def test_slices_skip_published_checksums(tmp_path, monkeypatch):
    path = tmp_path / "s.bin"
    path.write_bytes(b"slice")
    monkeypatch.setattr(common, "published_checksum", lambda url: pytest.fail("whole-file checksum used"))
    assert common.ensure_verified(path, "http://x/full", use_published=False)["kind"] == "sha256-tofu"


def test_triviaqa_encode_cannot_fall_back_to_the_hub_tokenizer():
    from workloads.q1 import cli
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["triviaqa-encode", "--questions", "q.npy", "--encoder-dir", "e", "--out", "o.npy"])
