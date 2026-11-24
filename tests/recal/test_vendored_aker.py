import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AKER = ROOT / "third_party/aker"
SOURCES = [p for p in AKER.rglob("*") if p.suffix in {".c", ".cc", ".h", ".hh", ".cpp"} and "extern" not in p.parts]


def test_the_library_leaves_out_the_ycsb_sequence_generator():
    text = (AKER / "CMakeLists.txt").read_text()
    glob_at = text.index("file(GLOB_RECURSE AKER_SRC_FILES")
    filter_at = text.index('list(FILTER AKER_SRC_FILES EXCLUDE REGEX "ak_sequence_gen")')
    assert glob_at < filter_at < text.index("add_library(aker SHARED")
    assert "ak_sequence_gen" in (AKER / "PATCHES.md").read_text()


def test_nothing_else_in_the_vendored_tree_uses_it():
    users = [p.name for p in SOURCES if p.name not in ("ak_sequence_gen.cc", "ak_sequence_gen.hh")
             and re.search(r"ak_sequence_gen|YcsbSeqGenerator|YCSB-C", p.read_text(errors="replace"))]
    assert users == []


def test_the_shim_and_the_replay_build_do_not_use_it_either():
    shim = (ROOT / "simulations/approximate_hits/aker_real/aker_shim.cc").read_text()
    assert "YcsbSeqGenerator" not in shim and "ak_sequence_gen" not in shim
    assert 'EXCLUDE REGEX "ak_sequence_gen"' in (ROOT / "harness/cpp/CMakeLists.txt").read_text()
