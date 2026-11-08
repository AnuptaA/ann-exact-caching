"""Static libfaiss.a needs BLAS and LAPACK on the driver link line, after libfaiss.a.

The real Linux link is only exercised on an instance or in CI; here a fake FAISS prefix (a stub archive) lets CMake
configure for real, and the generated link line is read.
"""

import platform
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tests.harness.test_deploy_scripts import DEPLOY, ROOT, _fake_system, _setup

CPP = ROOT / "harness" / "cpp"
SAMPLE_CACHE = """\
BLA_VENDOR:UNINITIALIZED=OpenBLAS
BLAS_openblas_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libopenblas.so
BLAS_blas_LIBRARY:FILEPATH=BLAS_blas_LIBRARY-NOTFOUND
LAPACK_lapack_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/liblapack.so
OpenMP_CXX_FLAGS:STRING=-fopenmp
OpenMP_gomp_LIBRARY:FILEPATH=/usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so
CMAKE_BUILD_TYPE:STRING=Release
"""


def _bash(snippet: str, **env: str) -> subprocess.CompletedProcess:
    import os
    full = {**os.environ, **env}
    return subprocess.run(["bash", "-c", f". {DEPLOY}/lib.sh; . {DEPLOY}/pins.env; . {DEPLOY}/toolchain.sh; {snippet}"],
                          capture_output=True, text=True, env=full)


def test_cmake_links_blas_and_lapack_after_faiss_and_before_the_rest():
    text = (CPP / "CMakeLists.txt").read_text()
    assert "find_package(BLAS)" in text and "find_package(LAPACK)" in text and "FATAL_ERROR" in text
    assert "set(FAISS_DEPS ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})" in text
    link = re.search(r"target_link_libraries\(\$\{name\} PRIVATE[^)]*\)", text).group(0)
    assert link.index("${FAISS_LIBRARY}") < link.index("${FAISS_DEPS}") < link.index("${OMP_LIBS}") \
        < link.index("Boost::program_options")


def test_cache_blas_info_keeps_found_libraries_and_the_vendor():
    cache_text = SAMPLE_CACHE
    out = _bash("cache_blas_info /dev/stdin <<'EOF'\n" + cache_text + "EOF")
    lines = out.stdout.splitlines()
    assert "BLA_VENDOR=OpenBLAS" in lines
    assert "BLAS_openblas_LIBRARY=/usr/lib/x86_64-linux-gnu/libopenblas.so" in lines
    assert "LAPACK_lapack_LIBRARY=/usr/lib/x86_64-linux-gnu/liblapack.so" in lines
    assert not any("NOTFOUND" in ln or "OpenMP" in ln or "BUILD_TYPE" in ln for ln in lines)


def _lib(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("")
    return path


def test_blas_match_accepts_the_same_library_through_a_symlink_and_rejects_another(tmp_path):
    real = _lib(tmp_path, "libopenblasp-r0.3.26.so")
    (tmp_path / "libopenblas.so").symlink_to(real)
    other = _lib(tmp_path, "libblas.so.3")
    faiss = tmp_path / "BLAS_INFO"
    drivers = tmp_path / "blas_info.txt"
    faiss.write_text(f"BLA_VENDOR=OpenBLAS\nBLAS_openblas_LIBRARY={tmp_path}/libopenblas.so\n")
    drivers.write_text(f"BLA_VENDOR=OpenBLAS\nBLAS_LIBRARIES={real}\nLAPACK_LIBRARIES={real}\n")
    assert _bash(f"check_blas_match {faiss} {drivers}").returncode == 0
    drivers.write_text(f"BLAS_LIBRARIES={other}\nLAPACK_LIBRARIES={other}\n")
    bad = _bash(f"check_blas_match {faiss} {drivers}")
    assert bad.returncode != 0 and "different BLAS" in bad.stderr
    faiss.write_text("BLA_VENDOR=\n")
    empty = _bash(f"check_blas_match {faiss} {drivers}")
    assert empty.returncode == 0 and "no BLAS libraries recorded" in empty.stdout


def test_toolchain_lines_name_vendor_libraries_and_openmp(tmp_path):
    lib = _lib(tmp_path, "libopenblas.so")
    (tmp_path / "blas_info.txt").write_text(f"BLA_VENDOR=OpenBLAS\nBLAS_LIBRARIES={lib}\nLAPACK_LIBRARIES={lib}\n")
    (tmp_path / "CMakeCache.txt").write_text(SAMPLE_CACHE)
    out = _bash(f"toolchain_blas_lines {tmp_path}").stdout
    assert "blas_vendor OpenBLAS" in out and f"blas_lib {lib.resolve()}" in out
    assert "openmp CXX_FLAGS -fopenmp" in out and "openmp gomp_LIBRARY /usr/lib/gcc" in out
    (tmp_path / "blas_info.txt").write_text("shared libfaiss: BLAS and LAPACK are recorded in the library\n")
    assert "blas recorded in the shared libfaiss" in _bash(f"toolchain_blas_lines {tmp_path}").stdout


def _faiss_env(tmp_path: Path, marker: bool = True, cache: bool = True, blas_info: bool = False) -> dict:
    tools = tmp_path / "tools" / "q1"
    faiss = tools / "faiss"
    faiss.mkdir(parents=True)
    if marker:
        (faiss / ".installed-943d08bdad7946b22f56d040756669ee444dd681").write_text("")
    if cache:
        build = tools / "src" / "faiss" / "build"
        build.mkdir(parents=True)
        (build / "CMakeCache.txt").write_text(SAMPLE_CACHE)
    if blas_info:
        (faiss / "BLAS_INFO").write_text("BLA_VENDOR=OpenBLAS\n")
    return {"Q1_TOOLS": str(tools), "root": str(ROOT), "PYTHON_BIN": "python3"}


def test_existing_faiss_install_gets_its_blas_recorded_without_a_rebuild(tmp_path):
    env = _faiss_env(tmp_path)
    out = _bash("ensure_faiss", **env)
    assert out.returncode == 0, out.stderr
    assert "BLAS recorded from its build tree" in out.stdout and "cmake" not in out.stdout.lower().split("recorded")[0]
    info = (Path(env["Q1_TOOLS"]) / "faiss" / "BLAS_INFO").read_text()
    assert "BLA_VENDOR=OpenBLAS" in info and "libopenblas.so" in info


def test_faiss_without_a_record_or_build_tree_is_rebuilt_with_the_pinned_vendor(tmp_path):
    env = _faiss_env(tmp_path, cache=False)
    out = _bash("CMAKE_BIN=cmake; ensure_faiss", Q1_DRYRUN="1", **env)
    assert out.returncode == 0, out.stderr
    assert "no BLAS record and no build tree: rebuilding" in out.stdout and "-DBLA_VENDOR=OpenBLAS" in out.stdout
    assert "record the FAISS BLAS libraries" in out.stdout


def test_faiss_with_a_record_is_left_alone(tmp_path):
    out = _bash("ensure_faiss", **_faiss_env(tmp_path, blas_info=True))
    assert out.returncode == 0 and "already built" in out.stdout and "-DBLA_VENDOR" not in out.stdout


def test_recording_fails_loudly_when_the_cache_has_no_blas(tmp_path):
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
    out = _bash(f"record_faiss_blas {cache} {tmp_path}/BLAS_INFO")
    assert out.returncode != 0 and "libopenblas-dev" in out.stderr


def test_setup_installs_the_blas_dev_package_and_pins_the_vendor(tmp_path):
    text = _setup(_fake_system(tmp_path, "1.83.0.1ubuntu2", "3.28.3", "13.2.0")).stdout
    assert "libopenblas-dev" in text and "-DBLA_VENDOR=OpenBLAS" in text


def _fake_pg(tmp_path: Path, listing: str) -> dict:
    bin_dir = tmp_path / "pgbin"
    bin_dir.mkdir()
    calls = tmp_path / "sudo_calls"
    for name, body in (("pg_lsclusters", f"printf '%s' '{listing}'"), ("sudo", f'echo "$@" >> {calls}')):
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    import os
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}", "calls": str(calls)}


def test_the_packaged_cluster_is_dropped_only_when_it_exists(tmp_path):
    env = _fake_pg(tmp_path, "16 main 5432 online postgres /var/lib/postgresql/16/main\n")
    out = _bash("drop_packaged_cluster", PATH=env["PATH"])
    assert out.returncode == 0 and "pg_dropcluster --stop 16 main" in Path(env["calls"]).read_text()
    absent = _fake_pg(tmp_path / "b", "") if (tmp_path / "b").mkdir() is None else {}
    out = _bash("drop_packaged_cluster", PATH=absent["PATH"])
    assert out.returncode == 0 and "no packaged cluster 16/main to drop" in out.stdout
    assert "specified cluster does not exist" not in out.stdout + out.stderr
    assert not Path(absent["calls"]).exists()


def _brew_prefix(name: str) -> str | None:
    if shutil.which("brew") is None:
        return None
    out = subprocess.run(["brew", "--prefix", name], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 and Path(out.stdout.strip()).exists() else None


def _make_stub_faiss(tmp_path: Path, static: bool) -> Path:
    prefix = tmp_path / ("faiss_static" if static else "faiss_shared")
    (prefix / "include" / "faiss").mkdir(parents=True)
    (prefix / "include" / "faiss" / "Index.h").write_text("")
    (prefix / "lib").mkdir()
    (tmp_path / "stub.c").write_text("int faiss_stub(void) { return 0; }\n")
    obj = tmp_path / "stub.o"
    subprocess.run(["cc", "-fPIC", "-c", str(tmp_path / "stub.c"), "-o", str(obj)], check=True)
    if static:
        subprocess.run(["ar", "rcs", str(prefix / "lib" / "libfaiss.a"), str(obj)], check=True)
    else:
        suffix = "dylib" if platform.system() == "Darwin" else "so"
        subprocess.run(["cc", "-shared", str(obj), "-o", str(prefix / "lib" / f"libfaiss.{suffix}")], check=True)
    return prefix


def _configure(tmp_path: Path, prefix: Path, *extra: str) -> subprocess.CompletedProcess:
    cmake = shutil.which("cmake")
    if cmake is None or shutil.which("cc") is None or shutil.which("ar") is None:
        pytest.skip("needs cmake, cc and ar")
    args = [cmake, "-S", str(CPP), "-B", str(tmp_path / "build"), f"-DFAISS_ROOT={prefix}", *extra]
    if platform.system() == "Darwin":
        boost, omp = _brew_prefix("boost"), _brew_prefix("libomp")
        if boost is None or omp is None:
            pytest.skip("needs Homebrew boost and libomp")
        args += [f"-DBOOST_ROOT={boost}", f"-DLIBOMP_ROOT={omp}"]
    out = subprocess.run(args, capture_output=True, text=True)
    if "Could NOT find Boost" in out.stdout + out.stderr:
        pytest.skip("Boost >= 1.80 not available")
    return out


def test_configure_with_a_static_faiss_puts_blas_and_lapack_after_it_on_the_link_line(tmp_path):
    out = _configure(tmp_path, _make_stub_faiss(tmp_path, static=True))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "FAISS is static: linking BLAS" in out.stdout
    info = (tmp_path / "build" / "blas_info.txt").read_text()
    link = (tmp_path / "build" / "CMakeFiles" / "ak_replay_standard.dir" / "link.txt").read_text()
    faiss_at = link.index("libfaiss.a")
    tokens = [tok for line in info.splitlines() if "LIBRARIES" in line for tok in line.split("=", 1)[1].split(";") if tok]
    assert tokens, info
    for token in tokens:
        if token.endswith(".framework"):             # macOS writes frameworks as "-framework Name"
            token = f"-framework {Path(token).stem}"
        assert token in link and link.index(token, faiss_at) > faiss_at, token


def test_configure_with_a_shared_faiss_does_not_require_blas(tmp_path):
    out = _configure(tmp_path, _make_stub_faiss(tmp_path, static=False))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "shared libfaiss" in (tmp_path / "build" / "blas_info.txt").read_text()


def test_configure_with_a_static_faiss_and_no_matching_blas_fails_loudly(tmp_path):
    out = _configure(tmp_path, _make_stub_faiss(tmp_path, static=True), "-DBLA_VENDOR=NoSuchVendor")
    assert out.returncode != 0
    assert "libfaiss.a needs BLAS and LAPACK" in out.stdout + out.stderr
    assert "libopenblas-dev" in out.stdout + out.stderr


def test_a_static_archive_needs_blas_after_it_on_the_link_line(tmp_path):
    """The failure seen on Ubuntu: undefined sgemm_ from a static archive linked without BLAS."""
    if shutil.which("cc") is None or shutil.which("ar") is None:
        pytest.skip("needs cc and ar")
    blas = ["-framework", "Accelerate"] if platform.system() == "Darwin" else ["-lopenblas"]
    (tmp_path / "lib.c").write_text("void sgemm_(void);\nint use_blas(void) { sgemm_(); return 0; }\n")
    (tmp_path / "main.c").write_text("int use_blas(void);\nint main(void) { return use_blas(); }\n")
    subprocess.run(["cc", "-c", str(tmp_path / "lib.c"), "-o", str(tmp_path / "lib.o")], check=True)
    subprocess.run(["ar", "rcs", str(tmp_path / "libstub.a"), str(tmp_path / "lib.o")], check=True)
    link = ["cc", str(tmp_path / "main.c"), str(tmp_path / "libstub.a")]
    without = subprocess.run(link + ["-o", str(tmp_path / "a.out")], capture_output=True, text=True)
    assert without.returncode != 0 and "sgemm_" in without.stderr
    with_blas = subprocess.run(link + blas + ["-o", str(tmp_path / "a.out")], capture_output=True, text=True)
    if with_blas.returncode != 0 and platform.system() != "Darwin":
        pytest.skip("no linkable BLAS on this machine")
    assert with_blas.returncode == 0, with_blas.stderr
