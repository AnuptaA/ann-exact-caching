import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ec2_bootstrap.sh"
COMPONENTS = "program_options log log_setup date_time thread system filesystem regex atomic chrono".split()


def script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return path


class World:
    """A fake repository, harness toolchain, system Boost and third-party prefix."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.root = tmp / "repo"
        self.tools = tmp / "opt"
        self.usr = tmp / "usr"
        self.tp = tmp / "tp"
        self.stubs = tmp / "stubs"
        script(self.root / ".venv-build/bin/cmake", 'echo "cmake version 3.27.9"')
        (self.root / "third_party/aker").mkdir(parents=True)
        (self.root / "third_party/aker/CMakeLists.txt").touch()
        (self.usr / "include/boost").mkdir(parents=True)
        (self.usr / "include/boost/version.hpp").write_text('#define BOOST_LIB_VERSION "1_83"\n')
        for comp in COMPONENTS:
            self.touch(self.usr / f"lib/libboost_{comp}.so")
        self.touch(self.usr / "lib/libopenblas.so")
        self.touch(tmp / "libgomp.so")
        script(self.stubs / "g++", 'echo "g++ (Ubuntu 13.2.0) 13.2.0"')
        script(self.stubs / "gcc", f'echo "{tmp}/libgomp.so"')

    @staticmethod
    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def faiss_ready(self) -> None:
        pins = (ROOT / "harness/deploy/pins.env").read_text()
        commit = [line.split("=")[1].split()[0] for line in pins.splitlines() if line.startswith("FAISS_COMMIT=")][0]
        self.touch(self.tp / "faiss-install" / f".installed-{commit}-shared")
        self.touch(self.tp / "faiss-install/lib64/libfaiss.so")
        self.touch(self.tp / "faiss-install/include/faiss/Index.h")

    def run(self, body: str, **env: str) -> subprocess.CompletedProcess:
        defaults = dict(Q1_TOOLS=str(self.tools), TP=str(self.tp), SYSTEM_BOOST_PREFIX=str(self.usr),
                        SYSTEM_LIB_DIRS=str(self.usr / "lib"), AKER_CXX=str(self.stubs / "g++"),
                        AKER_GCC=str(self.stubs / "gcc"), AKER_COMMANDS="sh", RECAL_TMP=str(self.tmp / "tmp"))
        full = {**os.environ, **defaults, **env}
        text = f"source {SCRIPT}\nroot={self.root}\n{body}"
        return subprocess.run(["bash", "-c", text], capture_output=True, text=True, env=full)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_nothing_is_missing_when_the_toolchain_is_complete(world):
    world.faiss_ready()
    out = world.run("aker_missing; echo done")
    assert out.stdout == "done\n"


def test_the_harness_cmake_is_found_without_being_on_the_path(world):
    out = world.run("find_cmake; tool_path", PATH="/usr/bin:/bin").stdout.split("\n")
    assert out[0] == f"{world.root}/.venv-build/bin/cmake" and out[1].startswith(f"{world.root}/.venv-build/bin:")


def test_a_cmake_that_is_too_old_is_not_used(world):
    script(world.root / ".venv-build/bin/cmake", 'echo "cmake version 3.10.2"')
    out = world.run("find_cmake; echo rc=$?", PATH="/usr/bin:/bin").stdout
    assert "rc=1" in out and ".venv-build" not in out.replace("rc=1", "")
    assert "cmake >= 3.23.1" in world.run("aker_missing", PATH="/usr/bin:/bin").stdout


def test_missing_tools_and_libraries_are_named_exactly(world):
    world.faiss_ready()
    (world.root / ".venv-build/bin/cmake").unlink()
    (world.usr / "lib/libboost_log_setup.so").unlink()
    out = world.run("aker_missing", PATH="/usr/bin:/bin", AKER_COMMANDS="sh no-such-tool").stdout.split("\n")[:-1]
    assert len(out) == 3
    assert out[0].startswith("cmake >= 3.23.1") and out[1] == "command no-such-tool"
    assert out[2] == f"Boost shared library libboost_log_setup.so under {world.usr}"


def test_an_old_boost_is_reported_with_its_version(world):
    world.faiss_ready()
    (world.usr / "include/boost/version.hpp").write_text('#define BOOST_LIB_VERSION "1_74"\n')
    assert f"Boost >= 1.80 (found 1.74 under {world.usr})" in world.run("aker_missing").stdout


def test_the_boost_the_harness_built_is_preferred_over_the_system_one(world):
    (world.tools / "boost").mkdir(parents=True)
    (world.tools / "boost/.installed-1.84.0").touch()
    assert world.run("boost_prefix").stdout.strip() == f"{world.tools}/boost"
    assert world.run("aker_missing").stdout.count("\n") >= 1 + len(COMPONENTS)


def test_building_faiss_needs_blas_and_openmp(world):
    (world.usr / "lib/libopenblas.so").unlink()
    (world.tmp / "libgomp.so").unlink()
    script(world.stubs / "gcc", 'echo libgomp.so')
    out = world.run("aker_missing").stdout
    assert "libopenblas.so" in out and "libgomp.so" in out


def test_require_missing_prints_the_list_and_fails(world):
    out = world.run("printf 'one\\ntwo\\n' | require_missing aker; echo rc=$?")
    assert "aker: missing:" in out.stdout and "  - one" in out.stdout and "  - two" in out.stdout and "rc=1" in out.stdout
    assert "rc=0" in world.run("printf '' | require_missing aker; echo rc=$?").stdout


def test_apt_packages_that_are_not_installed_are_named(world):
    script(world.stubs / "dpkg-query", 'case "$3" in libaio-dev) echo "install ok installed";; *) exit 1;; esac')
    out = world.run("apt_missing libaio-dev libnuma-dev", DPKG_QUERY=str(world.stubs / "dpkg-query")).stdout
    assert out == "apt package libnuma-dev\n"


def test_a_failing_requirement_check_stops_the_aker_step(world):
    (world.usr / "lib/libboost_regex.so").unlink()
    body = """
DRY=0
STEPS=(aker)
run() { echo "ran: $*"; }
as_user() { echo "ran: $*"; }
run_steps; echo "rc=$?"; print_summary
"""
    out = world.run(body, PATH="/usr/bin:/bin").stdout
    assert "aker: missing:" in out and "libboost_regex.so" in out and "rc=1" in out and "aker: FAIL" in out
    assert "build.sh" not in out


def test_the_aker_step_builds_shared_faiss_and_runs_the_build_with_the_toolchain_on_the_path():
    out = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--only", "aker"], capture_output=True, text=True).stdout
    assert "rsync" in out and "-DBUILD_SHARED_LIBS=ON" in out and "-DCMAKE_INSTALL_LIBDIR=lib64" in out
    assert "CMAKE_PREFIX_PATH=" in out and "faiss-install" in out and "env PATH=" in out
    assert "ln -s /usr" not in out.replace("ln -sfn /usr", "")


def test_the_build_script_names_a_missing_tool(tmp_path):
    result = subprocess.run(["/bin/sh", str(ROOT / "simulations/approximate_hits/aker_real/build.sh")], capture_output=True,
                            text=True, env={"PATH": str(tmp_path), "TP": str(tmp_path), "HOME": str(tmp_path)})
    assert result.returncode == 1 and "aker build: cmake not found on PATH" in result.stderr
