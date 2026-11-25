import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "scripts" / "instance_setup.sh"
UPLOAD = ROOT / "scripts" / "upload_secrets.sh"

ROOT_DISK = """\
NAME="loop0" SIZE="26214400" TYPE="loop" FSTYPE="squashfs" MOUNTPOINT="/snap/core/1" PKNAME=""
NAME="nvme0n1" SIZE="107374182400" TYPE="disk" FSTYPE="" MOUNTPOINT="" PKNAME=""
NAME="nvme0n1p1" SIZE="107262000000" TYPE="part" FSTYPE="ext4" MOUNTPOINT="/" PKNAME="nvme0n1"
NAME="nvme0n1p14" SIZE="4194304" TYPE="part" FSTYPE="" MOUNTPOINT="" PKNAME="nvme0n1"
NAME="nvme0n1p15" SIZE="111149056" TYPE="part" FSTYPE="vfat" MOUNTPOINT="/boot/efi" PKNAME="nvme0n1"
"""
DATA_DISK = 'NAME="nvme1n1" SIZE="429496729600" TYPE="disk" FSTYPE="" MOUNTPOINT="" PKNAME=""\n'
OTHER_DISK = 'NAME="nvme2n1" SIZE="214748364800" TYPE="disk" FSTYPE="" MOUNTPOINT="" PKNAME=""\n'
FORMATTED = 'NAME="nvme3n1" SIZE="214748364800" TYPE="disk" FSTYPE="ext4" MOUNTPOINT="" PKNAME=""\n'
PARTITIONED = ('NAME="nvme4n1" SIZE="214748364800" TYPE="disk" FSTYPE="" MOUNTPOINT="" PKNAME=""\n'
               'NAME="nvme4n1p1" SIZE="214748364800" TYPE="part" FSTYPE="" MOUNTPOINT="" PKNAME="nvme4n1"\n')


def sourced(script: Path, code: str, stdin: str = "", env: dict | None = None, path_prefix: Path | None = None):
    environment = {**os.environ, **(env or {})}
    if path_prefix is not None:
        environment["PATH"] = f"{path_prefix}:{environment['PATH']}"
    return subprocess.run(["bash", "-c", f'source "{script}"; {code}'], input=stdin, capture_output=True,
                          text=True, env=environment)


def pick(lsblk: str, size: str = ""):
    return sourced(SETUP, f"pick_data_volume {size}", lsblk)


def stub(directory: Path, name: str, body: str) -> None:
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def test_scripts_are_valid_bash():
    for script in (SETUP, UPLOAD):
        assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_one_unformatted_data_volume_is_picked():
    result = pick(ROOT_DISK + DATA_DISK)
    assert result.returncode == 0 and result.stdout.strip() == "/dev/nvme1n1"


def test_no_data_volume_means_none_and_the_root_disk_is_never_picked():
    result = pick(ROOT_DISK)
    assert result.returncode == 1 and result.stdout.strip() == ""
    assert pick(ROOT_DISK + FORMATTED + PARTITIONED).returncode == 1


def test_two_unformatted_volumes_are_ambiguous():
    result = pick(ROOT_DISK + DATA_DISK + OTHER_DISK)
    assert result.returncode == 2 and result.stdout.strip() == ""


def disk(name: str, gib: float) -> str:
    return f'NAME="{name}" SIZE="{int(gib * 2**30)}" TYPE="disk" FSTYPE="" MOUNTPOINT="" PKNAME=""\n'


def test_size_flag_picks_the_data_volume_among_several_unformatted_disks():
    zfs = disk("nvme2n1", 200)
    result = pick(ROOT_DISK + disk("nvme1n1", 300) + zfs, "300")
    assert result.returncode == 0 and result.stdout.strip() == "/dev/nvme1n1"
    assert pick(ROOT_DISK + zfs + disk("nvme1n1", 300), "200").stdout.strip() == "/dev/nvme2n1"
    assert pick(ROOT_DISK + disk("nvme1n1", 300) + zfs).returncode == 2


def test_size_flag_with_no_match_or_two_matches_refuses():
    assert pick(ROOT_DISK + disk("nvme1n1", 300) + disk("nvme2n1", 200), "400").returncode == 1
    both = ROOT_DISK + disk("nvme1n1", 300) + disk("nvme2n1", 303)
    result = pick(both, "300")
    assert result.returncode == 2 and result.stdout.strip() == ""


def test_size_flag_tolerance_is_two_percent():
    assert pick(ROOT_DISK + disk("nvme1n1", 305.9), "300").returncode == 0
    assert pick(ROOT_DISK + disk("nvme1n1", 306.5), "300").returncode == 1
    assert pick(ROOT_DISK + disk("nvme1n1", 294.2), "300").returncode == 0
    assert pick(ROOT_DISK + disk("nvme1n1", 293.5), "300").returncode == 1


def test_size_flag_never_selects_a_formatted_or_partitioned_disk_of_that_size():
    assert pick(ROOT_DISK + FORMATTED.replace("214748364800", str(300 * 2**30)) + PARTITIONED, "300").returncode == 1


def test_size_flag_is_validated_and_shown_in_the_dry_run(tmp_path):
    assert "of 300 GiB" in run_setup("--dry-run", "--data-size-gb", "300", "laion", tmp_home=tmp_path).stdout
    for bad in ("0", "-5", "abc", "3.5"):
        assert run_setup("--dry-run", "--data-size-gb", bad, "laion", tmp_home=tmp_path).returncode != 0


def test_usage_lists_the_size_flag(tmp_path):
    out = run_setup("--help", tmp_home=tmp_path).stdout
    assert "--data-size-gb" in out and "set -e" not in out


def run_setup(*args, env=None, tmp_home=None):
    environment = {**os.environ, **(env or {})}
    if tmp_home is not None:
        environment["HOME"] = str(tmp_home)
    return subprocess.run(["bash", str(SETUP), *args], capture_output=True, text=True, env=environment)


@pytest.mark.parametrize("dataset", ["spacev", "triviaqa", "medrag", "tripclick", "laion"])
def test_dry_run_lists_every_action_and_changes_nothing(tmp_path, dataset):
    result = run_setup("--dry-run", dataset, tmp_home=tmp_path)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for text in ("apt-get install", "awscli", "mkfs.ext4 -L recaldata", "LABEL=recaldata /data ext4 defaults,nofail",
                 "git clone https://github.com/AnuptaA/ann-exact-caching.git /data/thesis", ".git/info/exclude",
                 "export RECAL_STORE=s3://recal-eval-aa/runs", "export RECAL_QUEUE_DIR=/data/mx",
                 "ops/secrets/env.local", "ops/secrets/github_token", "ops/secrets/git_identity",
                 f"context/{dataset}.md", "claude.ai/install.sh",
                 f"sudo -E bash scripts/ec2_bootstrap.sh --role prepare --datasets {dataset}",
                 "tee /data/bootstrap.log", "new-window -t main -n agent", "tmux attach -t main",
                 "claude --remote-control", "agent_context.md"):
        assert text in out, text
    assert list(tmp_path.iterdir()) == []


def test_bad_arguments_stop_before_any_action(tmp_path):
    for args in ([], ["nosuchset"], ["--dry-run", "spacev", "laion"], ["--bogus", "laion"]):
        result = run_setup(*args, tmp_home=tmp_path)
        assert result.returncode != 0 and "+ " not in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_bucket_and_prefix_options_reach_the_settings_block(tmp_path):
    out = run_setup("--dry-run", "--bucket", "other", "--prefix", "p", "spacev", tmp_home=tmp_path).stdout
    assert "export RECAL_STORE=s3://other/p" in out and "s3://other/ops/secrets/env.local" in out


def test_refuses_to_run_as_root(tmp_path):
    stub(tmp_path / "bin", "id", 'echo 0\n')
    result = run_setup("laion", env={"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}, tmp_home=tmp_path)
    assert result.returncode != 0 and "not root" in result.stderr


def test_settings_block_is_appended_once(tmp_path):
    data = tmp_path / "data"
    code = f'DATA_DIR="{data}"; step_settings; step_settings'
    result = sourced(SETUP, code, env={"HOME": str(tmp_path)})
    assert result.returncode == 0, result.stderr
    rc = (tmp_path / ".bashrc").read_text()
    assert rc.count(">>> recal settings >>>") == 1 and "RECAL_THIRD_PARTY=" in rc
    assert (data / "mx").is_dir() and (data / "tmp").is_dir()


SECRET_BODY = '''
key="$3"
case "$key" in
  */env.local) printf 'RECAL_DRIVE_X=value1\\n' > "$4" ;;
  */github_token) printf 'ghp_secret_token\\n' > "$4" ;;
  */git_identity) printf 'Some Name\\nsome@example.com\\n' > "$4" ;;
  *) exit 1 ;;
esac
'''


def test_secrets_are_installed_with_private_modes_and_never_printed(tmp_path):
    stub(tmp_path / "bin", "aws", SECRET_BODY)
    repo = tmp_path / "repo"
    repo.mkdir()
    code = f'REPO_DIR="{repo}"; step_secrets; printf "%s" "$DONE"'
    result = sourced(SETUP, code, env={"HOME": str(tmp_path)}, path_prefix=tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    assert "ghp_secret_token" not in result.stdout + result.stderr and "value1" not in result.stdout + result.stderr
    assert stat.S_IMODE((repo / ".env.local").stat().st_mode) == 0o600
    creds = tmp_path / ".git-credentials"
    assert stat.S_IMODE(creds.stat().st_mode) == 0o600 and "ghp_secret_token" in creds.read_text()
    config = subprocess.run(["git", "config", "--global", "user.email"], capture_output=True, text=True,
                            env={**os.environ, "HOME": str(tmp_path)}).stdout.strip()
    assert config == "some@example.com"


def test_a_missing_secret_prints_where_to_upload_it_and_continues(tmp_path):
    stub(tmp_path / "bin", "aws", "exit 1\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    code = f'REPO_DIR="{repo}"; step_secrets; printf "%s" "$SKIPPED"'
    result = sourced(SETUP, code, env={"HOME": str(tmp_path)}, path_prefix=tmp_path / "bin")
    assert result.returncode == 0
    for name in ("env.local", "github_token", "git_identity"):
        assert f"s3://recal-eval-aa/ops/secrets/{name}" in result.stdout
    assert not (repo / ".env.local").exists()


# ---- upload_secrets.sh ------------------------------------------------------------------------------------------

def upload(tmp_path, args, blocked: str):
    log = tmp_path / "aws.log"
    stub(tmp_path / "bin", "aws", f'''
echo "$@" >> "{log}"
if [ "$1 $2" = "s3api get-public-access-block" ]; then echo '{blocked}'; fi
exit 0
''')
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    result = subprocess.run(["bash", str(UPLOAD), *args], capture_output=True, text=True, env=env)
    return result, log.read_text() if log.exists() else ""


ALL_ON = '{"BlockPublicAcls": true, "IgnorePublicAcls": true, "BlockPublicPolicy": true, "RestrictPublicBuckets": true}'
ONE_OFF = ALL_ON.replace('"RestrictPublicBuckets": true', '"RestrictPublicBuckets": false')


def test_upload_refuses_when_public_access_is_not_fully_blocked(tmp_path):
    secret = tmp_path / "env"
    secret.write_text("RECAL_DRIVE_X=topsecretvalue\n")
    result, log = upload(tmp_path, ["--env-file", str(secret)], ONE_OFF)
    assert result.returncode != 0 and "Block Public Access" in result.stderr
    assert " cp " not in log and "topsecretvalue" not in result.stdout + result.stderr


def test_upload_puts_each_file_under_its_key_without_echoing_values(tmp_path):
    secret, token = tmp_path / "env", tmp_path / "token"
    secret.write_text("RECAL_DRIVE_X=topsecretvalue\n")
    token.write_text("ghp_topsecret\n")
    context = tmp_path / "ctx"
    context.mkdir()
    (context / "common.md").write_text("hello")
    args = ["--env-file", str(secret), "--token-file", str(token), "--name", "Some Name", "--email", "a@b.c",
            "--context-dir", str(context)]
    result, log = upload(tmp_path, args, ALL_ON)
    assert result.returncode == 0, result.stderr
    for key in ("ops/secrets/env.local", "ops/secrets/github_token", "ops/secrets/git_identity",
                "ops/context/common.md"):
        assert f"s3://recal-eval-aa/{key}" in log
    shown = result.stdout + result.stderr + log
    assert "topsecretvalue" not in shown and "ghp_topsecret" not in shown and "a@b.c" not in shown


def test_upload_dry_run_and_argument_checks(tmp_path):
    secret = tmp_path / "env"
    secret.write_text("x\n")
    result, log = upload(tmp_path, ["--dry-run", "--env-file", str(secret)], ONE_OFF)
    assert result.returncode == 0 and "would upload" in result.stdout and log == ""
    assert upload(tmp_path, [], ALL_ON)[0].returncode != 0
    assert upload(tmp_path, ["--name", "only"], ALL_ON)[0].returncode != 0
