"""The manual Linux build workflow: the only Linux verification available off the instance."""

from tests.harness.test_deploy_scripts import ROOT

WORKFLOW = (ROOT / ".github" / "workflows" / "q1-linux-build.yml").read_text()


def test_workflow_is_manual_only_and_covers_both_ubuntu_releases():
    assert "workflow_dispatch:" in WORKFLOW
    for trigger in ("push:", "pull_request:", "schedule:"):
        assert trigger not in WORKFLOW
    assert "ubuntu-22.04" in WORKFLOW and "ubuntu-24.04" in WORKFLOW


def test_workflow_runs_setup_the_link_check_the_smoke_test_and_the_clean_tree_check():
    for needle in ("setup_instance.sh --ci", 'ldd "build/replay/$exe"', "not found", "TOOLCHAIN.txt",
                   "make q1-smoke", "git status --short", "Q1_SETUP_DATASETS: none"):
        assert needle in WORKFLOW, needle
    assert WORKFLOW.index("setup_instance.sh") < WORKFLOW.index("ldd") < WORKFLOW.index("q1-smoke")
