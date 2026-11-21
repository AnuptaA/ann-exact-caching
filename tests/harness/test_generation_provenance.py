"""Result headers say where the embedding and ground-truth stages ran."""

import json

from harness.runner import replay


def test_generation_stages_lists_the_device_of_each_compute_stage(tmp_path):
    done = tmp_path / ".done"
    done.mkdir()
    (done / "groundtruth.json").write_text(json.dumps({
        "seconds": 3.0, "device": "cpu", "requested_device": "auto", "threads": 8,
        "device_notice": "No GPU found: running ground truth on CPU (8 threads)", "torch": None}))
    (done / "embed.json").write_text(json.dumps({"seconds": 1.0}))
    (done / "to_f32.json").write_text(json.dumps({"n_base": 5}))
    stages = replay.generation_stages(tmp_path)
    assert stages["groundtruth"]["device"] == "cpu" and stages["groundtruth"]["threads"] == 8
    assert stages["groundtruth"]["device_notice"].startswith("No GPU found")
    assert stages["embed"] == {} and "to_f32" not in stages and "seconds" not in stages["groundtruth"]


def test_a_dataset_without_markers_has_no_generation_stages(tmp_path):
    assert replay.generation_stages(tmp_path) == {}
