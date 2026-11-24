import json

from recal.system_benefit import best_split
from simulations.approximate_hits import throughput_replay as tr


def rows():
    out = []
    for arm, qps in (("recal_n80", [10, 12]), ("recal_n40", [20, 22]), ("recal_n0", [20, 22]),
                     ("aker_n80", [5, 5]), ("aker_n60", [4, 4])):
        out += [{"arm": arm, "clients": 64, "qps": v} for v in qps]
        out.append({"arm": arm, "clients": 8, "qps": 100})
    return out


def test_best_split_is_the_highest_saturated_throughput_with_ties_to_more_nodes():
    specs = {r["arm"]: f"{r['arm']}=x.json,nodes=1" for r in rows()}
    best = best_split.best_splits(rows(), 64, specs)
    assert best["recal"]["node_share"] == 40 and best["recal"]["qps"] == 21
    assert best["aker"]["arm"] == "aker_n80"
    assert best["recal"]["spec"] == "recal_n40=x.json,nodes=1"


def test_cli_run_and_get(tmp_path, capsys):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps(rows()))
    out = tmp_path / "best.json"
    arms = []
    for name in sorted({r["arm"] for r in rows()}):
        arms += ["--arm", f"{name}=x.json,nodes=3"]
    best_split.main(["run", "--sweep", str(sweep), "--clients", "64", *arms, "--out", str(out)])
    best_split.main(["get", str(out), "recal.spec"])
    assert capsys.readouterr().out.strip() == "recal_n40=x.json,nodes=3"


def test_arm_spec_has_an_optional_node_cache_and_none_means_no_result_cache():
    assert tr.parse_arm("none") == ("none", None, None)
    assert tr.parse_arm("nodeonly=none,nodes=500") == ("nodeonly", None, 500)
    assert tr.parse_arm("aker_n40=results/a.json,nodes=12") == ("aker_n40", "results/a.json", 12)
    assert tr.parse_arm("ours=results/a.json") == ("ours", "results/a.json", None)
