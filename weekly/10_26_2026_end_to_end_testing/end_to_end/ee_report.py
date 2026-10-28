#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Assembles one report PDF (letter landscape slides, vector pages) from the saved stage D, E and F results: title, contents,
design, glossary, headline tables, then the figure slides of ee_plots.py by section, and the lambda = 1.0 sanity figures
last. Nothing is recomputed. An existing PDF is never overwritten unless --overwrite is given.

Run from the thesis root:  python3 weekly/10_26_2026_end_to_end_testing/end_to_end/ee_report.py"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ee_config as cfg  # noqa: E402
import ee_lsh_bench as lb  # noqa: E402
import ee_plots as ep  # noqa: E402
from plot_style import MUTED_TEXT, PRIMARY_TEXT  # noqa: E402
from simulations.pdf.report import PdfReportBuilder  # noqa: E402

OUT_PATH = cfg.PDF_DIR / "end_to_end_report.pdf"
DPI = 120
MONO = "DejaVu Sans Mono"
PAGE_H = ep.PAGE_H

#-------------------------------------------------------------------------------
# Text slides

def text_slide(title: str, sub_lines, lines: list, mono: bool = False, size: float = ep.MIN_FONT, step: float = 0.235):
    fig = plt.figure(figsize=ep.PAGE_SIZE)
    y = ep.draw_header(fig, title, ep.wrap_lines(sub_lines)) + 0.45 / PAGE_H
    for line in lines:
        fig.text(0.04, y, line, fontsize=size, color=PRIMARY_TEXT, va="top", ha="left", family=MONO if mono else None)
        y -= step / PAGE_H
    assert y > 0.02, f"text does not fit on the slide: {title}"
    return fig

def title_slide():
    fig = plt.figure(figsize=ep.PAGE_SIZE)
    fig.text(0.5, 0.60, "End-to-end test of cache population and hit policies", fontsize=28, fontweight="bold", ha="center",
             va="center", color=PRIMARY_TEXT)
    fig.text(0.5, 0.49, "Oracle radius arm (stage D) and LSH radius arm (stages E and F)", fontsize=17, ha="center", va="center",
             color="#333333")
    fig.text(0.5, 0.40, datetime.now().strftime("%B %d, %Y"), fontsize=14, ha="center", va="center", color=MUTED_TEXT)
    return fig

def section_slide(title: str, text: str):
    fig = plt.figure(figsize=ep.PAGE_SIZE)
    fig.text(0.5, 0.55, title, fontsize=26, fontweight="bold", ha="center", va="center", color=PRIMARY_TEXT)
    for i, line in enumerate(ep.wrap_lines(text)):
        fig.text(0.5, 0.45 - 0.045 * i, line, fontsize=14, ha="center", va="center", color="#333333")
    return fig

DESIGN = [
    "Datasets: SIFT (d = 128) and ESCI (d = 384), database of 50,000 vectors, angular distance, K = 10.",
    "Queries: 2,500 train queries (build the pool and learn the HP3 cap) and 2,500 test queries (the stream), seeds 0-4.",
    "Budgets: 1% (500), 5% (2,500) and 10% (5,000) of the database. Targets T = 0.8, 0.9, 0.95.",
    "Regimes: lambda = 0, 0.5, 1.0 (share of local queries, 50 train and 50 test copies per anchor) and diffuse (500 anchors x 5).",
    "Lambda = 1.0 saturates (5% and 10% give the same pool), so its figures are collected at the end as sanity figures.",
    "Pools: proposals Own, Group Union, Ball with scores SEL1-SEL5 (SEL4 with m = 0.5, 1.0, 1.5), and baselines Popularity,",
    "Random Own, threshold_baseline and no cache. Pools are built with exact deficits for both arms.",
    "Hit policies: HP1 (estimate >= T), HP2 (running slack), HP3 (HP2 plus a learned deficit cap), HP3f (HP3 plus estimate >= 0.5).",
    "Oracle arm: the radius is the true K-th neighbor distance. LSH arm: Hamming-LSH estimate, the configuration chosen on",
    "seeds 100-102 (SIFT L = 16, m = 128, nu = 2, S = 10,000; ESCI L = 8, m = 64, nu = 4, S = 10,000).",
    "Triplets: T1 HP1 Ball SEL1, T2 HP3 Group Union SEL2, T3 HP3 Own SEL5, T4 HP3 Ball SEL3, T5 HP3 Group Union SEL4 (m = 1.0).",
    "Plots: oracle solid, LSH dashed, one color per budget, markers at T = 0.8 (circle), 0.9 (square), 0.95 (triangle),",
    "bands are +/- 1 sd over the five seeds.",
]

GLOSSARY = [
    "hit rate: share of test queries answered from the cache. A miss is answered exactly and has recall 1.",
    "quality hit rate: share of test queries answered from the cache with true recall >= 0.5.",
    "mean true recall of hits: average true recall over the hit queries.",
    "achieved recall Y: average delivered recall over the stream (true recall for hits, 1 for misses). Y_t is the running value.",
    "target recall T: the average recall the policy must deliver. Floor: hit rate 1 - T is reachable with zero-recall hits.",
    "hindsight: best hit rate any policy could reach on the same stream with the same estimates, keeping Y >= T.",
    "HP3 cap: largest deficit the policy accepts, learned from the train queries. HP3f also requires estimate >= 0.5.",
    "regime lambda: share of local queries. Diffuse: 500 anchors with 5 train and 5 test copies each.",
    "pool: the set of database vectors in the cache. Own: top-K sets of single queries. Group Union: unions of nearby sets.",
    "Ball: all vectors within a ball around a query. SEL1-SEL5: the gain per cost score used by the greedy selection.",
    "oracle arm: radius = true K-th neighbor distance. LSH arm: radius estimated from LSH tables over labeled points.",
    "signed error: estimated minus true recall (negative means underestimated). Over rate: share overestimated by more than 0.05.",
    "gap to the oracle: LSH arm metric minus the oracle arm metric on the same pool, policy and seed.",
    "structure bytes: LSH codes, labels and hyperplane bits. Cache bytes: pool vectors as float32.",
]

#-------------------------------------------------------------------------------
# Headline tables

HEAD_POOLS = (("group_union+SEL2", "GU SEL2"), ("own+SEL5", "Own SEL5"), ("group_union+SEL4(m=1.0)", "GU SEL4"),
              ("ball+SEL3", "Ball SEL3"), ("popularity", "Popular."), ("random_own", "RandOwn"), ("threshold_baseline", "Thresh."))

def head_table(data: ep.Data, B: int, metric: str) -> list:
    lines = [f"{'dataset regime':<18}" + "".join(f"{label:>11}" for _, label in HEAD_POOLS)]
    for ds in cfg.DATASETS:
        for regime in ("lam0", "lam0.5", "diffuse", "lam1"):
            cells = []
            for pool, _ in HEAD_POOLS:
                o = data.value("oracle", ds, regime, pool, "HP3", B, 0.9, metric)
                lsh = data.value("lsh", ds, regime, pool, "HP3", B, 0.9, metric)
                cells.append(f"{o:.2f}/{lsh:.2f}".rjust(11))
            tag = f"{ep.DATASET_LABEL[ds]} {regime}" + (" (sanity)" if regime == "lam1" else "")
            lines.append(f"{tag:<18}" + "".join(cells))
    return lines

def headline_slide(data: ep.Data, B: int):
    lines = ["Hit rate, oracle / LSH:"] + head_table(data, B, "hit_rate") + ["", "Quality hit rate (true recall >= 0.5), oracle / LSH:"] + \
        head_table(data, B, "hit_rate_recall_ge_half")
    return text_slide(f"Headline: HP3 hit rates at T = 0.9, budget {ep.BUDGET_LABELS[B]}",
                      "Mean over seeds 0-4. Pools as columns (GU = Group Union). Lambda = 1.0 rows are the saturated sanity regime.",
                      lines, mono=True, size=10.5, step=0.21)

def safety_slide(data: ep.Data):
    f = data.f.reset_index()
    f = f[(f.policy == "HP3") & (f.pool != "no_cache")].copy()
    f["viol"] = f.y_min_minus_T_worst < -1e-12
    lines = ["HP3, all pools except no cache, all budgets and T:",
             f"{'dataset regime':<18}{'cells':>6}{'y_min<T':>9}{'frac<T':>8}{'worst':>8}{'hit gap':>9}{'qual gap':>10}{'signed':>8}{'over':>7}"]
    for ds in cfg.DATASETS:
        for regime in ("lam0", "lam0.5", "diffuse", "lam1"):
            g = f[(f.dataset == ds) & (f.regime == regime)]
            tag = f"{ep.DATASET_LABEL[ds]} {regime}" + (" (sanity)" if regime == "lam1" else "")
            lines.append(f"{tag:<18}{len(g):>6}{g.viol.mean():>9.2f}{g.frac_below_T.mean():>8.3f}{g.y_min_minus_T_worst.min():>8.2f}"
                         f"{g.gap_hit_rate.mean():>9.3f}{g.gap_hit_rate_recall_ge_half.mean():>10.3f}"
                         f"{g.est_signed_error.mean():>8.3f}{g.over_rate.mean():>7.3f}")
    a = data.f.reset_index()
    a = a[a.pool != "no_cache"].copy()
    a["viol"] = a.y_min_minus_T_worst < -1e-12
    lines += ["", "By policy, all pools except no cache, all regimes, budgets and T:",
              f"{'policy':<18}{'cells':>6}{'y_min<T':>9}{'frac<T':>8}{'hit gap':>9}{'qual gap':>10}"]
    for p in ep.POLICY_COLORS:
        g = a[a.policy == p]
        lines.append(f"{p:<18}{len(g):>6}{g.viol.mean():>9.2f}{g.frac_below_T.mean():>8.3f}{g.gap_hit_rate.mean():>9.3f}"
                     f"{g.gap_hit_rate_recall_ge_half.mean():>10.3f}")
    return text_slide("Headline: LSH arm safety and gap to the oracle",
                      "y_min<T: share of cells whose worst seed dips below T at some point. frac<T: mean share of steps with Y_t < T. "
                      "worst: lowest y_min - T over seeds. Gaps are LSH minus oracle.", lines, mono=True, size=10.5, step=0.21)

def stage_e_slide():
    chosen = json.loads((lb.OUT_DIR / "chosen_config.json").read_text())["config"]
    rows = pd.concat([pd.read_csv(p, comment="#", low_memory=False) for p in sorted(glob.glob(str(lb.UNIT_DIR / "*.csv.gz")))])
    dims = {"sift": 128, "esci": 384}
    lines = [f"{'dataset':<8}{'pool':<18}{'row':<24}{'MAE':>8}{'signed':>9}{'over':>8}{'bytes':>12}"]
    for ds in cfg.DATASETS:
        c = chosen[ds]
        for pool in lb.POOL_NAMES:
            r = rows[(rows.dataset == ds) & (rows.pool == pool)]
            g = r.groupby(["L", "m", "nu", "S"]).agg(mae=("mae", "mean"), over=("over_rate", "mean"), sgn=("mean_signed_error", "mean"),
                                                      by=("bytes", "first")).reset_index()
            pick = g[(g.L == c["L"]) & (g.m == c["m"]) & (g.nu == c["nu"]) & (g.S == c["S"])].iloc[0]
            same = g[(g.L == c["L"]) & (g.m == c["m"]) & (g.nu == c["nu"]) & (g.S == "full")].iloc[0]
            best = g[g.S == "full"].sort_values("mae").iloc[0]
            for label, x in (("chosen", pick), ("S = full, same config", same), (f"S = full, best ({best.L},{best.m},{best.nu})", best)):
                lines.append(f"{ep.DATASET_LABEL[ds]:<8}{pool:<18}{label:<24}{x.mae:>8.4f}{x.sgn:>9.4f}{x.over:>8.4f}{int(x.by):>12,}")
    lines += ["", "Chosen configurations:"]
    for ds in cfg.DATASETS:
        c = chosen[ds]
        lines.append(f"{ep.DATASET_LABEL[ds]}: L = {c['L']}, m = {c['m']}, nu = {c['nu']}, S = {c['S']}, "
                     f"structure {lb.config_bytes(c['L'], c['m'], dims[ds], cfg.N_QUERIES + int(c['S'])):,} bytes, same for both reference pools")
    return text_slide("Stage E: chosen LSH configuration against the S = full reference",
                      "Mean over regimes lambda = 0, 0.5, 1.0 and seeds 100-102, reference pools at the 5% budget (T = 0.9). "
                      "Signed: estimated minus true recall. Over: share overestimated by more than 0.05.", lines, mono=True,
                      size=10, step=0.2)

#-------------------------------------------------------------------------------
# Page plan

SECTION_TEXT = {
    "A": "Hit rate, achieved recall Y and quality hit rate for the five triplets, one set per dataset and regime.",
    "B": "Hit rate against budget for every pool at T = 0.9 with HP3, oracle and LSH side by side, with hindsight dotted.",
    "C": "HP1, HP2, HP3 and HP3f on the Group Union SEL2 pool at the 5% budget.",
    "D": "Heatmaps of the LSH minus oracle gap for HP3 hit rate and quality hit rate.",
    "E": "Final Y, share of steps below T, with and without the 100 query burn-in, and the minimum running Y_t.",
    "F": "Running Y_t of the oracle and LSH arms under HP3, seed 0, for two cells.",
    "G": "Recall estimation error of the chosen LSH configuration, and structure bytes against cache bytes.",
}
SANITY_TEXT = "Lambda = 1.0 saturates: the 5% and 10% budgets give the same pool. The same figures as above for that regime."

def plan(data: ep.Data) -> list:
    """Ordered items: ("page", title, builder) for one page, ("group", title, (maker, sanity), count) for the figure slides
    of one section. Counts come from a first pass that builds and discards every figure."""
    items = []
    for sanity in (False, True):
        if sanity:
            items.append(("page", "Sanity: lambda = 1.0", lambda: section_slide("Sanity: lambda = 1.0", SANITY_TEXT)))
        for key, name, maker in ep.SECTIONS:
            slides = list(maker(data, sanity))
            for s in slides:
                plt.close(s.fig)
            if not slides:
                continue
            title = f"{key}. {name}" + (" (lambda = 1.0)" if sanity else "")
            if not sanity:
                items.append(("page", title, lambda key=key, name=name: section_slide(f"{key}. {name}", SECTION_TEXT[key])))
            items.append(("group", title, (maker, sanity), len(slides)))
    return items

def contents_lines(n_front: int, items: list) -> list:
    lines = [f"{1:>4}  Title, contents, design, glossary", f"{5:>4}  Headline tables (budgets, LSH safety, stage E)"]
    page = n_front + 1
    for item in items:
        if item[0] == "page":
            lines.append(f"{page:>4}  {item[1]}")
            page += 1
        else:
            if item[1].endswith("(lambda = 1.0)"):
                lines.append(f"{page:>4}    {item[1]}")
            page += item[3]
    return lines + [f"{page - 1:>4}  last page"]

def front_pages(data: ep.Data, contents_text: list) -> list:
    return [("Title", title_slide),
            ("Contents", lambda: text_slide("Contents", "Page numbers refer to this PDF.", contents_text, mono=True)),
            ("Design", lambda: text_slide("Design", [], DESIGN)),
            ("Glossary", lambda: text_slide("Glossary", [], GLOSSARY)),
            ("Headline 1% budget", lambda: headline_slide(data, 500)),
            ("Headline 5% budget", lambda: headline_slide(data, 2500)),
            ("Headline 10% budget", lambda: headline_slide(data, 5000)),
            ("LSH safety headline", lambda: safety_slide(data)),
            ("Stage E headline", stage_e_slide)]

def render(data: ep.Data, front: list, items: list, out_path) -> list:
    manifest = []

    def add(report, fig, title):
        report._pdf.savefig(fig, dpi=DPI)
        plt.close(fig)
        manifest.append((len(manifest) + 1, title))
        if len(manifest) % 20 == 0:
            print(f"{len(manifest)} pages", file=sys.stderr, flush=True)

    with PdfReportBuilder(out_path) as report:
        for title, build in front:
            add(report, build(), title)
        for item in items:
            if item[0] == "page":
                add(report, item[2](), item[1])
            else:
                maker, sanity = item[2]
                for slide in maker(data, sanity):
                    add(report, slide.fig, slide.title)
    return manifest

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        raise FileExistsError(f"{out} exists; use --overwrite")
    data = ep.Data()
    items = plan(data)
    n_front = len(front_pages(data, []))
    front = front_pages(data, contents_lines(n_front, items))
    manifest = render(data, front, items, out)
    expected = n_front + sum(1 if i[0] == "page" else i[3] for i in items)
    assert len(manifest) == expected, (len(manifest), expected)
    print(f"Saved: {out} ({len(manifest)} pages)")
    pd.DataFrame(manifest, columns=["page", "title"]).to_csv(cfg.RESULTS_DIR / "report_manifest.csv", index=False)
    return 0

if __name__ == "__main__":
    sys.exit(main())
