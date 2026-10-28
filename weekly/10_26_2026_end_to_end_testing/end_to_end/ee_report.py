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
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import ee_config as cfg  # noqa: E402
import ee_lsh_bench as lb  # noqa: E402
import ee_plots as ep  # noqa: E402
from plot_style import AXIS_COLOR, MUTED_TEXT, PRIMARY_TEXT  # noqa: E402
from simulations.pdf.report import PdfReportBuilder  # noqa: E402

OUT_PATH = cfg.PDF_DIR / "end_to_end_report.pdf"
DPI = 120
PAGE_W, PAGE_H = ep.PAGE_W, ep.PAGE_H
LEFT = 0.4
FONT = ep.MIN_FONT
LAM1 = r"$\lambda$ = 1.0"
POOL_NAMES = {"popularity": "Popularity", "group_union_sel2": "Group Union SEL2"}

#-------------------------------------------------------------------------------
# Drawing helpers (positions in inches from the top left corner)

def put(fig, x: float, y: float, text: str, ha: str = "left", **kw):
    return fig.text(x / PAGE_W, 1 - y / PAGE_H, text, ha=ha, va="top", fontsize=kw.pop("fontsize", FONT),
                    color=kw.pop("color", PRIMARY_TEXT), **kw)

def header_bottom(fig, title: str, sub_lines) -> float:
    """Draws the header and returns the top of the content area in inches from the top."""
    top = ep.draw_header(fig, title, ep.wrap_lines(sub_lines))
    return (1 - top) * PAGE_H - 0.4

def title_slide():
    fig = plt.figure(figsize=ep.PAGE_SIZE)
    for i, line in enumerate(("End-to-end test of cache population", "and hit policies")):
        fig.text(0.5, 0.64 - 0.095 * i, line, fontsize=30, fontweight="bold", ha="center", va="center", color=PRIMARY_TEXT)
    fig.text(0.5, 0.43, "Oracle radius arm (stage D) and LSH radius arm (stages E and F)", fontsize=17, ha="center", va="center",
             color="#333333")
    fig.text(0.5, 0.34, datetime.now().strftime("%B %d, %Y"), fontsize=14, ha="center", va="center", color=MUTED_TEXT)
    return fig

def section_slide(title: str, text: str):
    fig = plt.figure(figsize=ep.PAGE_SIZE)
    fig.text(0.5, 0.58, title, fontsize=26, fontweight="bold", ha="center", va="center", color=PRIMARY_TEXT)
    for i, line in enumerate(textwrap.wrap(text, width=80, break_on_hyphens=False)):
        fig.text(0.5, 0.46 - 0.05 * i, line, fontsize=14, ha="center", va="center", color="#333333")
    return fig

#-------------------------------------------------------------------------------
# Term and definition slides (bold term, wrapped text)

def term_slides(title: str, sub_lines, items: list, label_w: float, wrap: int, step: float = 0.25, gap: float = 0.15) -> list:
    """Pages of (bold label, wrapped text) rows with generous spacing; a new page starts when the next row does not fit."""
    pages, current, y_used = [], [], 0.0
    limit = PAGE_H - 1.2 - 0.45
    for label, text in items:
        lines = textwrap.wrap(text, width=wrap, break_on_hyphens=False)
        h = len(lines) * step + gap
        if y_used + h > limit and current:
            pages.append(current)
            current, y_used = [], 0.0
        current.append((label, lines))
        y_used += h
    pages.append(current)
    figs = []
    for k, page in enumerate(pages):
        fig = plt.figure(figsize=ep.PAGE_SIZE)
        suffix = f" ({k + 1} of {len(pages)})" if len(pages) > 1 else ""
        y = header_bottom(fig, title + suffix, sub_lines) + 0.1
        for label, lines in page:
            put(fig, LEFT, y, label, fontweight="bold")
            for i, line in enumerate(lines):
                put(fig, LEFT + label_w, y + i * step, line)
            y += len(lines) * step + gap
        figs.append(fig)
    return figs

DESIGN = [
    ("Datasets", "SIFT ($d$ = 128) and ESCI ($d$ = 384), database of 50,000 vectors, angular distance, $K$ = 10."),
    ("Queries", "2,500 train queries (they build the pool and learn the HP3 cap) and 2,500 test queries (the stream), seeds 0 to 4."),
    ("Budgets", "1% (500), 5% (2,500) and 10% (5,000) of the database."),
    ("Targets", "$T$ = 0.8, 0.9 and 0.95."),
    ("Regimes", r"$\lambda$ = 0, 0.5 and 1.0 (share of local queries, 50 train and 50 test copies per anchor) and diffuse "
                r"(500 anchors $\times$ 5 copies). $\lambda$ = 1.0 saturates (the 5% and 10% budgets give the same pool), so "
                "its figures come last as sanity figures."),
    ("Pools", "Proposals Own, Group Union and Ball with scores SEL1 to SEL5 (SEL4 with $m$ = 0.5, 1.0, 1.5), and the baselines "
              "Popularity, Random Own, threshold_baseline and no cache. Pools are built with exact deficits for both arms."),
    ("Hit policies", "HP1 (estimate $\\geq T$), HP2 (running slack), HP3 (HP2 plus a learned deficit cap) and HP3f (HP3 plus "
                     "estimate $\\geq$ 0.5)."),
    ("Oracle arm", "The radius is the true $K$-th neighbor distance."),
    ("LSH arm", r"Hamming-LSH radius estimate. Configuration chosen on seeds 100 to 102: SIFT $L$ = 16, $m$ = 128, $\nu$ = 2, "
                r"$S$ = 10,000; ESCI $L$ = 8, $m$ = 64, $\nu$ = 4, $S$ = 10,000."),
    ("Triplets", "T1 HP1 Ball SEL1, T2 HP3 Group Union SEL2, T3 HP3 Own SEL5, T4 HP3 Ball SEL3, T5 HP3 Group Union SEL4 "
                 "($m$ = 1.0)."),
    ("Plots", "Oracle solid, LSH dashed, one color per budget, markers at $T$ = 0.8 (circle), 0.9 (square) and 0.95 (triangle), "
              r"bands are $\pm$ 1 sd over the five seeds."),
]

GLOSSARY = [
    ("hit rate", "Share of test queries answered from the cache. A miss is answered exactly and has recall 1."),
    ("quality hit rate", "Share of test queries answered from the cache with true recall $\\geq$ 0.5."),
    ("mean true recall of hits", "Average true recall over the hit queries."),
    ("achieved recall $Y$", "Average delivered recall over the stream (true recall for a hit, 1 for a miss)."),
    ("running $Y_t$", "The achieved recall after the first $t$ queries of the stream."),
    ("target recall $T$", "The average recall the policy must deliver."),
    ("floor", "A hit rate of $1-T$ is reachable with zero-recall hits alone."),
    ("hindsight", "The best hit rate any policy could reach on the same stream with the same estimates while keeping $Y\\geq T$."),
    ("HP3 cap", "The largest deficit HP3 accepts, learned from the train queries. HP3f also requires an estimate of at least 0.5."),
    ("regime $\\lambda$", "Share of local queries. Diffuse: 500 anchors with 5 train and 5 test copies each."),
    ("pool", "The set of database vectors in the cache. Own: top-$K$ sets of single queries. Group Union: unions of nearby "
             "sets. Ball: all vectors within a ball around a query."),
    ("SEL1 to SEL5", "The gain-per-cost scores used by the greedy pool selection."),
    ("oracle arm", "Radius equals the true $K$-th neighbor distance."),
    ("LSH arm", "Radius estimated from LSH tables over labeled points ($L$ tables, $m$ bits, $\\nu$ neighbors per table, $S$ sampled "
                "database points)."),
    ("signed error", "Estimated minus true recall. Negative means underestimated."),
    ("over rate", "Share of test queries whose recall is overestimated by more than 0.05."),
    ("gap to the oracle", "LSH arm metric minus oracle arm metric on the same pool, policy and seed."),
    ("structure bytes", "LSH codes, labels and hyperplane bits. Cache bytes: pool vectors stored as float32."),
]

#-------------------------------------------------------------------------------
# Table slides (cells placed by column, so mathtext can be used anywhere)

def table_slide(title: str, sub_lines, blocks: list):
    """blocks: dicts with optional caption, header (cells may hold newlines), rows and cols [(width in, 'l' or 'r')]."""
    fig = plt.figure(figsize=ep.PAGE_SIZE)
    y = header_bottom(fig, title, sub_lines)
    for block in blocks:
        xs, x = [], LEFT
        for width, align in block["cols"]:
            xs.append((x, x + width, align))
            x += width
        if block.get("caption"):
            put(fig, LEFT, y, block["caption"], fontweight="bold")
            y += 0.32
        n_lines = max(c.count("\n") + 1 for c in block["header"])
        for cell, (x0, x1, align) in zip(block["header"], xs):
            put(fig, x0 if align == "l" else x1, y, cell, ha="left" if align == "l" else "right", fontweight="bold")
        y += 0.24 * n_lines + 0.08
        fig.add_artist(Line2D([LEFT / PAGE_W, x / PAGE_W], [1 - y / PAGE_H] * 2, color=AXIS_COLOR, linewidth=0.8))
        y += 0.08
        for row in block["rows"]:
            for cell, (x0, x1, align) in zip(row, xs):
                put(fig, x0 if align == "l" else x1, y, str(cell), ha="left" if align == "l" else "right")
            y += 0.265
        y += 0.3
    assert y < PAGE_H - 0.1, f"table does not fit: {title}"
    return fig

HEAD_POOLS = (("group_union+SEL2", "GU SEL2"), ("own+SEL5", "Own SEL5"), ("group_union+SEL4(m=1.0)", "GU SEL4"),
              ("ball+SEL3", "Ball SEL3"), ("popularity", "Popularity"), ("random_own", "Random\nOwn"),
              ("threshold_baseline", "threshold\nbaseline"))
REGIME_ROWS = ("lam0", "lam0.5", "diffuse", "lam1")

def row_label(ds: str, regime: str) -> str:
    return f"{ep.DATASET_LABEL[ds]}, {ep.REGIME_SHORT[regime]}" + (" (sanity)" if regime == "lam1" else "")

def head_block(data: ep.Data, B: int, metric: str, caption: str) -> dict:
    rows = []
    for ds in cfg.DATASETS:
        for regime in REGIME_ROWS:
            cells = [row_label(ds, regime)]
            for pool, _ in HEAD_POOLS:
                o = data.value("oracle", ds, regime, pool, "HP3", B, 0.9, metric)
                lsh = data.value("lsh", ds, regime, pool, "HP3", B, 0.9, metric)
                cells.append(f"{o:.2f} / {lsh:.2f}")
            rows.append(cells)
    return dict(caption=caption, header=["dataset, regime"] + [label for _, label in HEAD_POOLS], rows=rows,
                cols=[(2.45, "l")] + [(1.05, "r")] * len(HEAD_POOLS))

def headline_slide(data: ep.Data, B: int):
    blocks = [head_block(data, B, "hit_rate", "Hit rate, oracle / LSH"),
              head_block(data, B, "hit_rate_recall_ge_half", r"Quality hit rate (true recall $\geq$ 0.5), oracle / LSH")]
    return table_slide(f"Headline: HP3 hit rates at $T$ = 0.9, budget {ep.BUDGET_LABELS[B]}",
                       [r"Mean over seeds 0-4. GU = Group Union. The $\lambda$ = 1.0 rows are the saturated sanity regime."], blocks)

def safety_slide(data: ep.Data):
    f = data.f.reset_index()
    f = f[f.pool != "no_cache"].copy()
    f["viol"] = f.y_min_minus_T_worst < -1e-12
    h = f[f.policy == "HP3"]
    rows = []
    for ds in cfg.DATASETS:
        for regime in REGIME_ROWS:
            g = h[(h.dataset == ds) & (h.regime == regime)]
            rows.append([row_label(ds, regime), len(g), f"{g.viol.mean():.2f}", f"{g.frac_below_T.mean():.3f}",
                         f"{g.y_min_minus_T_worst.min():.2f}", f"{g.gap_hit_rate.mean():+.3f}",
                         f"{g.gap_hit_rate_recall_ge_half.mean():+.3f}", f"{g.est_signed_error.mean():+.3f}",
                         f"{g.over_rate.mean():.3f}"])
    head = ["dataset, regime", "cells", "cells with\n$Y_t<T$", "steps with\n$Y_t<T$", "worst\n$\\min\\,Y_t-T$",
            "hit gap", "quality\nhit gap", "signed\nerror", "over\nrate"]
    cols = [(2.2, "l"), (0.6, "r")] + [(1.1, "r")] * 7
    prow = []
    for p in ep.POLICY_COLORS:
        g = f[f.policy == p]
        prow.append([p, len(g), f"{g.viol.mean():.2f}", f"{g.frac_below_T.mean():.3f}", f"{g.gap_hit_rate.mean():+.3f}",
                     f"{g.gap_hit_rate_recall_ge_half.mean():+.3f}"])
    blocks = [dict(caption="HP3, all pools except no cache, all budgets and $T$", header=head, rows=rows, cols=cols),
              dict(caption="By policy, all pools except no cache, all regimes, budgets and $T$",
                   header=["policy", "cells", "cells with\n$Y_t<T$", "steps with\n$Y_t<T$", "hit gap",
                           "quality\nhit gap"], rows=prow, cols=[(2.2, "l"), (0.6, "r")] + [(1.1, "r")] * 4)]
    return table_slide("Headline: LSH arm safety and gap to the oracle",
                       ["A cell is a (pool, budget, $T$, regime). Cells with $Y_t<T$: share of cells where the worst seed dips below $T$ "
                        "at some point. Steps with $Y_t<T$: mean share of steps. Worst: lowest $Y_t-T$ over seeds. Gaps are "
                        "LSH minus oracle."], blocks)

def stage_e_slide():
    chosen = json.loads((lb.OUT_DIR / "chosen_config.json").read_text())["config"]
    rows = pd.concat([pd.read_csv(p, comment="#", low_memory=False) for p in sorted(glob.glob(str(lb.UNIT_DIR / "*.csv.gz")))])
    dims = {"sift": 128, "esci": 384}
    body = []
    for ds in cfg.DATASETS:
        c = chosen[ds]
        for pool in lb.POOL_NAMES:
            r = rows[(rows.dataset == ds) & (rows.pool == pool)]
            g = r.groupby(["L", "m", "nu", "S"]).agg(mae=("mae", "mean"), over=("over_rate", "mean"), sgn=("mean_signed_error", "mean"),
                                                      by=("bytes", "first")).reset_index()
            pick = g[(g.L == c["L"]) & (g.m == c["m"]) & (g.nu == c["nu"]) & (g.S == c["S"])].iloc[0]
            same = g[(g.L == c["L"]) & (g.m == c["m"]) & (g.nu == c["nu"]) & (g.S == "full")].iloc[0]
            best = g[g.S == "full"].sort_values("mae").iloc[0]
            for label, x in (("chosen", pick), ("full $S$, same config", same),
                             (f"full $S$, best: {best.L}/{best.m}/{best.nu}", best)):
                body.append([ep.DATASET_LABEL[ds], POOL_NAMES[pool], label, f"{x.mae:.4f}", f"{x.sgn:+.4f}", f"{x.over:.4f}", f"{int(x.by):,}"])
    cfg_rows = [[ep.DATASET_LABEL[ds], chosen[ds]["L"], chosen[ds]["m"], chosen[ds]["nu"], f"{int(chosen[ds]['S']):,}",
                 f"{lb.config_bytes(chosen[ds]['L'], chosen[ds]['m'], dims[ds], cfg.N_QUERIES + int(chosen[ds]['S'])):,}"]
                for ds in cfg.DATASETS]
    blocks = [dict(header=["dataset", "reference pool", "row", "MAE", "signed", "over", "bytes"], rows=body,
                   cols=[(0.8, "l"), (1.8, "l"), (3.0, "l"), (0.9, "r"), (0.9, "r"), (0.9, "r"), (1.3, "r")]),
              dict(caption="Chosen configurations (the same for both reference pools)",
                   header=["dataset", "$L$", "$m$", r"$\nu$", "$S$", "structure bytes"], rows=cfg_rows,
                   cols=[(0.8, "l"), (0.8, "r"), (0.8, "r"), (0.8, "r"), (1.2, "r"), (1.8, "r")])]
    return table_slide("Stage E: chosen LSH configuration against the full-$S$ reference",
                       [r"Mean over regimes $\lambda$ = 0, 0.5, 1.0 and seeds 100-102, reference pools at the 5% budget ($T$ = 0.9). "
                        r"Signed: estimated minus true recall. Over: share overestimated by more than 0.05. MAE: mean absolute "
                        "error."], blocks)

#-------------------------------------------------------------------------------
# Page plan

SECTION_TEXT = {
    "A": "Hit rate, achieved recall $Y$ and quality hit rate for the five triplets, one set per dataset and regime.",
    "B": "Hit rate against budget for every pool at $T$ = 0.9 with HP3, oracle and LSH side by side, with hindsight dotted.",
    "C": "HP1, HP2, HP3 and HP3f on the Group Union SEL2 pool at the 5% budget.",
    "D": "Heatmaps of the LSH minus oracle gap for HP3 hit rate and quality hit rate.",
    "E": r"Final $Y$, share of steps below $T$, with and without the 100 query burn-in, and the minimum running $Y_t$.",
    "F": r"Running $Y_t$ of the oracle and LSH arms under HP3, seed 0, for two cells.",
    "G": "Recall estimation error of the chosen LSH configuration, and structure bytes against cache bytes.",
}
SANITY_TEXT = r"$\lambda$ = 1.0 saturates: the 5% and 10% budgets give the same pool. The same figures as above for that regime."
SANITY_TAG = r" ($\lambda$ = 1.0)"

def plan(data: ep.Data) -> list:
    """Ordered items: ("page", title, builder) for one page, ("group", title, (maker, sanity), count) for the figure slides
    of one section. Counts come from a first pass that builds and discards every figure."""
    items = []
    for sanity in (False, True):
        if sanity:
            items.append(("page", "Sanity: " + LAM1, lambda: section_slide("Sanity: " + LAM1, SANITY_TEXT)))
        for key, name, maker in ep.SECTIONS:
            slides = list(maker(data, sanity))
            for s in slides:
                plt.close(s.fig)
            if not slides:
                continue
            title = f"{key}. {name}" + (SANITY_TAG if sanity else "")
            if not sanity:
                items.append(("page", title, lambda key=key, name=name: section_slide(f"{key}. {name}", SECTION_TEXT[key])))
            items.append(("group", title, (maker, sanity), len(slides)))
    return items

def front_pages(data: ep.Data, contents_rows: list) -> list:
    pages = [("Title", lambda: [title_slide()]),
             ("Contents", lambda: [contents_slide(contents_rows)]),
             ("Design", lambda: term_slides("Design", [], DESIGN, label_w=1.6, wrap=96)),
             ("Glossary", lambda: term_slides("Glossary", [], GLOSSARY, label_w=2.7, wrap=82)),
             ("Headline 1% budget", lambda: [headline_slide(data, 500)]),
             ("Headline 5% budget", lambda: [headline_slide(data, 2500)]),
             ("Headline 10% budget", lambda: [headline_slide(data, 5000)]),
             ("LSH safety headline", lambda: [safety_slide(data)]),
             ("Stage E headline", lambda: [stage_e_slide()])]
    return pages

def contents_slide(rows: list):
    fig = plt.figure(figsize=ep.PAGE_SIZE)
    y = header_bottom(fig, "Contents", "Page numbers refer to this PDF.")
    for page, title, indent in rows:
        put(fig, LEFT + 0.6, y, str(page), ha="right")
        put(fig, LEFT + 0.85 + (0.3 if indent else 0), y, title)
        y += 0.3
    assert y < PAGE_H - 0.1
    return fig

def n_front_pages(data: ep.Data) -> int:
    figs = [f for _, build in front_pages(data, []) for f in build()]
    for f in figs:
        plt.close(f)
    return len(figs)

def contents_rows(n_front: int, items: list) -> list:
    rows = [(1, "Title, contents, design, glossary", False), (n_front - 4, "Headline tables, LSH safety, stage E", False)]
    page = n_front + 1
    for item in items:
        if item[0] == "page":
            rows.append((page, item[1], False))
            page += 1
        else:
            if item[1].endswith(SANITY_TAG):
                rows.append((page, item[1], True))
            page += item[3]
    return rows

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
            for k, fig in enumerate(build()):
                add(report, fig, title if k == 0 else f"{title} (continued)")
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
    n_front = n_front_pages(data)
    front = front_pages(data, contents_rows(n_front, items))
    manifest = render(data, front, items, out)
    expected = n_front + sum(1 if i[0] == "page" else i[3] for i in items)
    assert len(manifest) == expected, (len(manifest), expected)
    print(f"Saved: {out} ({len(manifest)} pages)")
    pd.DataFrame(manifest, columns=["page", "title"]).to_csv(cfg.RESULTS_DIR / "report_manifest.csv", index=False)
    return 0

if __name__ == "__main__":
    sys.exit(main())
