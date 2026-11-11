"""Figures 7, 8, 9 (and 10 when hardware trials are present) from the logs.

    python analysis/make_figures.py [--logs logs] [--trials hardware/trials] [--out analysis/results]

Bar charts of means over three seeds, black text, no orange, 2x2 panels in a single column.
Y axes start above zero on purpose (differences between baseline and UCB are small), so
mention the shortened axis in the caption.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hardware_coverage as H
import make_tables as M

ROOT = Path(__file__).resolve().parents[1]
BASE_C, UCB_C, SIM_C, HW_C = "#8da0b8", "#1f4e79", "#1f4e79", "#4f9d69"
plt.rcParams.update({"font.size": 7, "text.color": "black", "axes.labelcolor": "black",
                     "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
W = 3.4  # inches, one IEEE column


def _ylim(vals, pad=0.12):
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-3)
    return max(0.0, lo - 2.2 * span * 0.5 - pad * span), hi + pad * 2.5 * span


def _bars(ax, tab, title, ylabel, fmt, tag):
    ns = sorted(tab.robots.unique())
    x = np.arange(len(ns))
    w = 0.38
    vb, vu = tab.baseline.to_numpy(), tab.ucb.to_numpy()
    ax.bar(x - w / 2, vb, w, color=BASE_C, label="Baseline")
    ax.bar(x + w / 2, vu, w, color=UCB_C, label="UCB")
    lo, hi = _ylim(list(vb) + list(vu))
    ax.set_ylim(lo, hi)
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlabel("Robots")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({tag}) {title}", loc="left", fontsize=7)
    if fmt is not None:
        for xi, b, u in zip(x, vb, vu):
            ax.text(xi - w / 2, b, fmt(b), ha="center", va="bottom", fontsize=4.5)
            ax.text(xi + w / 2, u, fmt(u), ha="center", va="bottom", fontsize=4.5)


def grid_figure(tabs, out: Path):
    """tabs: [(left_tab, left_label), (right_tab, right_label)]"""
    fig, axs = plt.subplots(2, 2, figsize=(W, 3.5), constrained_layout=True)
    tags = iter("abcd")
    for row, env in enumerate(M.ENVS):
        for col, (tab, label) in enumerate(tabs):
            t = tab[tab.env == env]
            fmt = None if label == "Collision rate" else (lambda v: f"{v:.2f}")
            _bars(axs[row, col], t, env.capitalize(), label, fmt, next(tags))
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False, fontsize=6)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def fleet_figure(cov, succ, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.9), constrained_layout=True)
    steps = ["1→2", "2→3", "3→4"]
    x = np.arange(3)
    w = 0.38
    for k, (env, c) in enumerate((("maze", BASE_C), ("warehouse", UCB_C))):
        s = cov[cov.env == env].set_index("robots").baseline
        gain = [100 * (s[n + 1] - s[n]) for n in (1, 2, 3)]
        axs[0].bar(x + (k - 0.5) * w, gain, w, color=c, label=env.capitalize())
        for xi, g in zip(x, gain):
            axs[0].text(xi + (k - 0.5) * w, g, f"{g:.1f}", ha="center", va="bottom", fontsize=5)
    axs[0].set_xticks(x, steps)
    axs[0].set_xlabel("Robots added")
    axs[0].set_ylabel("Coverage gain (pp)")
    axs[0].set_title("(a)", loc="left", fontsize=7)
    allv = []
    for env, c in (("maze", BASE_C), ("warehouse", UCB_C)):
        s = succ[succ.env == env].set_index("robots").baseline
        allv += list(s.values)
    lo, hi = _ylim(allv)
    xs = np.arange(4)
    for k, (env, c) in enumerate((("maze", BASE_C), ("warehouse", UCB_C))):
        s = succ[succ.env == env].set_index("robots").baseline
        axs[1].bar(xs + (k - 0.5) * w, s.values, w, color=c)
    axs[1].set_ylim(lo, hi)
    axs[1].set_xticks(xs, ["1", "2", "3", "4"])
    axs[1].set_xlabel("Robots")
    axs[1].set_ylabel("Success rate")
    axs[1].set_title("(b)", loc="left", fontsize=7)
    axs[0].legend(frameon=False, fontsize=6)
    for ax in axs:
        for sp in ax.spines.values():
            sp.set_color("black")
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def hw_figure(cov, trials: Path, out: Path):
    tr = H.summarise(H.collect(trials, H.Floor()))
    sim = cov[cov.env == "warehouse"].set_index("robots").ucb * 100
    fig, ax = plt.subplots(figsize=(W, 2.0), constrained_layout=True)
    x = np.arange(4)
    w = 0.38
    ax.bar(x - w / 2, [sim[n] for n in (1, 2, 3, 4)], w, color=SIM_C, label="Simulation (warehouse)")
    ax.bar(x + w / 2, tr["mean"], w, yerr=tr["std"], capsize=2, color=HW_C, label="Hardware (5 trials)",
           error_kw={"elinewidth": 0.7, "ecolor": "black"})
    ax.set_xticks(x, ["1", "2", "3", "4"])
    ax.set_xlabel("Robots")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(20, 100)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(ROOT / "logs"))
    ap.add_argument("--trials", default=str(ROOT / "hardware" / "trials"))
    ap.add_argument("--out", default=str(ROOT / "analysis" / "results"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = M.collect(Path(a.logs))
    succ, coll = M.pair_table(runs, "navigation", "success"), M.pair_table(runs, "navigation", "collision")
    cov, ovl = M.pair_table(runs, "coverage", "coverage"), M.pair_table(runs, "coverage", "overlap")
    grid_figure([(succ, "Success rate"), (coll, "Collision rate")], out / "fig7_results_navigation")
    grid_figure([(cov, "Coverage"), (ovl, "Overlap")], out / "fig8_results_coverage")
    fleet_figure(cov, succ, out / "fig9_fleet_size")
    if (Path(a.trials) / "coverage").exists():
        hw_figure(cov, Path(a.trials), out / "fig10_hw_coverage")
    print("figures written to", out)


if __name__ == "__main__":
    main()
