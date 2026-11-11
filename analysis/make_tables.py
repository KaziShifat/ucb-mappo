"""Build Tables 4, 5, 6 (simulation) and 8 (hardware navigation) from the logs.

    python analysis/make_tables.py [--logs logs] [--trials hardware/trials] [--out analysis/results]

Rules (as in the paper):
  * each run contributes the last 50 training episodes, averaged;
  * Tables 4/5 average those values over seeds 1..3;
  * success is per robot, collision rate is inter-robot contacts per timestep (sum / sum);
  * "change" is (UCB / baseline - 1) * 100 computed BEFORE rounding the means;
  * Table 6 is computed from the rounded means of Tables 4 and 5, one robot added at a time;
  * Table 8: trial succeeds only if every robot reaches its goal; 95 % Wilson interval.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAST = 50
ENVS = ("maze", "warehouse")
NS = (1, 2, 3, 4)
SEEDS = (1, 2, 3)


def run_metrics(csv: Path, task: str) -> dict:
    df = pd.read_csv(csv, comment="#").tail(LAST)
    if len(df) < LAST:
        raise ValueError(f"{csv}: fewer than {LAST} episodes")
    out = {"collision": df["robot_collisions"].sum() / df["steps"].sum()}
    if task == "navigation":
        out["success"] = df["success_rate"].mean()
    else:
        out["coverage"] = df["coverage"].mean()
        out["overlap"] = df["overlap"].mean()
    return out


def collect(logs: Path) -> pd.DataFrame:
    rows = []
    for task in ("navigation", "coverage"):
        for env in ENVS:
            for n in NS:
                for g in ("baseline", "ucb"):
                    for s in SEEDS:
                        f = logs / task / env / f"n{n}" / g / f"seed{s}" / "training_log.csv"
                        if not f.exists():
                            raise FileNotFoundError(f)
                        rows.append({"task": task, "env": env, "robots": n, "group": g, "seed": s,
                                     **run_metrics(f, task)})
    return pd.DataFrame(rows)


def means(runs: pd.DataFrame, task: str, metric: str) -> dict:
    """{(env, n, group): unrounded mean over seeds}"""
    d = runs[runs.task == task].groupby(["env", "robots", "group"])[metric].mean()
    return d.to_dict()


def pct_change(b: float, u: float) -> float | None:
    return None if b == 0 else (u / b - 1) * 100


def fmt_change(c: float | None) -> str:
    return "n/a" if c is None or c != c else f"{c:+.1f}"


def pair_table(runs, task, metric, nd=3):
    m = means(runs, task, metric)
    rows = []
    for env in ENVS:
        for n in NS:
            b, u = m[env, n, "baseline"], m[env, n, "ucb"]
            rows.append({"env": env, "robots": n, "baseline": round(b, nd), "ucb": round(u, nd),
                         "change_pct": pct_change(b, u)})
    return pd.DataFrame(rows)


def table6(t_cov, t_succ, t_col):
    """Change when one robot is added, from the rounded means of the other tables."""
    rows = []
    for env in ENVS:
        for g in ("baseline", "ucb"):
            def series(t):
                s = t[t.env == env].set_index("robots")[g]
                return [s[n + 1] - s[n] for n in (1, 2, 3)]
            cov = [100 * x for x in series(t_cov)]
            rows.append({"env": env, "group": g,
                         **{f"cov_pp_{a}to{a + 1}": round(v, 1) for a, v in zip((1, 2, 3), cov)},
                         **{f"succ_{a}to{a + 1}": round(v, 3) for a, v in zip((1, 2, 3), series(t_succ))},
                         **{f"coll_{a}to{a + 1}": round(v, 3) for a, v in zip((1, 2, 3), series(t_col))}})
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def table8(trials: Path) -> pd.DataFrame | None:
    base = trials / "navigation"
    if not base.exists():
        return None
    rows = []
    for nd in sorted(base.glob("n*"), key=lambda p: int(p.name[1:])):
        res = [json.loads((t / "result.json").read_text())["success"] for t in sorted(nd.glob("trial*"))]
        k, n = sum(res), len(res)
        lo, hi = wilson(k, n)
        rows.append({"robots": int(nd.name[1:]), "algorithm": "UCB-PPO" if nd.name == "n1" else "UCB-MAPPO",
                     "successes": k, "trials": n, "rate": k / n, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- writers
def _md(banner, title, header, lines):
    return "\n".join([f"<!-- {banner}{title} -->", "| " + " | ".join(header) + " |",
                      "|" + "---|" * len(header)] + lines) + "\n"


def write_pair(out: Path, name: str, title: str, tabs: dict, banner: str, nd: int = 3) -> None:
    """tabs: {'Coverage': df, 'Overlap': df} or {'Success': df, 'Collision': df}"""
    flat = []
    for label, df in tabs.items():
        d = df.copy()
        d.insert(0, "metric", label)
        flat.append(d)
    pd.concat(flat).to_csv(out / f"{name}.csv", index=False)
    header = ["Environment", "Robots"] + [f"{l} {c}" for l in tabs for c in ("base", "UCB", "chg %")]
    lines, tex = [], []
    first = next(iter(tabs.values()))
    for i in range(len(first)):
        r0 = first.iloc[i]
        cells = []
        for df in tabs.values():
            r = df.iloc[i]
            cells += [f"{r.baseline:.{nd}f}", f"{r.ucb:.{nd}f}", fmt_change(r.change_pct)]
        env = r0.env.capitalize() if r0.robots == 1 else ""
        lines.append("| " + " | ".join([env, str(r0.robots)] + cells) + " |")
        tex.append(" & ".join([env, str(r0.robots)] + cells).replace("%", r"\%") + r" \\")
    (out / f"{name}.md").write_text(_md(banner, title, header, lines))
    col = "ll" + "rrr" * len(tabs)
    (out / f"{name}.tex").write_text("\n".join(
        [f"% {banner}{title}", rf"\begin{{tabular}}{{@{{}}{col}@{{}}}}", r"\toprule",
         " & ".join(["Environment", "Robots"] + [rf"\multicolumn{{3}}{{c}}{{{l}}}" for l in tabs]) + r" \\",
         r"& & " + " & ".join(["Base & UCB & Chg.\\%"] * len(tabs)) + r" \\", r"\midrule"] + tex +
        [r"\bottomrule", r"\end{tabular}"]) + "\n")


def write_t6(out: Path, t6: pd.DataFrame, banner: str) -> None:
    t6.to_csv(out / "table6_robot_scaling.csv", index=False)
    header = ["Environment", "Algorithm"] + [f"{m} {a}->{a + 1}" for m in ("cov pp", "succ", "coll") for a in (1, 2, 3)]
    keys = [f"{p}_{a}to{a + 1}" for p in ("cov_pp", "succ", "coll") for a in (1, 2, 3)]
    lines = []
    for _, r in t6.iterrows():
        vals = [f"{r[k]:+.1f}" if k.startswith("cov") else f"{r[k]:+.3f}" for k in keys]
        lines.append("| " + " | ".join([r.env, "Baseline" if r.group == "baseline" else "UCB"] + vals) + " |")
    (out / "table6_robot_scaling.md").write_text(_md(banner, "Table 6: change when one robot is added", header, lines))


def write_t8(out: Path, t8: pd.DataFrame, banner: str) -> None:
    t8.round(3).to_csv(out / "table8_hardware_navigation.csv", index=False)
    lines = [f"| {r.robots} | {r.algorithm} | {r.successes}/{r.trials} | {100 * r.rate:.0f} | "
             f"[{100 * r.ci_low:.0f}, {100 * r.ci_high:.0f}] |" for _, r in t8.iterrows()]
    (out / "table8_hardware_navigation.md").write_text(_md(
        banner, "Table 8: navigation on the physical robots",
        ["Robots", "Algorithm", "Successful trials", "Rate (%)", "95% Wilson CI (%)"], lines))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(ROOT / "logs"))
    ap.add_argument("--trials", default=str(ROOT / "hardware" / "trials"))
    ap.add_argument("--out", default=str(ROOT / "analysis" / "results"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    banner = ""
    runs = collect(Path(a.logs))
    succ, coll = pair_table(runs, "navigation", "success"), pair_table(runs, "navigation", "collision")
    cov, ovl = pair_table(runs, "coverage", "coverage"), pair_table(runs, "coverage", "overlap")
    write_pair(out, "table4_navigation", "Table 4: navigation, last 50 training episodes, mean of 3 seeds",
               {"Success": succ, "Collision": coll}, banner)
    write_pair(out, "table5_coverage", "Table 5: coverage, last 50 training episodes, mean of 3 seeds",
               {"Coverage": cov, "Overlap": ovl}, banner)
    write_t6(out, table6(cov, succ, coll), banner)
    t8 = table8(Path(a.trials))
    if t8 is not None:
        write_t8(out, t8, banner)
    runs.to_csv(out / "per_run_metrics.csv", index=False)
    for f in ("table4_navigation.md", "table5_coverage.md", "table6_robot_scaling.md"):
        print((out / f).read_text())
    if t8 is not None:
        print((out / "table8_hardware_navigation.md").read_text())


if __name__ == "__main__":
    main()
