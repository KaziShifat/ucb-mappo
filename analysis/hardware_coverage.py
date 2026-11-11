"""Trial recordings -> Table 7 (coverage on the physical robots).

    python analysis/hardware_coverage.py

Reads  hardware/trials/coverage/n<1..4>/trial<k>/poses.csv   (t, robot, x, y, yaw)
Writes analysis/results/table7_hardware_coverage.{csv,md,tex}
       analysis/results/hardware_coverage_trials.csv        (one row per trial)

Rule (paper, Section V-F): coverage = share of the free floor that came within 1 m of at
least one robot at some point of the 200 step (100 s) trial, on a grid of 0.05 m cells.
The standard deviation is the sample standard deviation (ddof = 1) over the five trials:
check this against how you computed yours.

FLOOR_RECTS below is a floor plan sized to about 85 m^2 (arena + corridor + side bay).
Replace it with the free floor of your real lab (a list of rectangles, or load a map).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, RADIUS = 0.05, 1.0
# (x0, y0, x1, y1, name) in metres, lab frame
FLOOR_RECTS = [(0.0, -3.0, 10.0, 3.0, "arena"),          # 10 x 6 m = 60 m^2
               (-10.0, -0.75, 0.0, 0.75, "corridor"),    # 10 x 1.5 m = 15 m^2
               (-6.0, 0.75, -1.0, 2.75, "side bay")]     # 5 x 2 m = 10 m^2
X_MIN, Y_MIN, X_MAX, Y_MAX = -10.5, -3.5, 10.5, 3.5


class Floor:
    def __init__(self, rects=FLOOR_RECTS, res: float = RES, radius: float = RADIUS):
        self.res, self.radius = res, radius
        self.nx, self.ny = int(round((X_MAX - X_MIN) / res)), int(round((Y_MAX - Y_MIN) / res))
        xs = X_MIN + (np.arange(self.nx) + 0.5) * res
        ys = Y_MIN + (np.arange(self.ny) + 0.5) * res
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")
        free = np.zeros((self.nx, self.ny), dtype=bool)
        for x0, y0, x1, y1, *_ in rects:
            free |= (self.X >= x0) & (self.X <= x1) & (self.Y >= y0) & (self.Y <= y1)
        self.free, self.n_free = free, int(free.sum())
        self.area = self.n_free * res * res
        self.half = int(np.ceil(radius / res)) + 1

    def is_free(self, x: float, y: float) -> bool:
        i, j = int((x - X_MIN) / self.res), int((y - Y_MIN) / self.res)
        return 0 <= i < self.nx and 0 <= j < self.ny and bool(self.free[i, j])

    def window(self, x: float, y: float):
        """Index window and boolean mask of the cells within `radius` of (x, y)."""
        ci, cj = int((x - X_MIN) / self.res), int((y - Y_MIN) / self.res)
        i0, i1 = max(ci - self.half, 0), min(ci + self.half + 1, self.nx)
        j0, j1 = max(cj - self.half, 0), min(cj + self.half + 1, self.ny)
        mask = np.hypot(self.X[i0:i1, j0:j1] - x, self.Y[i0:i1, j0:j1] - y) <= self.radius
        return i0, i1, j0, j1, mask


class CoverageCounter:
    """Counts, per cell, how many recorded poses cover it. Add and remove poses cheaply."""

    def __init__(self, floor: Floor):
        self.f = floor
        self.count = np.zeros((floor.nx, floor.ny), dtype=np.int32)

    def add(self, x: float, y: float, sign: int = 1) -> None:
        i0, i1, j0, j1, m = self.f.window(x, y)
        self.count[i0:i1, j0:j1] += sign * m

    def coverage(self) -> float:
        return float(((self.count > 0) & self.f.free).sum() / self.f.n_free)


def coverage_from_poses(floor: Floor, xy: np.ndarray) -> float:
    c = CoverageCounter(floor)
    for x, y in xy:
        c.add(x, y)
    return c.coverage()


def read_poses(path: Path) -> np.ndarray:
    df = pd.read_csv(path, comment="#")
    return df[["x", "y"]].to_numpy()


def collect(trials_root: Path, floor: Floor) -> pd.DataFrame:
    rows = []
    for ndir in sorted((trials_root / "coverage").glob("n*"), key=lambda p: int(p.name[1:])):
        for tdir in sorted(ndir.glob("trial*"), key=lambda p: int(p.name[5:])):
            cov = coverage_from_poses(floor, read_poses(tdir / "poses.csv"))
            rows.append({"robots": int(ndir.name[1:]), "trial": int(tdir.name[5:]), "coverage_pct": 100 * cov})
    return pd.DataFrame(rows)


def summarise(trials: pd.DataFrame) -> pd.DataFrame:
    out = []
    for n, g in trials.groupby("robots"):
        v = g["coverage_pct"].to_numpy()
        out.append({"robots": n, "algorithm": "UCB-PPO" if n == 1 else "UCB-MAPPO", "trials": len(v),
                    "mean": v.mean(), "std": v.std(ddof=1), "min": v.min(), "max": v.max()})
    return pd.DataFrame(out)


def write_table(tab: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = tab.copy()
    for c in ("mean", "std", "min", "max"):
        r[c] = r[c].round(1)
    banner = ""
    r.to_csv(out / "table7_hardware_coverage.csv", index=False)
    lines = [f"<!-- {banner}Table 7: coverage on the physical robots, five trials -->",
             "| Robots | Algorithm | Mean (%) | Std. | Min. | Max. |", "|---|---|---|---|---|---|"]
    lines += [f"| {x.robots} | {x.algorithm} | {x['mean']:.1f} | {x['std']:.1f} | {x['min']:.1f} | {x['max']:.1f} |"
              for _, x in r.iterrows()]
    (out / "table7_hardware_coverage.md").write_text("\n".join(lines) + "\n")
    tex = [f"% {banner}Table 7", r"\begin{tabular}{@{}cccccc@{}}", r"\toprule",
           r" & & \multicolumn{4}{c}{Coverage (\%)} \\", r"\cmidrule(l){3-6}",
           r"Robots & Algorithm & Mean & Std. & Min. & Max. \\", r"\midrule"]
    tex += [f"{x.robots} & {x.algorithm} & {x['mean']:.1f} & {x['std']:.1f} & {x['min']:.1f} & {x['max']:.1f} \\\\"
            for _, x in r.iterrows()]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "table7_hardware_coverage.tex").write_text("\n".join(tex) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default=str(ROOT / "hardware" / "trials"))
    ap.add_argument("--out", default=str(ROOT / "analysis" / "results"))
    a = ap.parse_args(argv)
    floor = Floor()
    trials = collect(Path(a.trials), floor)
    if trials.empty:
        raise SystemExit("no trials found under " + a.trials)
    tab = summarise(trials)
    write_table(tab, Path(a.out))
    trials.round(3).to_csv(Path(a.out) / "hardware_coverage_trials.csv", index=False)
    print(f"free floor {floor.area:.1f} m^2 on a {RES} m grid, radius {RADIUS} m")
    print(tab.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
