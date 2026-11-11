"""Train one run.

    python -m tb4_multi_nav.train --task navigation --env maze --robots 4 \
        --algo ucb_mappo --seed 1

Algorithm and fleet size must match: ppo / ucb_ppo need --robots 1,
mappo / ucb_mappo need --robots 2 to 4.

Writes   logs/<task>/<env>/n<robots>/<baseline|ucb>/seed<seed>/training_log.csv
         logs/<task>/<env>/n<robots>/<baseline|ucb>/seed<seed>/actors.pt
         logs/<task>/<env>/n<robots>/<baseline|ucb>/seed<seed>/config.yaml
Hyperparameters come from configs/<algo>.yaml (paper Table 2). One row per
training episode; metrics are later read from the last 50 episodes of each run.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from tb4_multi_nav.agents import ALGORITHMS, PPOConfig
from tb4_multi_nav.envs import make_env

FIELDS = ["episode", "steps", "mean_return", "success_rate", "collision_rate", "coverage",
          "overlap", "robot_collisions", "beta", "uncertainty"]
GROUP = {"ppo": "baseline", "mappo": "baseline", "ucb_ppo": "ucb", "ucb_mappo": "ucb"}


def run_dir(root: str, task: str, env: str, robots: int, algo: str, seed: int) -> Path:
    return Path(root) / task / env / f"n{robots}" / GROUP[algo] / f"seed{seed}"


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("env", {})
    cfg.setdefault("ppo", {})
    return cfg


def check_pair(algo: str, robots: int) -> None:
    single = algo in ("ppo", "ucb_ppo")
    if single and robots != 1:
        raise SystemExit(f"{algo} is the one-robot algorithm; use mappo or ucb_mappo for {robots} robots")
    if not single and robots < 2:
        raise SystemExit(f"{algo} needs 2 to 4 robots; use ppo or ucb_ppo for one robot")


def train(task: str, env_name: str, robots: int, algo: str, seed: int, cfg: dict,
          log_root: str = "logs", device: str = "cpu", episodes: int | None = None,
          quiet: bool = False) -> Path:
    check_pair(algo, robots)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)                    # tiny networks: one thread is faster
    ppo_cfg = PPOConfig(**cfg["ppo"])
    if episodes:
        ppo_cfg.episodes = episodes
    env_kw = {k: v for k, v in cfg["env"].items() if k != "reward"}
    reward = (cfg["env"].get("reward") or {}).get(task, {})
    env = make_env(task, env_name, robots, seed=seed, reward=reward, **env_kw)
    agent = ALGORITHMS[algo](env.obs_dim, robots, cfg=ppo_cfg, device=device)

    out = run_dir(log_root, task, env_name, robots, algo, seed)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({"task": task, "env": env_name, "robots": robots, "algo": algo,
                        "seed": seed, **cfg}, f, sort_keys=False)
    fh = open(out / "training_log.csv", "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    wr.writeheader()

    t0 = time.time()
    for ep in range(ppo_cfg.episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        buf = {k: [] for k in ("obs", "next_obs", "act", "logp", "unc", "rew", "term", "active")}
        active = np.ones(robots, dtype=np.float32)
        ep_ret = np.zeros(robots)
        while True:
            raw, logp, unc = agent.act(obs)
            nobs, rew, term, trunc, info = env.step(agent.to_command(raw))
            for k, v in zip(buf, (obs, nobs, raw, logp, unc, rew, term.astype(np.float32), active.copy())):
                buf[k].append(v)
            ep_ret += rew * active
            active = active * (1.0 - term.astype(np.float32))
            obs = nobs
            if trunc or term.all():
                break
        stats = agent.update({k: np.asarray(v, dtype=np.float32) for k, v in buf.items()})
        steps = info["t"]
        row = {"episode": ep + 1, "steps": steps, "mean_return": round(float(ep_ret.mean()), 3),
               "success_rate": info.get("success_rate", ""), "coverage": info.get("coverage", ""),
               "overlap": info.get("overlap", ""), "robot_collisions": info["robot_collisions"],
               "collision_rate": info["robot_collisions"] / steps, "beta": round(stats["beta"], 5),
               "uncertainty": f"{stats['uncertainty']:.3e}"}
        wr.writerow(row)
        fh.flush()
        if not quiet and (ep + 1) % 20 == 0:
            m = row["success_rate"] if task == "navigation" else row["coverage"]
            print(f"[{task}/{env_name}/n{robots}/{algo}/seed{seed}] episode {ep + 1}/{ppo_cfg.episodes} "
                  f"{'success' if task == 'navigation' else 'coverage'} {float(m):.2f} "
                  f"return {row['mean_return']:.0f} beta {row['beta']:.3f} {time.time() - t0:.0f}s", flush=True)
    fh.close()
    agent.save(str(out / "actors.pt"), {"task": task, "env": env_name})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--task", required=True, choices=["navigation", "coverage"])
    ap.add_argument("--env", required=True, choices=["maze", "warehouse"])
    ap.add_argument("--robots", required=True, type=int, choices=[1, 2, 3, 4])
    ap.add_argument("--algo", required=True, choices=list(ALGORITHMS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--config", default=None, help="default: configs/<algo>.yaml")
    ap.add_argument("--log-root", default="logs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--episodes", type=int, default=None, help="override the 400 of the paper (smoke tests)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config or f"configs/{a.algo}.yaml")
    out = train(a.task, a.env, a.robots, a.algo, a.seed, cfg, a.log_root, a.device, a.episodes)
    print("done:", out)


if __name__ == "__main__":
    main()
