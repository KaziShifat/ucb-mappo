from collections import deque

import numpy as np
import pytest

from tb4_multi_nav.envs import make_env
from tb4_multi_nav.layouts import get_layout
from tb4_multi_nav.observation import OBS_DIM, build_obs, scan_to_sectors


def test_layout_counts_match_paper():
    maze, wh = get_layout("maze"), get_layout("warehouse")
    assert maze.size == 18 and wh.size == 20
    assert sum(b.sx == 3.0 for b in maze.interior) == 10      # horizontal segments
    assert sum(b.sy == 3.0 for b in maze.interior) == 10      # vertical segments
    assert len(wh.interior) == 15 and all((b.sx, b.sy) == (0.5, 3.0) for b in wh.interior)
    assert len({b.cx for b in wh.interior}) == 3              # three columns of five


@pytest.mark.parametrize("name", ["maze", "warehouse"])
def test_start_goal_reachable(name):
    L, res, r = get_layout(name), 0.1, 0.17 + 0.05
    n = int(L.size / res)
    xs = (np.arange(n) + 0.5) * res - L.half
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    occ = np.zeros((n, n), bool)
    for b in L.all_boxes():
        x0, y0, x1, y1 = b.bounds
        occ |= np.hypot(X - np.clip(X, x0, x1), Y - np.clip(Y, y0, y1)) < r
    idx = lambda p: (int((p[0] + L.half) / res), int((p[1] + L.half) / res))  # noqa: E731
    for k in range(4):
        s, g = idx(L.starts[k]), idx(L.goals[k])
        seen, q = {s}, deque([s])
        while q:
            i, j = q.popleft()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < n and 0 <= b < n and not occ[a, b] and (a, b) not in seen:
                    seen.add((a, b))
                    q.append((a, b))
        assert g in seen


def test_observation_is_38_values_and_normalised():
    ranges = np.full(360, 20.0)
    ranges[:10] = 3.0
    obs = build_obs(ranges, np.deg2rad(np.arange(360)), (0, 0, 0), (3, 4), 25.0)
    assert obs.shape == (OBS_DIM,) == (38,)
    assert obs[0] == pytest.approx(3.0 / 12.0) and obs[1] == pytest.approx(1.0)   # 20 m -> 12 m -> 1
    assert obs[36] == pytest.approx(5.0 / 25.0)
    assert obs[37] == pytest.approx(np.arctan2(4, 3) / np.pi)


def test_real_lidar_with_800_points_gives_same_sectors():
    ang = np.linspace(-np.pi, np.pi, 800, endpoint=False)
    rng = np.where(np.abs(ang) < 0.1, 2.4, np.inf)            # invalid readings -> max range
    out = scan_to_sectors(rng, ang)
    assert out.shape == (36,) and out.min() == pytest.approx(2.4 / 12) and out.max() == 1.0


@pytest.mark.parametrize("task", ["navigation", "coverage"])
@pytest.mark.parametrize("name", ["maze", "warehouse"])
@pytest.mark.parametrize("n", [1, 4])
def test_shapes_and_finite(task, name, n):
    env = make_env(task, name, n, seed=1)
    obs, _ = env.reset()
    assert obs.shape == (n, 38)
    for _ in range(10):
        obs, rew, term, trunc, info = env.step(np.zeros((n, 2)))
    assert np.isfinite(obs).all() and rew.shape == (n,) and term.shape == (n,)
    if task == "coverage":
        assert np.allclose(obs[:, 36:], 0)                    # no goal in the coverage task


def test_step_is_half_a_second_and_episode_is_200_steps():
    env = make_env("navigation", "warehouse", 1, seed=0)
    assert env.control_dt == 0.5 and env.max_steps == 200
    env.reset()
    for _ in range(199):
        _, _, _, trunc, _ = env.step(np.zeros((1, 2)))
    assert not trunc
    _, _, _, trunc, _ = env.step(np.zeros((1, 2)))
    assert trunc


def test_wall_blocks_robot_and_penalty_is_once_per_contact():
    env = make_env("navigation", "warehouse", 1, seed=0, start_jitter=0)
    env.reset()
    env.sim.poses[0] = [0.0, -9.5 + 0.5, -np.pi / 2]
    hits = 0
    for _ in range(10):
        _, rew, _, _, info = env.step(np.array([[0.3, 0.0]]))
        hits += int(rew[0] < -40)
    assert hits == 1 and env.sim.poses[0, 1] > -10.0


def test_goal_reward_and_robot_parks():
    env = make_env("navigation", "warehouse", 1, seed=0, start_jitter=0)
    env.reset()
    env.sim.poses[0] = [7.8, 7.8, np.pi / 4]
    env.prev_dist = env._goal_dist()
    got = False
    for _ in range(10):
        _, rew, term, _, info = env.step(np.array([[0.3, 0.0]]))
        if term[0]:
            got = True
            assert rew[0] > 50 and info["success_rate"] == 1.0
            break
    assert got
    pose = env.sim.poses[0].copy()
    env.step(np.array([[0.3, 0.0]]))
    assert np.allclose(env.sim.poses[0], pose)


def test_robot_collisions_count_pairs_once():
    env = make_env("navigation", "warehouse", 2, seed=0, start_jitter=0)
    env.reset()
    env.sim.poses[0] = [-2.0, 0.0, 0.0]                      # aisle between two shelf columns
    env.sim.poses[1] = [-1.5, 0.0, np.pi]
    for _ in range(4):
        env.step(np.array([[0.3, 0.0], [0.3, 0.0]]))
    assert env.robot_collisions == 1


def test_coverage_grows_and_overlap_is_a_share():
    env = make_env("coverage", "warehouse", 2, seed=0)
    env.reset()
    c0 = env.coverage
    rng = np.random.default_rng(0)
    for _ in range(100):
        _, _, _, _, info = env.step(np.stack([[0.3, rng.uniform(-1, 1)]] * 2))
    assert info["coverage"] > c0 and 0 <= info["overlap"] <= 1
