import numpy as np
import pytest
import torch

from tb4_multi_nav.agents import ALGORITHMS, PPOConfig
from tb4_multi_nav.agents.ppo import Actor
from tb4_multi_nav.envs import make_env


def _episode(env, agent, T=40):
    obs, _ = env.reset()
    buf = {k: [] for k in ("obs", "next_obs", "act", "logp", "unc", "rew", "term", "active")}
    active = np.ones(env.n, dtype=np.float32)
    for _ in range(T):
        raw, lp, u = agent.act(obs)
        nobs, rew, term, _, _ = env.step(agent.to_command(raw))
        for k, v in zip(buf, (obs, nobs, raw, lp, u, rew, term.astype(np.float32), active.copy())):
            buf[k].append(v)
        active = active * (1 - term.astype(np.float32))
        obs = nobs
    return {k: np.asarray(v, dtype=np.float32) for k, v in buf.items()}


@pytest.mark.parametrize("algo,n", [("ppo", 1), ("ucb_ppo", 1), ("mappo", 3), ("ucb_mappo", 4)])
def test_update_runs_and_changes_actor(algo, n):
    torch.manual_seed(0)
    env = make_env("navigation", "warehouse", n, seed=0)
    agent = ALGORITHMS[algo](env.obs_dim, n, cfg=PPOConfig(epochs=2, n_mc=8))
    before = [p.clone() for p in agent.actors[0].parameters()]
    stats = agent.update(_episode(env, agent))
    assert np.isfinite([stats["policy_loss"], stats["value_loss"]]).all()
    assert any(not torch.equal(b, a) for b, a in zip(before, agent.actors[0].parameters()))


def test_critic_input_is_152_for_four_robots_and_has_no_dropout():
    a = ALGORITHMS["ucb_mappo"](38, 4)
    assert a.critic.body[0].in_features == 152
    assert not any(isinstance(m, torch.nn.Dropout) for m in a.critic.modules())
    assert ALGORITHMS["ppo"](38, 1).critic.body[0].in_features == 38


def test_one_actor_per_robot_and_dropout_only_in_ucb():
    ucb, base = ALGORITHMS["ucb_mappo"](38, 3), ALGORITHMS["mappo"](38, 3)
    assert len(ucb.actors) == 3 and len({id(a) for a in ucb.actors}) == 3
    has_do = lambda ag: any(isinstance(m, torch.nn.Dropout) for m in ag.actors[0].modules())  # noqa: E731
    assert has_do(ucb) and not has_do(base)
    assert [m.p for m in ucb.actors[0].modules() if isinstance(m, torch.nn.Dropout)] == [0.1, 0.1]


def test_mc_dropout_uncertainty_is_variance_of_passes():
    torch.manual_seed(0)
    actor = Actor(38, 64, p_drop=0.2, forward_bias=0.0)
    for p in actor.body[-1].parameters():
        torch.nn.init.normal_(p, std=0.5)                    # make the output layer non-trivial
    x = torch.rand(5, 38)
    mean, u = actor(x, n_mc=50)
    assert mean.shape == (5, 2) and (u > 0).all()
    plain = Actor(38, 64, p_drop=0.0)
    assert (plain(x, 50)[1] == 0).all()


def test_beta_schedule_reaches_five_percent_at_last_episode():
    a = ALGORITHMS["ucb_mappo"](38, 2, cfg=PPOConfig(episodes=400))
    assert a.beta == pytest.approx(1.0)
    a.episode = 400
    assert a.beta == pytest.approx(0.05, rel=1e-6)
    assert ALGORITHMS["mappo"](38, 2).beta == 0.0


def test_wrong_pairing_is_rejected():
    with pytest.raises(AssertionError):
        ALGORITHMS["ppo"](38, 2)
    with pytest.raises(AssertionError):
        ALGORITHMS["ucb_mappo"](38, 1)


def test_save_load_roundtrip(tmp_path):
    a = ALGORITHMS["ucb_mappo"](38, 2)
    a.save(str(tmp_path / "x.pt"))
    b = ALGORITHMS["ucb_mappo"](38, 2)
    b.load(str(tmp_path / "x.pt"))
    obs = np.random.default_rng(0).uniform(0, 1, (2, 38)).astype(np.float32)
    assert np.allclose(a.act(obs, True)[0], b.act(obs, True)[0])
