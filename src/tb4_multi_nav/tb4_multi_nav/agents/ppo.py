"""PPO core used by all four algorithms (paper Sections III-B to III-G).

                 1 robot     2 to 4 robots
    baseline     PPO         MAPPO
    UCB          UCB-PPO     UCB-MAPPO

What is shared by all four
  * One actor per robot (no parameter sharing), 38 -> 256 -> 256 -> 2, ReLU, tanh output.
  * One critic that sees the observations of all robots side by side
    (38 x N inputs, 152 for four robots) and returns ONE value for the fleet.
    With one robot it is simply the robot's own 38 inputs. The critic has no dropout.
  * Gaussian policy with a learned standard deviation. GAE (gamma 0.99, lambda 0.95),
    clip 0.2, 10 epochs over mini-batches of 64, one update at the end of every episode.

What only the UCB versions have
  * Dropout p = 0.1 after each hidden layer of the ACTOR.
  * The action mean is the mean of N_mc = 50 dropout passes, m_bar(o).
  * Uncertainty u(o) = variance of the 50 passes, averaged over the two outputs.
  * Reward r~ = r + beta_e * u(o), beta_e = beta_0 * exp(-e / tau), tau = E / ln(1/0.05).

Choices this code made that the paper does not state (compare with your code):
  * u(o) is measured on the tanh outputs in [-1, 1], before scaling to velocity units.
  * The bonus is added to the raw reward, then everything is multiplied by reward_scale.
  * The critic is regressed on the mean over active robots of each robot's return.
  * reward_scale = 0.05 and forward_bias = 0.5 (initial forward speed) are NOT in
    the paper's Table 2. Set reward_scale: 1.0 and forward_bias: 0.0 to remove them.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

V_MAX, W_MAX = 0.3, 1.0


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    batch_size: int = 64
    hidden: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    episodes: int = 400
    # UCB only
    dropout_p: float = 0.1
    n_mc: int = 50
    beta0: float = 1.0
    beta_final_ratio: float = 0.05
    # not in the paper, see module docstring
    reward_scale: float = 0.05
    forward_bias: float = 0.5


def _mlp(inp: int, hidden: int, out: int, p_drop: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(inp, hidden), nn.ReLU()]
    if p_drop > 0:
        layers.append(nn.Dropout(p_drop))
    layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    if p_drop > 0:
        layers.append(nn.Dropout(p_drop))
    layers.append(nn.Linear(hidden, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 38, hidden: int = 256, p_drop: float = 0.0,
                 forward_bias: float = 0.0, log_std_init: float = -0.5):
        super().__init__()
        self.p_drop = p_drop
        self.body = _mlp(obs_dim, hidden, 2, p_drop)
        self.log_std = nn.Parameter(torch.full((2,), log_std_init))
        nn.init.orthogonal_(self.body[-1].weight, gain=0.01)
        nn.init.zeros_(self.body[-1].bias)
        with torch.no_grad():
            self.body[-1].bias[0] = forward_bias

    def forward(self, obs: torch.Tensor, n_mc: int = 1):
        """obs [B, 38] -> (mean [B, 2] in [-1, 1], uncertainty [B])."""
        if self.p_drop > 0 and n_mc > 1:
            self.train()                                    # dropout ON for the passes
            out = torch.tanh(self.body(obs.unsqueeze(0).expand(n_mc, *obs.shape)))
            return out.mean(0), out.var(0, unbiased=False).mean(-1)
        out = torch.tanh(self.body(obs))
        return out, torch.zeros(obs.shape[0], device=obs.device)

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.clamp(-3.0, 0.5).exp()


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.body = _mlp(in_dim, hidden, 1)                  # no dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).squeeze(-1)


class PPOFamilyAgent:
    def __init__(self, obs_dim: int, n_agents: int, ucb: bool, cfg: PPOConfig,
                 device: str = "cpu"):
        self.obs_dim, self.n, self.ucb, self.cfg = obs_dim, n_agents, ucb, cfg
        self.device = torch.device(device)
        p = cfg.dropout_p if ucb else 0.0
        self.actors = nn.ModuleList(
            [Actor(obs_dim, cfg.hidden, p, cfg.forward_bias) for _ in range(n_agents)]).to(self.device)
        self.critic = Critic(n_agents * obs_dim, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(list(self.actors.parameters()) + list(self.critic.parameters()),
                                    lr=cfg.lr, eps=1e-5)
        self.episode = 0
        self.tau = cfg.episodes / math.log(1.0 / cfg.beta_final_ratio)

    @property
    def beta(self) -> float:
        return self.cfg.beta0 * math.exp(-self.episode / self.tau) if self.ucb else 0.0

    # ----------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """obs [n, 38] -> raw action [n, 2] in policy units (not clipped), logp [n], u [n]."""
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        raw, logp, unc = [], [], []
        for k, actor in enumerate(self.actors):
            if deterministic:
                actor.eval()
                mean, u = actor(o[k:k + 1], 1)
            else:
                mean, u = actor(o[k:k + 1], self.cfg.n_mc if self.ucb else 1)
            if deterministic:
                a = mean
            else:
                dist = torch.distributions.Normal(mean, actor.std)
                a = dist.sample()
                logp.append(dist.log_prob(a).sum(-1))
            raw.append(a)
            unc.append(u)
        raw = torch.cat(raw).cpu().numpy()
        logp = torch.cat(logp).cpu().numpy() if logp else np.zeros(self.n, dtype=np.float32)
        return raw, logp, torch.cat(unc).cpu().numpy()

    @staticmethod
    def to_command(raw: np.ndarray) -> np.ndarray:
        """Policy units [-1, 1] -> (v, w) in m/s and rad/s."""
        return np.clip(raw, -1.0, 1.0) * np.array([V_MAX, W_MAX], dtype=np.float32)

    # ----------------------------------------------------------------- update
    def update(self, buf: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update at the end of one episode. Arrays are [T, n, ...] except `end` [T].

        keys: obs, next_obs, act (raw), logp, unc, rew (raw), term, active
        """
        c, dev, n = self.cfg, self.device, self.n
        T = buf["rew"].shape[0]
        t = lambda k: torch.as_tensor(buf[k], dtype=torch.float32, device=dev)  # noqa: E731
        obs, nobs, act, logp_old = t("obs"), t("next_obs"), t("act"), t("logp")
        unc, term, active = t("unc"), t("term"), t("active")
        beta = self.beta
        rew = (t("rew") + beta * unc) * c.reward_scale        # r~ = r + beta u, then scaled

        with torch.no_grad():
            v = self.critic(obs.reshape(T, -1))                # [T], one value for the fleet
            v_next = self.critic(nobs.reshape(T, -1))
            adv = torch.zeros(T, n, device=dev)
            last = torch.zeros(n, device=dev)
            for s in reversed(range(T)):
                delta = rew[s] + c.gamma * (1 - term[s]) * v_next[s] - v[s]
                last = delta + c.gamma * c.gae_lambda * (1 - term[s]) * last
                adv[s] = last
            ret_k = adv + v[:, None]
            w = active / active.sum(1, keepdim=True).clamp(min=1)
            target = (ret_k * w).sum(1)                         # mean return of active robots
            step_ok = (active.sum(1) > 0)
            a_flat = adv[active > 0.5]
            adv = (adv - a_flat.mean()) / (a_flat.std() + 1e-8) if a_flat.numel() > 1 else adv

        pl_s = vl_s = ent_s = 0.0
        n_mb = 0
        for _ in range(c.epochs):
            perm = torch.randperm(T, device=dev)
            for s0 in range(0, T, c.batch_size):
                idx = perm[s0:s0 + c.batch_size]
                p_loss, ent = 0.0, 0.0
                for k, actor in enumerate(self.actors):
                    mean, _ = actor(obs[idx, k], c.n_mc if self.ucb else 1)
                    dist = torch.distributions.Normal(mean, actor.std)
                    logp = dist.log_prob(act[idx, k]).sum(-1)
                    ratio = (logp - logp_old[idx, k]).exp()
                    a_ = adv[idx, k]
                    m = active[idx, k]
                    l = -torch.min(ratio * a_, ratio.clamp(1 - c.clip_eps, 1 + c.clip_eps) * a_)
                    p_loss = p_loss + (l * m).sum() / m.sum().clamp(min=1)
                    ent = ent + (dist.entropy().sum(-1) * m).sum() / m.sum().clamp(min=1)
                vm = step_ok[idx].float()
                v_loss = 0.5 * (((self.critic(obs[idx].reshape(len(idx), -1)) - target[idx]) ** 2) * vm).sum() \
                    / vm.sum().clamp(min=1)
                loss = p_loss / n + c.value_coef * v_loss - c.entropy_coef * ent / n
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actors.parameters()) + list(self.critic.parameters()),
                                         c.max_grad_norm)
                self.opt.step()
                pl_s += p_loss.item() / n
                vl_s += v_loss.item()
                ent_s += ent.item() / n
                n_mb += 1
        self.episode += 1
        return {"policy_loss": pl_s / n_mb, "value_loss": vl_s / n_mb, "entropy": ent_s / n_mb,
                "beta": beta, "uncertainty": float((unc * active).sum() / active.sum().clamp(min=1))}

    # ---------------------------------------------------------------- persist
    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({"actors": [a.state_dict() for a in self.actors], "critic": self.critic.state_dict(),
                    "obs_dim": self.obs_dim, "n_agents": self.n, "ucb": self.ucb,
                    "episode": self.episode, "cfg": asdict(self.cfg), "extra": extra or {}}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        for a, sd in zip(self.actors, ck["actors"]):
            a.load_state_dict(sd)
        self.critic.load_state_dict(ck["critic"])
        self.episode = ck.get("episode", 0)


class PPOAgent(PPOFamilyAgent):
    """Single robot baseline."""
    def __init__(self, obs_dim: int, n_agents: int = 1, cfg: PPOConfig | None = None, **kw):
        assert n_agents == 1, "PPO is the one-robot baseline"
        super().__init__(obs_dim, 1, ucb=False, cfg=cfg or PPOConfig(), **kw)
