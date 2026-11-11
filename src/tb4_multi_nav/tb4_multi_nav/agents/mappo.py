"""MAPPO: two to four robots, one actor each, centralized critic, no dropout, no bonus."""
from tb4_multi_nav.agents.ppo import PPOConfig, PPOFamilyAgent


class MAPPOAgent(PPOFamilyAgent):
    def __init__(self, obs_dim: int, n_agents: int, cfg: PPOConfig | None = None, **kw):
        assert n_agents >= 2, "MAPPO needs at least two robots"
        super().__init__(obs_dim, n_agents, ucb=False, cfg=cfg or PPOConfig(), **kw)
