"""UCB-MAPPO: MAPPO plus MC Dropout on every actor and the UCB exploration bonus."""
from tb4_multi_nav.agents.ppo import PPOConfig, PPOFamilyAgent


class UCBMAPPOAgent(PPOFamilyAgent):
    def __init__(self, obs_dim: int, n_agents: int, cfg: PPOConfig | None = None, **kw):
        assert n_agents >= 2, "UCB-MAPPO needs at least two robots"
        super().__init__(obs_dim, n_agents, ucb=True, cfg=cfg or PPOConfig(), **kw)
