"""UCB-PPO: one robot, actor with MC Dropout and the exploration bonus."""
from tb4_multi_nav.agents.ppo import PPOConfig, PPOFamilyAgent


class UCBPPOAgent(PPOFamilyAgent):
    def __init__(self, obs_dim: int, n_agents: int = 1, cfg: PPOConfig | None = None, **kw):
        assert n_agents == 1, "UCB-PPO is the one-robot version"
        super().__init__(obs_dim, 1, ucb=True, cfg=cfg or PPOConfig(), **kw)
