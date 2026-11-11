from tb4_multi_nav.agents.mappo import MAPPOAgent
from tb4_multi_nav.agents.ppo import PPOAgent, PPOConfig, PPOFamilyAgent
from tb4_multi_nav.agents.ucb_mappo import UCBMAPPOAgent
from tb4_multi_nav.agents.ucb_ppo import UCBPPOAgent

ALGORITHMS = {"ppo": PPOAgent, "ucb_ppo": UCBPPOAgent, "mappo": MAPPOAgent, "ucb_mappo": UCBMAPPOAgent}
