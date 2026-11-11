from tb4_multi_nav.envs.coverage_env import CoverageEnv
from tb4_multi_nav.envs.navigation_env import NavigationEnv


def make_env(task: str, env: str, robots: int, **kw):
    cls = {"navigation": NavigationEnv, "coverage": CoverageEnv}[task]
    return cls(layout=env, n_agents=robots, **kw)
