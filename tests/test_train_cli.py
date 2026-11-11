import pandas as pd
import pytest

from tb4_multi_nav import train as T


def test_run_dir_layout():
    assert str(T.run_dir("logs", "navigation", "maze", 4, "ucb_mappo", 2)) == \
        "logs/navigation/maze/n4/ucb/seed2"
    assert str(T.run_dir("logs", "coverage", "warehouse", 1, "ppo", 1)) == \
        "logs/coverage/warehouse/n1/baseline/seed1"


def test_pairing_check():
    with pytest.raises(SystemExit):
        T.check_pair("ppo", 3)
    with pytest.raises(SystemExit):
        T.check_pair("mappo", 1)
    T.check_pair("ucb_ppo", 1)


def test_short_run_writes_log_and_checkpoint(tmp_path):
    cfg = T.load_config("configs/ucb_mappo.yaml")
    out = T.train("navigation", "warehouse", 2, "ucb_mappo", 1, cfg, str(tmp_path), episodes=3, quiet=True)
    df = pd.read_csv(out / "training_log.csv")
    assert list(df.columns) == T.FIELDS and len(df) == 3
    assert (out / "actors.pt").exists() and (out / "config.yaml").exists()
    assert df["beta"].iloc[0] == pytest.approx(1.0) and df["beta"].iloc[-1] < 1.0
