import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1] / "src" / "tb4_multi_nav"


def test_worlds_have_four_robots_and_right_object_counts():
    for env, n_inner, tag in (("maze", 20, "wall_"), ("warehouse", 15, "shelf_")):
        root = ET.parse(PKG / "worlds" / f"{env}_4robots.sdf").getroot()
        names = [m.get("name") for m in root.iter("model")]
        assert {"robot1", "robot2", "robot3", "robot4"} <= set(names)
        assert sum(n.startswith(tag) for n in names) == n_inner


def test_robot_models_and_bridge():
    for i in range(1, 5):
        ET.parse(PKG / "models" / f"robot{i}_model.sdf")
    entries = yaml.safe_load((PKG / "config" / "bridge.yaml").read_text())
    topics = {e["ros_topic_name"] for e in entries}
    assert {f"/robot{i}/{t}" for i in range(1, 5) for t in ("cmd_vel", "odom", "scan")} <= topics
