import os
from glob import glob

from setuptools import find_packages, setup

pkg = "tb4_multi_nav"
setup(
    name=pkg,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + pkg]),
        ("share/" + pkg, ["package.xml"]),
        (os.path.join("share", pkg, "launch"), glob("launch/*.py")),
        (os.path.join("share", pkg, "config"), glob("config/*.yaml")),
        (os.path.join("share", pkg, "worlds"), glob("worlds/*.sdf")),
        (os.path.join("share", pkg, "models"), glob("models/*.sdf")),
    ],
    install_requires=["setuptools", "numpy", "torch", "pyyaml"],
    zip_safe=True,
    maintainer="TODO maintainer",
    maintainer_email="you@example.edu",
    description="Uncertainty-driven exploration for multi-robot reinforcement learning.",
    license="MIT",
    entry_points={"console_scripts": ["ucb_train = tb4_multi_nav.train:main"]},
)
