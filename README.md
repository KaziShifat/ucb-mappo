# UCB-MAPPO

Code, logs and analysis scripts for the UCB-MAPPO paper: MAPPO with Monte Carlo dropout uncertainty
as an exploration bonus, for multi-robot navigation and area coverage with one to four TurtleBot4 robots.

## Layout

    src/tb4_multi_nav/   ROS 2 package: Gazebo worlds, robot models, launch files, envs, agents, train.py
    configs/             one YAML per algorithm (values from Table 2)
    logs/                logs/<task>/<env>/n<k>/<baseline|ucb>/seed<1..3>/training_log.csv  (96 runs)
    analysis/            make_tables.py (Tables 4, 5, 6, 8), make_figures.py (Figs 7-10), hardware_coverage.py (Table 7)
    hardware/            ONNX export, ROS nodes for the TurtleBot4, models, trial recordings
    videos/              list of trial videos
    tools/               world and robot model generator
    tests/               pytest suite

PPO and UCB-PPO are used with one robot, MAPPO and UCB-MAPPO with two to four.
16 configurations x 2 versions x 3 seeds = 96 runs.

## Reproduce the tables

    pip install -r requirements.txt
    python analysis/make_tables.py
    python analysis/make_figures.py
    python analysis/hardware_coverage.py

Metrics come from the last 50 training episodes of each run, averaged over seeds 1 to 3.
Percent changes are computed before rounding. Table 6 is computed from the rounded means of Tables 4 and 5.

## Train

    python -m tb4_multi_nav.train --task navigation --env warehouse --robots 2 --algo ucb_mappo --seed 1

Needs `PYTHONPATH=src/tb4_multi_nav`. The default backend is a fast kinematic simulator.
The Gazebo backend (`gazebo_backend.py`) needs ROS 2 and was not run.

## Status

Tested: environments, agents, training CLI, world generation, analysis scripts (33 tests).
Not tested: ROS 2 launch files, the Gazebo backend, `hardware/policy_node.py`, `logger_node.py`, anything on real robots.
Learning of the training code from this repo is not demonstrated: short single-seed runs in the kinematic
simulator did not reach goals. The numbers in the paper come from your own training runs.

Choices the paper did not fix, so check them against your code: coverage cover radius 2.0 m, zero goal
inputs for coverage, `reward_scale` 0.05 and `forward_bias` 0.5, sample standard deviation (ddof 1) for Table 7,
invented maze and shelf coordinates, and the hardware floor plan used to compute Table 7.

## Citation and license

See `CITATION.cff` (fill in names and DOI) and `LICENSE` (MIT, fill in your name).
