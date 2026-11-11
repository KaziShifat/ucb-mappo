"""Environment geometry, one source of truth.

The kinematic simulator, the Gazebo world generator (tools/generate_worlds.py)
and the tests all read from here, so the arenas cannot drift apart.

Maze:      18 x 18 m, ten horizontal and ten vertical wall segments of 3 m.
Warehouse: 20 x 20 m, 15 shelves of 3.0 x 0.5 m in three columns of five,
           aisles between the columns about 3 m wide.
Robots start in the four corners and drive to the diagonally opposite corner.

The wall positions of the maze and the shelf positions are NOT taken from your
Gazebo files. They are a layout written to match the paper's description.
Replace them with the real coordinates before comparing numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

WALL_T = 0.2  # wall thickness (m)


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    sx: float
    sy: float

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.cx - self.sx / 2, self.cy - self.sy / 2,
                self.cx + self.sx / 2, self.cy + self.sy / 2)


@dataclass(frozen=True)
class Layout:
    name: str
    size: float
    interior: Tuple[Box, ...]
    starts: Tuple[Tuple[float, float], ...]
    goals: Tuple[Tuple[float, float], ...]
    wall_height: float = 1.0
    interior_height: float = 1.0

    @property
    def half(self) -> float:
        return self.size / 2

    @property
    def diagonal(self) -> float:
        return math.hypot(self.size, self.size)

    def outer_walls(self) -> Tuple[Box, ...]:
        h, s, t = self.half, self.size, WALL_T
        return (Box(0, h + t / 2, s + 2 * t, t), Box(0, -h - t / 2, s + 2 * t, t),
                Box(h + t / 2, 0, t, s), Box(-h - t / 2, 0, t, s))

    def all_boxes(self) -> Tuple[Box, ...]:
        return self.outer_walls() + self.interior

    def start_pose(self, i: int) -> Tuple[float, float, float]:
        x, y = self.starts[i]
        return x, y, math.atan2(-y, -x)          # facing the arena centre


_MAZE_H = [(-7.5, -3), (-7.5, 6), (-4.5, -6), (-1.5, -6), (-1.5, 0),
           (1.5, -6), (1.5, -3), (1.5, 6), (4.5, 0), (4.5, 6)]
_MAZE_V = [(-3, -4.5), (-3, -1.5), (-3, 1.5), (-3, 4.5), (0, 7.5),
           (3, -7.5), (6, -4.5), (6, 1.5), (6, 4.5), (6, 7.5)]


def maze() -> Layout:
    walls = tuple(Box(x, y, 3.0, WALL_T) for x, y in _MAZE_H) + \
            tuple(Box(x, y, WALL_T, 3.0) for x, y in _MAZE_V)
    c = 7.0
    return Layout("maze", 18.0, walls, ((-c, -c), (c, -c), (-c, c), (c, c)),
                  ((c, c), (-c, c), (c, -c), (-c, -c)))


def warehouse() -> Layout:
    shelves = tuple(Box(x, y, 0.5, 3.0) for x in (-3.5, 0.0, 3.5) for y in (-8, -4, 0, 4, 8))
    c = 8.0
    return Layout("warehouse", 20.0, shelves, ((-c, -c), (c, -c), (-c, c), (c, c)),
                  ((c, c), (-c, c), (c, -c), (-c, -c)), interior_height=1.6)


LAYOUTS = {"maze": maze, "warehouse": warehouse}


def get_layout(name: str) -> Layout:
    if name not in LAYOUTS:
        raise ValueError(f"unknown environment {name!r}; choose from {sorted(LAYOUTS)}")
    return LAYOUTS[name]()
