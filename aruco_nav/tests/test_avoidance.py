from pathlib import Path
import math
import sys
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from avoidance import detour_waypoints, near_route, intersects_cube_square


HALF = 0.045 + 0.10


def test_exact_square_size():
    cube = (7.0, 6.0)

    assert intersects_cube_square((6.86, 6.0), (6.86, 6.0), cube, HALF)
    assert not intersects_cube_square((6.84, 6.16), (6.84, 6.16), cube, HALF)
    assert not intersects_cube_square((6.7, 6.16), (7.3, 6.16), cube, HALF)
    assert near_route((6.7, 6), (7.3, 6), [(1, cube)], HALF)


def test_virtual_cube_detour():
    grid = np.zeros((120, 160), dtype=np.int16)
    start, target, cube = (2.0, 6.0), (13.0, 6.0), (7.0, 6.0)
    result = detour_waypoints(grid, start, target, [(1, cube)],
                              origin=(0, 0), yaw=0.0, resolution=0.1,
                              cube_half_size=HALF)
    assert result is not None and result != []
    for a, b in zip([start] + result, result):
        assert not intersects_cube_square(a, b, cube, HALF)
    assert math.dist(result[-1], target) < .01
    assert detour_waypoints(grid, start, (5.0, 6.0), [(1, cube)],
                            origin=(0, 0), yaw=0, resolution=0.1,
                            cube_half_size=HALF) is None


def test_doorway_and_no_route():
    grid = np.zeros((120, 160), dtype=np.int16)
    grid[:, 80:82] = 100
    grid[54:68, 80:82] = 0
    start, goal, cube = (3.0, 6.0), (14.0, 6.0), (6.0, 6.0)
    result = detour_waypoints(grid, start, goal, [(0, cube)],
                              origin=(0, 0), yaw=0, resolution=0.1,
                              cube_half_size=HALF)
    assert result
    for a, b in zip([start] + result, result):
        assert not intersects_cube_square(a, b, cube, HALF)


def test_blocked_goal():
    grid = np.zeros((120, 160), dtype=np.int16)
    result = detour_waypoints(grid, (2, 6), (7, 6), [(3, (7, 6))],
                              origin=(0, 0), yaw=0, resolution=0.1,
                              cube_half_size=HALF)
    assert result == []
