from pathlib import Path
import sys


import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from planner import evenly_spaced_waypoints, coverage_anchors, travel_matrix, tsp_path


def test_wide_area_first_pass():

    grid = np.zeros((130, 240), dtype=np.int16)
    grid[:, 118:122] = 100
    grid[53:65, 118:122] = 0
    start = (2.0, 6.0)
    points = evenly_spaced_waypoints(grid, start, (0, 0), 0.0, 0.1,
                                    clearance=0.35, count=24)
    assert len(points) == 24, len(points)
    anchors = coverage_anchors(points, start, count=6)
    assert len(anchors) == 6 and len(set(anchors)) == 6
    assert any(points[i][0] < 11.8 for i in anchors)
    assert any(points[i][0] > 12.2 for i in anchors)
    costs, _ = travel_matrix(grid, points + [start], (0, 0), 0.0, 0.1, max_seconds=15.0)
    route = tsp_path(anchors, len(points), costs)
    assert set(route) == set(anchors)
    assert any(points[i][0] > 12.2 for i in route[:6])
    print(f'PASS: {len(points)} total candidates, {len(anchors)} broad-first-pass anchors')


if __name__ == '__main__':
    test_wide_area_first_pass()
