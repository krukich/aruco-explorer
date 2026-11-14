from pathlib import Path
import sys


import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from planner import evenly_spaced_waypoints, travel_matrix, tsp_path, route_length


def main():

    grid = np.zeros((150, 220), dtype=np.int16)
    grid[:, 110:113] = 100
    grid[65:71, 110:113] = 0
    robot = (1.0, 6.7)
    points = evenly_spaced_waypoints(grid, robot, (0, 0), 0, .1, .35, 50)
    assert len(points) == 50
    assert sum(x < 11.0 for x, y in points) > 10
    assert sum(x > 11.0 for x, y in points) > 10

    costs, metric = travel_matrix(grid, points + [robot], (0, 0), 0, .1)
    assert np.isfinite(costs).all()
    assert np.allclose(costs, costs.T)
    route = tsp_path(range(50), 50, costs)
    assert len(route) == 50 and len(set(route)) == 50
    print('50 waypoints, two rooms:', metric)
    print('Approximate tour length:', round(route_length(route, 50, costs), 2), 'm')
    print('PASS')


if __name__ == '__main__':
    main()
