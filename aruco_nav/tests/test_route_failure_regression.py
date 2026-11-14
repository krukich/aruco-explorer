import ast
from collections import deque
from pathlib import Path
import sys
import threading


import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from avoidance import detour_waypoints, intersects_cube_square


def test_robot_can_escape_previously_confirmed_keepout():

    grid = np.zeros((120, 160), dtype=np.int16)
    start, cube, target = (7.02, 6.03), (7.0, 6.0), (13.0, 6.0)
    result = detour_waypoints(grid, start, target, [(4, cube)],
                              origin=(0, 0), yaw=0.0, resolution=0.1,
                              cube_half_size=0.145)
    assert result is not None and result != []
    assert np.linalg.norm(np.asarray(result[0]) - np.asarray(cube)) > .145
    assert not intersects_cube_square(result[0], result[0], cube, .145)
    for a, b in zip(result, result[1:]):
        assert not intersects_cube_square(a, b, cube, .145)
    assert np.linalg.norm(np.asarray(result[-1]) - np.asarray(target)) < .01


def test_unconfirmed_far_sighting_does_not_block_navigation():
    source = Path(__file__).resolve().parents[1] / 'scripts' / 'search.py'
    parsed = ast.parse(source.read_text())
    klass = next(node for node in parsed.body
                 if isinstance(node, ast.ClassDef) and node.name == 'SearchNode')
    method = next(node for node in klass.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'record_sighting')
    ns = {'np': np, 'time': __import__('time')}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)

    class Fake:
        active = True
        verifying_mid = None
        def __init__(self):
            self.measurements = {0: deque()}
            self.sighting_lock = threading.RLock()
            self.sighting_tracks = {0: deque(maxlen=10)}
            self.pending_sightings = {}
            self.marker_obstacles = {}
        def get_parameter(self, name):
            assert name == 'sighting_cluster_radius'
            return type('Param', (), {'value': .7})()
        def get_logger(self):
            return type('Log', (), {'info': lambda *args: None})()

    instance = Fake()

    ns['record_sighting'](instance, 0, np.array([6.2, -2.2, 0]), 4.0)
    assert 0 not in instance.pending_sightings
    ns['record_sighting'](instance, 0, np.array([6.21, -2.21, 0]), 3.9)
    assert 0 in instance.pending_sightings
    assert instance.marker_obstacles == {}


def test_no_route_not_miscounted_as_visited():
    source = (Path(__file__).resolve().parents[1] / 'scripts' / 'search.py').read_text()
    assert "return 'no_route'" in source
    assert "if reached == 'no_route':" in source
    assert 'temporarily_blocked.add(selected)' in source

    assert 'visited += 1\n            temporarily_blocked.clear()' in source
