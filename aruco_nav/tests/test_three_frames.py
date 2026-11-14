import ast
from collections import deque
from pathlib import Path
import threading
import numpy as np


def test_three_frame_confirmation():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'search.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SearchNode')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'add_marker_candidate')
    fn.decorator_list = []
    ns = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)

    class Fake:
        def __init__(self):
            self.candidates = {1: deque(maxlen=120)}
            self.measurements = {1: deque(maxlen=20)}
            self.marker_obstacles = {}
            self.sighting_lock = threading.RLock()
        def get_parameter(self, name):
            values = {'cluster_radius': .35, 'min_confirmations': 3}
            return type('Param', (), {'value': values[name]})()
        def get_logger(self):
            return type('Log', (), {'info': lambda self, msg: None})()

    fake = Fake()
    for p in [(1,2,0), (1.01,1.98,.01)]:
        ns['add_marker_candidate'](fake, 1, np.asarray(p,dtype=float), 1.4)
        assert not fake.measurements[1]
    ns['add_marker_candidate'](fake, 1, np.asarray((1.02,2.01,.00)), 1.3)
    assert len(fake.measurements[1]) == 3
    assert np.linalg.norm(np.median(np.asarray(fake.measurements[1]),axis=0)[:2]
                          - np.asarray((1,2))) < .03
