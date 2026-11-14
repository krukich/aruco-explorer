import ast
from pathlib import Path
import numpy as np
import pytest
from approach import incremental_step, map_pose_safe


def test_step_maximum_distance_and_direction():
    p = incremental_step((0,0),(5,0), stand_off=2.1, max_step=1.15)
    assert np.allclose(p,(1.15,0.0))
    assert np.linalg.norm(p) <= 1.15 + 1e-9


def test_step_stops_at_standoff_never_target_center():
    p = incremental_step((0,0),(2.8,0), stand_off=2.1, max_step=1.15)
    assert np.allclose(p,(.7,0.0))
    assert np.linalg.norm(np.asarray((2.8,0))-p) == pytest.approx(2.1)
    assert incremental_step((0,0),(2.0,0),2.1,1.15) is None


def test_invalid_geometry_is_rejected():
    assert incremental_step((np.nan,0), (5,0),2.1,1.15) is None
    assert incremental_step((0,0), (5,0),2.1,0) is None


def test_wall_clearance_check():
    grid = np.ones((20,20),dtype=bool)
    to_cell = lambda x,y: (round(y), round(x))
    assert map_pose_safe((10,10),grid,to_cell,1.0,clearance=1.0)
    grid[10,11] = False
    assert not map_pose_safe((10,10),grid,to_cell,1.0,clearance=1.0)
    assert not map_pose_safe((0,0),grid,to_cell,1.0,clearance=1.0)


def test_no_multi_view_trips_and_3_frames_while_driving():
    source = (Path(__file__).parents[1]/'scripts'/'search.py').read_text()
    parsed = ast.parse(source)
    cls = next(x for x in parsed.body if isinstance(x,ast.ClassDef) and x.name=='SearchNode')
    verifier = next(x for x in cls.body if isinstance(x,ast.AsyncFunctionDef)
                    and x.name=='verify_sighting')
    assert 'capture_viewpoint' not in source
    assert 'consistent_view_samples' not in source
    assert 'self.add_marker_candidate(mid, point.copy(), distance)' in source
    assert 'incremental_step(' in source
    assert 'camera_emergency_range' in source
    assert 'for step_number in range(1, 5)' in ast.get_source_segment(source,verifier)


def test_coverage_default_is_24_with_unlimited_sightings():
    source = (Path(__file__).parents[1]/'scripts'/'search.py').read_text()
    assert "self.declare_parameter('max_waypoints', 24)" in source
    assert "self.declare_parameter('sighting_distance', 0.0)" in source


def test_confirm_without_three_nav_trips_when_already_in_camera_range():
    import asyncio
    from collections import deque
    source = (Path(__file__).parents[1]/'scripts'/'search.py').read_text()
    parsed=ast.parse(source)
    cls=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='SearchNode')
    fn=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='verify_sighting')
    ns={'np':np, 'math':__import__('math')}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'search.py','exec'),ns)
    class Fake:
        def __init__(self):
            self.measurements={2:deque()}
            self.verifying_mid=None
            self.movements=[]
        def mark_approach_finished(self,mid):pass
        def recent_camera_range(self,mid):return 2.6
        def latest_marker_hint(self,mid,hint):return np.asarray(hint[:2],dtype=float)
        def marker_heading(self,hint):return 0.0
        def publish_progress(self,*args):pass
        async def navigate(self,*args,**kwargs):
            self.movements.append(args)
            return True
        async def turn(self,*args,**kwargs):
            raise AssertionError('should not rotate after 3 good frames')
        async def delay(self,seconds):
            self.measurements[2].extend([np.array([2.0,3.0,0])]*3)
        def get_parameter(self,name):
            return type('Param',(),{'value':2.4})()
    fake=Fake()
    goal=type('Goal',(),{'is_cancel_requested':False})()
    asyncio.run(ns['verify_sighting'](fake,(2.6,2,np.array([2.0,3.0,0])),goal,0,50))
    assert len(fake.measurements[2])==3
    assert fake.movements==[]
    assert fake.verifying_mid is None


def test_new_approach_module_is_installed_by_cmake():
    cmake=(Path(__file__).parents[1]/'CMakeLists.txt').read_text()
    assert 'scripts/approach.py' in cmake
    assert 'scripts/viewpoints.py' not in cmake
