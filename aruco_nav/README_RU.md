# ArUco Nav — ROS 2 Jazzy, ARM Lab05

**v3:** New package/action names, at most 50 evenly distributed candidate stops, obstacle-aware route via reduced occupancy grid, nearest-neighbor + 2-opt (open TSP), live visual-coverage memory, skipping redundant stops and spins.

## Installation

**Ubuntu host** (outside Docker):

```bash
cd ~/Downloads
unzip -o aruco_nav_v3.zip
docker cp aruco_nav ARM_05:/arm_ws/src/
```

**Inside running ARM_05 container** (keep Gazebo/Nav2 running):

```bash
cd /arm_ws
colcon build --packages-select aruco_nav --symlink-install
source /arm_ws/install/setup.bash
ros2 interface show aruco_nav/action/Search
ros2 run aruco_nav search --ros-args -p use_sim_time:=true
```

**Separate container terminal:**

```bash
source /arm_ws/install/setup.bash
ros2 action send_goal /search_aruco aruco_nav/action/Search "{timeout_sec: 900.0}" --feedback
```

Do not run the container creation script again: it may delete your container. Stop the old `arm05_search` server with Ctrl+C before starting the new server. The old package can coexist on disk until the new one is verified.

## Interface

- Package: `aruco_nav`
- Node: `aruco_search`
- Executable: `search`
- Action type: `aruco_nav/action/Search`
- Action topic: `/search_aruco`
- Action goal: `{timeout_sec: 900.0}`
- Result: `success`, `ids`, `positions` (as before)
- Feedback: `found_count`, `visited_count`, `total_waypoints`, `state`.

## Approach

1. Read `/map`; sample up to 50 spread-out safe locations on the reachable free space. The farthest-point selection covers free areas, **not** an exact uniform rectangular layout.
2. Compute a pairwise approximate travel-distance matrix with Dijkstra on a downsampled occupancy map, respecting most walls and corridors. If map reduction loses a corridor, some pairs receive a penalized Euclidean fallback; Nav2 still chooses the actual driving path.
3. Find a short **open** route (no forced return to start) using nearest-neighbor + 2-opt. This is **not** an exact optimal TSP solver.
4. Process camera frames while driving and turning. Store a visibility map of free cells in the camera field-of-view (within ~2.5 m, respecting visible map obstacles). This is an *approximate observation model*, not proof that a marker is absent.
5. As the robot sees more of the environment, exclude redundant candidate stops *before sending a Nav2 goal*, reorder the remaining tour, and only turn through 360 degrees at useful stops.
6. Save found marker IDs 0–4 with globally transformed coordinates and return them through the ROS2 Action.

The algorithm stops immediately when it has detected all five markers (after finishing the current Nav2 action), or when time/route is exhausted. If an Action fails after covering the map, reduce the scan thresholds and/or increase candidate count.

## Configurable parameters

All via `ros2 run ... --ros-args -p key:=value`:

| Parameter | Default | Description |
|---|---:|---|
| `max_waypoints` | `50` | Maximum candidate stop points (not all necessarily visited) |
| `clearance` | `0.35` | Minimum goal clearance from wall, meters |
| `view_range` | `2.5` | Assumed useful camera viewing range, meters |
| `min_scan_gain` | `110` | Minimum unviewed free map cells to justify stop/scan |
| `min_scan_fraction` | `0.16` | Minimum unviewed share of a scan's map cells |
| `marker_size` | `0.09` | ArUco marker size assumed in simulation, meters |
| `route_budget_sec` | `12.0` | Computation budget for road-distance matrix, wall seconds |
| `navigate_timeout` | `90.0` | Per-goal navigation timeout |
| `spin_timeout` | `35.0` | Per-spin timeout |

If the new algorithm misses cubes, prioritize completeness with e.g.:

```bash
ros2 run aruco_nav search --ros-args -p use_sim_time:=true \
  -p min_scan_gain:=60 -p min_scan_fraction:=0.08
```

To stop more rarely, increase `min_scan_gain` or `min_scan_fraction`; this increases the risk of overlooking a marker. All detection uses the camera. The ground-truth `latest.txt` file is **not** read by the program.

## Validation

ROS 2 itself is not installed in the environment in which this archive was produced. The code was syntax-checked and the standalone planner tested on synthetic two-room/doorway maps; full runtime behavior should be verified in the running Jazzy/Gazebo container.
