# Aruco-explorer

Autonomous ArUco marker detection and exploration using ROS 2 and Nav2.

## Requirements

- Ubuntu with Docker
- ARM Lab 05 Docker image (`arm/lab05`)
- ROS 2 Jazzy
- Gazebo Harmonic
- Nav2

## Installation

### 1. Clone the repository

Run on Ubuntu:

```bash
git clone https://github.com/krukich/aruco-explorer.git
cd aruco-explorer
```

### 2. Copy the package into Docker

Make sure the `ARM_05` container is running.

```bash
docker cp aruco_nav ARM_05:/arm_ws/src/
```

### 3. Build the package

Enter the container:

```bash
docker exec -it ARM_05 bash
```

Build:

```bash
cd /arm_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select aruco_nav --symlink-install
source install/setup.bash
```

## Running

Open four terminals.

### Terminal 1 — Gazebo simulation

Inside the container:

```bash
cd /arm_ws
source install/setup.bash
ros2 launch arm05_sim world.launch.py
```

### Terminal 2 — RViz (optional)

```bash
docker exec -it ARM_05 bash
source /arm_ws/install/setup.bash
ros2 launch nav2_bringup rviz_launch.py use_sim_time:=true
```

### Terminal 3 — Action Server

```bash
docker exec -it ARM_05 bash
source /arm_ws/install/setup.bash
ros2 run aruco_nav search --ros-args -p use_sim_time:=true
```

### Terminal 4 — Start autonomous exploration

```bash
docker exec -it ARM_05 bash
source /arm_ws/install/setup.bash
ros2 action send_goal /search_aruco aruco_nav/action/Search "{timeout_sec: 900.0}" --feedback
```

## Result

The robot autonomously explores the environment, detects all five ArUco markers, and returns their IDs and positions in the `map` coordinate frame.

Example:

```yaml
Result:
  success: true
  ids:
  - 0
  - 1
  - 2
  - 3
  - 4
  positions:
  - x: 1.769
    y: 3.566
    z: -0.005
  - x: 7.644
    y: 5.247
    z: 0.018
  - x: 6.067
    y: 2.945
    z: 0.013
  - x: 12.050
    y: 0.290
    z: -0.004
  - x: 7.657
    y: -2.574
    z: -0.007

Goal finished with status: SUCCEEDED
```

Marker positions are randomly generated in each simulation run.
