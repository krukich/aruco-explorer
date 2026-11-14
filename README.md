# ArUco Explorer

Autonomous ArUco marker detection and exploration using **ROS 2, Nav2, OpenCV, and Gazebo**.

ArUco Explorer is an autonomous mobile robot system that explores a known environment, detects ArUco markers using an RGB camera, estimates their positions, and returns the results through a custom ROS 2 Action.

The project uses a simulated **TurtleBot3 Waffle** robot and combines computer vision, autonomous navigation, path optimization, and environment coverage.

## Installation

The project uses the ARM Lab 05 simulation environment.

Docker commands must be executed on the **Ubuntu host**, not inside the container.

### 1. Clone the repository

```bash
cd ~
git clone https://github.com/krukich/aruco-explorer.git
```

### 2. Download the Docker environment

Create a separate directory:

```bash
mkdir -p ~/aruco-docker
cd ~/aruco-docker
```

Download the Dockerfile and build script:

```bash
wget https://raw.githubusercontent.com/kamilmlodzikowski/LabARM/main/Lab05-Problem1/Dockerfile -O Dockerfile

wget https://raw.githubusercontent.com/kamilmlodzikowski/LabARM/main/Lab05-Problem1/arm_05_build.sh -O arm_05_build.sh
```

### 3. Build the Docker image

The original build script may use the parent directory (`./..`) as its Docker build context.

Change it to the current directory (`.`):

```bash
sed -i '/docker build/s#\./\.\.#.#g' arm_05_build.sh
```

Build the image:

```bash
bash ./arm_05_build.sh
```

### 4. Download the container startup scripts

**NVIDIA GPU:**

```bash
wget https://raw.githubusercontent.com/kamilmlodzikowski/LabARM/main/Lab05-Problem1/arm_05_run_gpu_nvidia.sh -O arm_05_run_gpu_nvidia.sh
```

NVIDIA drivers and NVIDIA Container Toolkit are required.

**CPU:**

```bash
wget https://raw.githubusercontent.com/kamilmlodzikowski/LabARM/main/Lab05-Problem1/arm_05_run_cpu.sh -O arm_05_run_cpu.sh
```

### 5. Create and start the Docker container

Allow graphical applications to access the Ubuntu display:

```bash
xhost +local:root
```

Choose one startup method.

**NVIDIA GPU:**

```bash
bash ./arm_05_run_gpu_nvidia.sh
```

**CPU:**

```bash
bash ./arm_05_run_cpu.sh
```

The startup script creates a container named `ARM_05`.

### 6. Copy the project into Docker

On the Ubuntu host:

```bash
docker cp ~/aruco-explorer/aruco_nav ARM_05:/arm_ws/src/
```

### 7. Build the ROS 2 package

Enter the container:

```bash
docker exec -it ARM_05 bash
```

Build the package:

```bash
cd /arm_ws
source /opt/ros/jazzy/setup.bash

colcon build --packages-select aruco_nav --symlink-install

source /arm_ws/install/setup.bash
```

Installation is complete.

## Running the Project

The simulation, RViz, Action Server, and Action Client run in separate terminals.

Open **four terminal sessions inside the running Docker container**.

From each new Ubuntu terminal, enter the container with:

```bash
docker exec -it ARM_05 bash
```

Inside each container terminal, initialize the ROS 2 environment:

```bash
source /arm_ws/install/setup.bash
```

Then run the following commands in their respective terminals.

### Terminal 1 — Gazebo Simulation

```bash
ros2 launch arm05_sim world.launch.py
```

Wait until Gazebo, Nav2, AMCL, and the TurtleBot3 robot have started.

The simulation generates five ArUco markers at random positions in the environment.

### Terminal 2 — RViz (Optional)

```bash
ros2 launch nav2_bringup rviz_launch.py use_sim_time:=true
```

RViz can be used to monitor the robot's position, navigation, and exploration.

### Terminal 3 — Action Server

```bash
ros2 run aruco_nav search --ros-args -p use_sim_time:=true
```

The Action Server is now ready to receive exploration requests.

### Terminal 4 — Start Autonomous Exploration

```bash
ros2 action send_goal \
  /search_aruco \
  aruco_nav/action/Search \
  "{timeout_sec: 900.0}" \
  --feedback
```

The robot will autonomously navigate through the environment, detect ArUco markers, estimate their positions, and return the results.

The exploration ends when all five markers are found or the search reaches its completion condition.

## Example Result

Example of a successful search:

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

All positions are expressed in meters relative to the global `map` coordinate frame.

Marker positions change between simulation runs.

## How It Works

### 1. Environment Analysis

The robot receives an occupancy grid from `/map`.

The exploration planner generates up to 50 candidate waypoints distributed across accessible free space.

### 2. Route Optimization

The system uses:

- **Dijkstra** to estimate travel distances while accounting for obstacles.
- **Nearest Neighbor** to construct an initial exploration route.
- **2-opt** to improve the waypoint visiting order.

This provides an approximate solution to the Traveling Salesman Problem (TSP).

### 3. Autonomous Navigation

The robot navigates between selected waypoints using the Nav2 `/navigate_to_pose` Action.

Nav2 handles path planning, local obstacle avoidance, and robot movement.

### 4. Adaptive Exploration

The RGB camera continuously processes images while the robot moves.

The system maintains an approximate representation of previously observed areas.

Waypoints that are unlikely to provide useful new observations can be skipped.

The robot performs 360° scans only when additional coverage is expected.

This reduces unnecessary stops and repeated scanning.

### 5. ArUco Detection and Localization

ArUco markers are detected using OpenCV.

Their positions relative to the camera are estimated using camera calibration parameters.

TF2 transforms these positions into the global `map` coordinate frame.

Multiple observations are combined to improve localization stability.

### 6. Returning Results

The custom ROS 2 Action Server returns the detected marker IDs and estimated positions.

## ROS 2 Interfaces

### Subscribed Topics

| Topic | Purpose |
|---|---|
| `/camera/image_raw` | RGB camera images |
| `/camera/camera_info` | Camera calibration |
| `/map` | Occupancy grid |
| `/tf` | Coordinate transformations |
| `/tf_static` | Static transformations |

### Nav2 Actions

| Action | Purpose |
|---|---|
| `/navigate_to_pose` | Autonomous navigation |
| `/spin` | Rotational scanning |

### Custom Action

**Action name:**

```text
/search_aruco
```

**Action type:**

```text
aruco_nav/action/Search
```

The Action provides exploration progress feedback and returns the detected markers with their estimated positions.

## Acknowledgments

The simulation environment and Docker configuration are based on the ARM Lab 05 materials:

[LabARM — Lab05 Problem 1](https://github.com/kamilmlodzikowski/LabARM/tree/main/Lab05-Problem1)

The autonomous exploration, route optimization, ArUco detection, and ROS 2 Action implementation are provided in this repository.

## Author

**Artsiom Kruk**

Poznań University of Technology
