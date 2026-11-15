#!/usr/bin/env bash
set -eo pipefail

workspace="${1:-/arm_ws}"
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$workspace/src/aruco_nav"

if [[ ! -f "$workspace/src/arm05_sim/package.xml" ]]; then
    echo "Use the LabARM Lab 05 container with arm05_sim already installed." >&2
    exit 1
fi

mkdir -p "$package_dir"
if [[ "$(realpath "$project_dir/aruco_nav")" != "$(realpath "$package_dir")" ]]; then
    cp -a "$project_dir/aruco_nav/." "$package_dir/"
fi

source /opt/ros/jazzy/setup.bash
if [[ -f "$workspace/install/setup.bash" ]]; then
    source "$workspace/install/setup.bash"
fi

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    rosdep init
fi
if [[ ! -d "$HOME/.ros/rosdep/sources.cache" ]]; then
    rosdep update --rosdistro jazzy
fi
rosdep install --from-paths "$package_dir" --ignore-src --rosdistro jazzy -y

cd "$workspace"
colcon build --packages-select aruco_nav --symlink-install
