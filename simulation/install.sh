#!/usr/bin/env bash
set -eo pipefail

workspace="${1:-/arm_ws}"
simulation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$workspace/src/arm05_sim"

if [[ ! -f "$package_dir/package.xml" ]]; then
    echo "arm05_sim was not found in $workspace/src. Install the LabARM environment first." >&2
    exit 1
fi

source /opt/ros/jazzy/setup.bash
if [[ -f "$workspace/install/setup.bash" ]]; then
    source "$workspace/install/setup.bash"
fi

cp -a "$simulation_dir/arm05_sim/." "$package_dir/"
cd "$workspace"
colcon build --packages-select arm05_sim --symlink-install
