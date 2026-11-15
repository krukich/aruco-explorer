#!/usr/bin/env python3
import os
import random
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Graphics and resources


def _graphics_environment():
    actions = [
        SetEnvironmentVariable(
            'QT_QPA_PLATFORM', os.environ.get('QT_QPA_PLATFORM', 'xcb')
        ),
        SetEnvironmentVariable(
            'QT_X11_NO_MITSHM', os.environ.get('QT_X11_NO_MITSHM', '1')
        ),
    ]
    authority = os.environ.get('XAUTHORITY', '')
    if not Path(authority).is_file():
        for candidate in (Path('/tmp/arm05.xauth'), Path.home() / '.Xauthority'):
            if candidate.is_file():
                actions.append(SetEnvironmentVariable('XAUTHORITY', str(candidate)))
                break
        else:
            if authority:
                actions.append(UnsetEnvironmentVariable('XAUTHORITY'))

    vendor = Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
    software = os.environ.get('LIBGL_ALWAYS_SOFTWARE', '').lower() in ('1', 'true')
    if vendor.is_file() and list(Path('/dev').glob('nvidia[0-9]*')) and not software:
        actions.extend(
            [
                SetEnvironmentVariable(
                    '__GLX_VENDOR_LIBRARY_NAME',
                    os.environ.get('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
                ),
                SetEnvironmentVariable(
                    '__EGL_VENDOR_LIBRARY_FILENAMES',
                    os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES', str(vendor)),
                ),
            ]
        )
    return actions


def _resource_path(name, paths):
    existing = os.environ.get(name, '')
    return os.pathsep.join([*paths, existing] if existing else paths)


def _include(directory, filename, **arguments):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(directory, filename)),
        launch_arguments=arguments.items(),
    )


# Simulation and navigation


def generate_launch_description():
    simulation = get_package_share_directory('arm05_sim')
    turtlebot = get_package_share_directory('turtlebot3_gazebo')
    gazebo = os.path.join(get_package_share_directory('ros_gz_sim'), 'launch')
    navigation = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')
    robot_launch = os.path.join(turtlebot, 'launch')
    models = [os.path.join(simulation, 'models'), os.path.join(turtlebot, 'models')]
    model = os.environ.get('TURTLEBOT3_MODEL', 'waffle')
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = os.path.join(simulation, 'worlds', 'arm_house.world')

    environment = _graphics_environment()
    for name in (
        'GAZEBO_MODEL_PATH',
        'GZ_SIM_RESOURCE_PATH',
        'IGN_GAZEBO_RESOURCE_PATH',
    ):
        environment.append(SetEnvironmentVariable(name, _resource_path(name, models)))

    return LaunchDescription(
        [
            DeclareLaunchArgument('use_sim_time', default_value='true'),
            DeclareLaunchArgument('x_pose', default_value='0.0'),
            DeclareLaunchArgument('y_pose', default_value='0.0'),
            DeclareLaunchArgument('turtlebot3_model', default_value=model),
            DeclareLaunchArgument(
                'aruco_seed', default_value=str(random.randrange(10_000_000))
            ),
            SetEnvironmentVariable(
                'TURTLEBOT3_MODEL', LaunchConfiguration('turtlebot3_model')
            ),
            *environment,
            _include(
                gazebo,
                'gz_sim.launch.py',
                gz_args=f'-r -s -v2 {world}',
                on_exit_shutdown='true',
            ),
            _include(
                gazebo, 'gz_sim.launch.py', gz_args='-g -v2', on_exit_shutdown='true'
            ),
            _include(
                robot_launch,
                'robot_state_publisher.launch.py',
                use_sim_time=use_sim_time,
            ),
            _include(
                robot_launch,
                'spawn_turtlebot3.launch.py',
                x_pose=LaunchConfiguration('x_pose'),
                y_pose=LaunchConfiguration('y_pose'),
                turtlebot3_model=LaunchConfiguration('turtlebot3_model'),
            ),
            TimerAction(
                period=5.0,
                actions=[
                    Node(
                        package='arm05_sim',
                        executable='spawn_aruco.py',
                        namespace='spawn_cubes',
                        output='screen',
                        arguments=['--seed', LaunchConfiguration('aruco_seed')],
                    )
                ],
            ),
            _include(
                navigation,
                'bringup_launch.py',
                map=os.path.join(simulation, 'map', 'map.yaml'),
                params_file=os.path.join(simulation, 'param', 'waffle.yaml'),
                use_sim_time=use_sim_time,
                autostart='true',
                use_composition='False',
            ),
        ]
    )
