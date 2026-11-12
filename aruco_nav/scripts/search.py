#!/usr/bin/env python3
import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from rclpy.task import Future
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from aruco_nav.action import Search
from planner import evenly_spaced_waypoints, travel_matrix, tsp_path, route_length


class SearchNode(Node):
    def __init__(self):
        super().__init__('aruco_search')
        self.declare_parameter('marker_size', 0.09)
        self.declare_parameter('view_range', 2.5)
        self.declare_parameter('min_scan_gain', 110)
        self.declare_parameter('min_scan_fraction', 0.16)
        self.declare_parameter('clearance', 0.35)
        self.declare_parameter('max_waypoints', 50)
        self.declare_parameter('route_budget_sec', 12.0)
        self.declare_parameter('navigate_timeout', 90.0)
        self.declare_parameter('spin_timeout', 35.0)

        self.group = ReentrantCallbackGroup()
        self.bridge = CvBridge()
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = (cv2.aruco.DetectorParameters_create()
                           if hasattr(cv2.aruco, 'DetectorParameters_create')
                           else cv2.aruco.DetectorParameters())
        self.detector = (cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
                         if hasattr(cv2.aruco, 'ArucoDetector') else None)
        self.k = None
        self.d = None
        self.world_map = None
        self.seen = None
        self.scan_cache = {}
        self.grid = None
        self.free_cells = None
        self.last_coverage = 0.0
        self.camera_fov = math.radians(70)
        self.measurements = {i: deque(maxlen=20) for i in range(5)}
        self.last_image = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos,
                                 callback_group=self.group)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback,
                                 qos_profile_sensor_data, callback_group=self.group)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback,
                                 qos_profile_sensor_data, callback_group=self.group)

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose',
                                       callback_group=self.group)
        self.spin_client = ActionClient(self, Spin, '/spin', callback_group=self.group)
        self.server = ActionServer(self, Search, '/search_aruco',
                                   execute_callback=self.execute,
                                   goal_callback=self.accept_goal,
                                   cancel_callback=self.cancel_goal,
                                   callback_group=self.group)
        self.active = False
        self.get_logger().info('Ready: /search_aruco (waiting for goal)')

    def accept_goal(self, request):
        if self.active:
            self.get_logger().warn('Search already running; rejecting concurrent goal')
            return GoalResponse.REJECT
        self.active = True
        self.measurements = {i: deque(maxlen=20) for i in range(5)}
        if self.seen is not None:
            self.seen.fill(False)
        self.scan_cache.clear()
        self.last_coverage = 0.0
        return GoalResponse.ACCEPT

    def cancel_goal(self, goal_handle):
        return CancelResponse.ACCEPT

    def map_callback(self, msg):
        if self.seen is None or self.seen.shape != (msg.info.height, msg.info.width):
            self.seen = np.zeros((msg.info.height, msg.info.width), dtype=np.bool_)
            self.scan_cache.clear()
        self.world_map = msg
        self.grid = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        self.free_cells = (self.grid >= 0) & (self.grid < 20)

    def info_callback(self, msg):
        if msg.k[0] > 0 and msg.width > 0:
            self.camera_fov = 2.0 * math.atan(msg.width / (2.0 * msg.k[0]))
        self.k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.d = np.array(msg.d, dtype=np.float64)

    @staticmethod
    def transform_point(point, transform):
        q = transform.rotation
        v = np.array([q.x, q.y, q.z])
        p = point + 2 * q.w * np.cross(v, point) + 2 * np.cross(v, np.cross(v, point))
        return p + np.array([transform.translation.x,
                             transform.translation.y, transform.translation.z])

    def world_to_cell(self, x, y):
        info = self.world_map.info
        origin = info.origin.position
        q = info.origin.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y*q.y + q.z*q.z))
        dx, dy = x - origin.x, y - origin.y
        c, s = math.cos(yaw), math.sin(yaw)
        col = int(math.floor((c*dx + s*dy) / info.resolution))
        row = int(math.floor((-s*dx + c*dy) / info.resolution))
        return row, col

    @staticmethod
    def yaw_from_rotation(q):
        return math.atan2(2.0 * (q.w*q.z + q.x*q.y),
                          1.0 - 2.0 * (q.y*q.y + q.z*q.z))

    def visible_cells(self, x, y, yaw, fov, rays=35):
        if self.grid is None:
            return set()
        distance = float(self.get_parameter('view_range').value)
        resolution = self.world_map.info.resolution
        step = max(resolution, 0.12)
        cells = set()
        rows, cols = self.grid.shape
        for angle in np.linspace(yaw - fov / 2.0, yaw + fov / 2.0, rays):
            cs, sn = math.cos(angle), math.sin(angle)
            for radius in np.arange(0.12, distance + step, step):
                row, col = self.world_to_cell(x + radius * cs, y + radius * sn)
                if row < 0 or row >= rows or col < 0 or col >= cols:
                    break
                if not self.free_cells[row, col]:
                    break
                cells.add((row, col))
        return cells

    def update_coverage(self, frame_id):
        if self.grid is None or time.monotonic() - self.last_coverage < 0.55:
            return
        self.last_coverage = time.monotonic()
        try:
            tf = self.tf_buffer.lookup_transform('map', frame_id, Time())
        except TransformException:
            return
        pos = tf.transform.translation
        yaw = self.yaw_from_rotation(tf.transform.rotation)
        for row, col in self.visible_cells(pos.x, pos.y, yaw, self.camera_fov):
            self.seen[row, col] = True

    def scan_gain(self, x, y):
        if self.seen is None:
            return 0, 0.0
        key = (round(x, 3), round(y, 3))
        if key not in self.scan_cache:
            visible = self.visible_cells(x, y, 0.0, 2.0 * math.pi, rays=95)
            self.scan_cache[key] = (np.asarray(list(visible), dtype=np.int32).reshape(-1, 2)
                                    if visible else np.zeros((0, 2), dtype=np.int32))
        coords = self.scan_cache[key]
        if len(coords) == 0:
            return 0, 0.0
        unseen = int(np.count_nonzero(~self.seen[coords[:, 0], coords[:, 1]]))
        return unseen, unseen / len(coords)

    def needs_scan(self, x, y):
        cells, fraction = self.scan_gain(x, y)
        enough_cells = cells >= int(self.get_parameter('min_scan_gain').value)
        enough_fraction = fraction >= float(self.get_parameter('min_scan_fraction').value)
        return enough_cells and enough_fraction

    def image_callback(self, msg):
        if self.k is None or time.monotonic() - self.last_image < 0.25:
            return
        self.last_image = time.monotonic()
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.detector is not None:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self.dictionary, parameters=self.parameters)
            # Only images actually analyzed by OpenCV count toward visual coverage.
            if self.active:
                self.update_coverage(msg.header.frame_id)
            if ids is None:
                return
            tf = self.tf_buffer.lookup_transform('map', msg.header.frame_id, Time())
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, float(self.get_parameter('marker_size').value), self.k, self.d)
        except (TransformException, cv2.error, ValueError) as exc:
            self.get_logger().debug('Detector: ' + str(exc))
            return

        for index, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid not in self.measurements:
                continue
            if cv2.contourArea(corners[index][0].astype(np.float32)) < 100:
                continue
            x, y, z = tvecs[index][0]
            point = self.transform_point(np.array([z, -x, -y]), tf.transform)
            if not np.all(np.isfinite(point)):
                continue
            first = not self.measurements[mid]
            self.measurements[mid].append(point)
            if first:
                self.get_logger().info(f'Found marker {mid}: ({point[0]:.2f}, {point[1]:.2f})')

    def robot_position(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
            return np.array([tf.transform.translation.x, tf.transform.translation.y])
        except TransformException:
            return None

    def map_parameters(self):
        info = self.world_map.info
        q = info.origin.orientation
        origin_yaw = self.yaw_from_rotation(q)
        return (info.origin.position.x, info.origin.position.y), origin_yaw, info.resolution

    def make_waypoints(self, start):
        origin, yaw, resolution = self.map_parameters()
        return evenly_spaced_waypoints(
            self.grid, start, origin, yaw, resolution,
            clearance=float(self.get_parameter('clearance').value),
            count=int(self.get_parameter('max_waypoints').value))

    async def delay(self, seconds):
        future = Future()
        def finish():
            if not future.done():
                future.set_result(True)
        timer = self.create_timer(seconds, finish, callback_group=self.group)
        try:
            await future
        finally:
            self.destroy_timer(timer)

    async def wait_until(self, future, timeout, goal, stop_if_complete=False):
        start = time.monotonic()
        while rclpy.ok() and not future.done():
            if (goal.is_cancel_requested or
                    (stop_if_complete and all(self.measurements.values())) or
                    time.monotonic() - start > timeout):
                return False
            await self.delay(0.1)
        return future.done()

    async def run_nav_action(self, client, message, timeout, goal):
        if goal.is_cancel_requested:
            return False
        sent = client.send_goal_async(message)
        if not await self.wait_until(sent, 10.0, goal):
            return False
        handle = sent.result()
        if not handle.accepted:
            return False
        result_future = handle.get_result_async()
        if not await self.wait_until(result_future, timeout, goal, stop_if_complete=True):
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
            return False
        return result_future.result().status == GoalStatus.STATUS_SUCCEEDED

    async def navigate(self, x, y, goal):
        message = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        current = self.robot_position()
        if current is not None and math.hypot(x - current[0], y - current[1]) > 0.05:
            theta = math.atan2(y - current[1], x - current[0])
            pose.pose.orientation.z = math.sin(theta * 0.5)
            pose.pose.orientation.w = math.cos(theta * 0.5)
        else:
            pose.pose.orientation.w = 1.0
        message.pose = pose
        return await self.run_nav_action(
            self.nav_client, message,
            float(self.get_parameter('navigate_timeout').value), goal)

    async def turn(self, radians, goal):
        message = Spin.Goal()
        message.target_yaw = float(radians)
        message.time_allowance = Duration(seconds=float(
            self.get_parameter('spin_timeout').value)).to_msg()
        return await self.run_nav_action(self.spin_client, message,
                                         float(self.get_parameter('spin_timeout').value) + 5, goal)

    def publish_progress(self, goal, visited, total, state):
        feedback = Search.Feedback()
        feedback.found_count = sum(bool(v) for v in self.measurements.values())
        feedback.visited_count = visited
        feedback.total_waypoints = total
        feedback.state = state
        goal.publish_feedback(feedback)
        self.get_logger().info(f'{state} | found {feedback.found_count}/5 | visited {visited}/{total}')

    def make_result(self, success):
        result = Search.Result()
        result.success = bool(success)
        for mid, points in self.measurements.items():
            if len(points) == 0:
                continue
            xyz = np.mean(points, axis=0)
            p = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
            result.ids.append(mid)
            result.positions.append(p)
        return result

    async def execute(self, goal):
        self.get_logger().info('Search started')
        try:
            return await self.search(goal)
        except Exception as exc:
            self.get_logger().error(f'Search failed: {exc}')
            goal.abort()
            return self.make_result(False)
        finally:
            self.active = False

    async def search(self, goal):
        limit = float(goal.request.timeout_sec)
        if limit <= 0:
            limit = 900.0
        started = time.monotonic()

        while self.world_map is None or self.k is None or self.robot_position() is None:
            if goal.is_cancel_requested:
                goal.canceled()
                return self.make_result(False)
            if time.monotonic() - started > min(limit, 40.0):
                goal.abort()
                self.get_logger().error('No map, camera_info or robot TF')
                return self.make_result(False)
            await self.delay(0.2)

        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            goal.abort()
            self.get_logger().error('Nav2 /navigate_to_pose unavailable')
            return self.make_result(False)
        has_spin = self.spin_client.wait_for_server(timeout_sec=5.0)
        if not has_spin:
            self.get_logger().warn('Nav2 /spin unavailable; only moving-camera coverage will be used')

        start = self.robot_position()
        points = self.make_waypoints(start)
        total = len(points)
        self.get_logger().info(f'Generated {total} evenly distributed candidate stops (maximum 50)')
        if total == 0:
            goal.abort()
            self.get_logger().error('No reachable safe waypoints in /map')
            return self.make_result(False)

        origin, yaw, resolution = self.map_parameters()
        self.publish_progress(goal, 0, total, 'Planning obstacle-aware TSP route')
        costs, metric = travel_matrix(
            self.grid, points + [tuple(start)], origin, yaw, resolution,
            max_seconds=float(self.get_parameter('route_budget_sec').value))
        initial = tsp_path(range(total), total, costs)
        length = route_length(initial, total, costs)
        self.get_logger().info(f'Route planned: {metric}, open TSP 2-opt, estimated length {length:.1f} m')

        remaining = set(range(total))
        last_position_index = total
        visited = 0
        skipped = 0
        if has_spin and self.needs_scan(*start):
            self.publish_progress(goal, visited, total, 'Initial 360-degree scan')
            await self.turn(2.0 * math.pi, goal)

        while remaining and time.monotonic() - started < limit:
            if goal.is_cancel_requested:
                goal.canceled()
                return self.make_result(False)
            if all(self.measurements.values()):
                goal.succeed()
                self.publish_progress(goal, visited, total, 'Found all five markers')
                return self.make_result(True)

            # Seen-from-camera cells are stored while driving and turning.
            # If a point is already visually covered, don't even stop there.
            redundant = [i for i in remaining if not self.needs_scan(*points[i])]
            for i in redundant:
                remaining.remove(i)
            skipped += len(redundant)
            if redundant:
                self.get_logger().info(f'Skipped {len(redundant)} already-seen stops '
                                       f'({skipped} total), {len(remaining)} left')
            if not remaining:
                break

            # Recompute a short OPEN tour for remaining stops, no return to start.
            # Distances are based on approximate shortest paths around obstacles.
            route = tsp_path(remaining, last_position_index, costs)
            selected = route[0]
            x, y = points[selected]
            remaining.remove(selected)

            self.publish_progress(goal, visited, total,
                                  f'Navigating to ({x:.1f}, {y:.1f}), {len(remaining)} candidates left')
            reached = await self.navigate(x, y, goal)
            visited += 1
            if not reached:
                self.get_logger().warn(f'Nav2 could not reach ({x:.1f}, {y:.1f}); skipping')
                continue
            last_position_index = selected

            if all(self.measurements.values()):
                goal.succeed()
                self.publish_progress(goal, visited, total, 'Found all five markers')
                return self.make_result(True)
            cells, fraction = self.scan_gain(x, y)
            if has_spin and self.needs_scan(x, y):
                self.publish_progress(goal, visited, total,
                                      f'Visual scan: {cells} unseen cells ({fraction:.0%})')
                await self.turn(2.0 * math.pi, goal)
            else:
                self.get_logger().info(
                    f'No rotation at stop: unseen {cells} cells, {fraction:.0%} of reachable view')
            await self.delay(0.2)

        success = all(self.measurements.values())
        if goal.is_cancel_requested:
            goal.canceled()
        elif success:
            goal.succeed()
        else:
            goal.abort()
            self.get_logger().warn(f'Finished with {visited} navigation goals and '
                                   f'{skipped} skipped stops; not all markers were found')
        self.publish_progress(goal, visited, total, 'Finished')
        return self.make_result(success)


def main():
    rclpy.init()
    node = SearchNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
