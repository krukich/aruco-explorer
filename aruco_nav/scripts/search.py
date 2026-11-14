#!/usr/bin/env python3
import math
import time
import threading
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
from planner import evenly_spaced_waypoints, coverage_anchors, travel_matrix, tsp_path, route_length
from avoidance import detour_waypoints, heading_delta
from approach import incremental_step, map_pose_safe


# ROS node and state
class SearchNode(Node):
    def __init__(self):
        super().__init__('aruco_search')
        self.declare_parameter('marker_size', 0.09)
        self.declare_parameter('view_range', 2.5)
        self.declare_parameter('min_scan_gain', 110)
        self.declare_parameter('min_scan_fraction', 0.16)
        self.declare_parameter('clearance', 0.35)
        self.declare_parameter('max_waypoints', 24)
        self.declare_parameter('priority_waypoints', 6)
        self.declare_parameter('route_budget_sec', 12.0)
        self.declare_parameter('navigate_timeout', 65.0)
        self.declare_parameter('spin_timeout', 35.0)

        self.declare_parameter('max_marker_distance', 3.2)
        self.declare_parameter('min_marker_side_px', 13.0)
        self.declare_parameter('max_reprojection_px', 2.5)
        self.declare_parameter('min_confirmations', 3)
        self.declare_parameter('cluster_radius', 0.40)

        self.declare_parameter('sighting_distance', 0.0)
        self.declare_parameter('sighting_min_side_px', 8.0)
        self.declare_parameter('sighting_reprojection_px', 5.0)
        self.declare_parameter('sighting_cluster_radius', 0.70)
        self.declare_parameter('sighting_timeout', 8.0)
        self.declare_parameter('approach_distance', 2.10)
        self.declare_parameter('cube_keepout_margin', 0.10)
        self.declare_parameter('detour_robot_clearance', 0.24)
        self.declare_parameter('approach_wait_sec', 2.4)
        self.declare_parameter('approach_max_step', 1.15)
        self.declare_parameter('camera_emergency_range', 1.15)
        self.declare_parameter('camera_sweep_rad', 0.40)
        self.declare_parameter('cube_robot_footprint', 0.16)
        self.declare_parameter('approach_timeout_sec', 24.0)
        self.declare_parameter('approach_cooldown_sec', 25.0)
        self.declare_parameter('max_approaches_per_marker', 5)

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

        self.candidates = {i: deque(maxlen=120) for i in range(5)}

        self.sighting_lock = threading.RLock()
        self.sighting_tracks = {i: deque(maxlen=10) for i in range(5)}
        self.pending_sightings = {}
        self.marker_obstacles = {}
        self.approach_blocked_until = {i: 0.0 for i in range(5)}
        self.approach_attempts = {i: 0 for i in range(5)}
        self.verifying_mid = None
        self.live_camera_ranges = {}
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
        self.verifying_mid = None

        self.candidates = {i: deque(maxlen=120) for i in range(5)}
        with self.sighting_lock:
            self.sighting_tracks = {i: deque(maxlen=10) for i in range(5)}
            self.pending_sightings = {}
            self.marker_obstacles = {}
            self.approach_blocked_until = {i: 0.0 for i in range(5)}
            self.approach_attempts = {i: 0 for i in range(5)}
            self.live_camera_ranges = {}
        if self.seen is not None:
            self.seen.fill(False)
        self.scan_cache.clear()
        self.last_coverage = 0.0
        return GoalResponse.ACCEPT

    def cancel_goal(self, goal_handle):
        return CancelResponse.ACCEPT

    # Map and camera inputs
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

    # Visual coverage
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

    def update_coverage(self, transform):
        if self.grid is None or time.monotonic() - self.last_coverage < 0.55:
            return
        self.last_coverage = time.monotonic()
        pos = transform.translation
        yaw = self.yaw_from_rotation(transform.rotation)
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

    # Marker tracking and approach
    def record_sighting(self, mid, point, distance):
        if not self.active or self.measurements[mid]:
            return
        now = time.monotonic()
        with self.sighting_lock:
            track = self.sighting_tracks[mid]
            track.append((now, point.copy(), float(distance)))
            radius = float(self.get_parameter('sighting_cluster_radius').value)
            recent = [(p, d) for ts, p, d in track
                      if now - ts < 2.5 and np.linalg.norm(p[:2] - point[:2]) < radius]
            if len(recent) < 2:
                return
            estimate = np.median(np.asarray([p for p, _ in recent]), axis=0)
            was_pending = mid in self.pending_sightings
            self.pending_sightings[mid] = (now, estimate, min(d for _, d in recent))

        if not was_pending:
            self.get_logger().info(
                f'Marker {mid} seen in 2 frames at ~{distance:.1f} m: safe incremental approach')

    def next_sighting(self):
        now = time.monotonic()
        with self.sighting_lock:
            choices = []
            for mid, (ts, xyz, dist) in self.pending_sightings.items():
                if (self.measurements[mid] or mid == self.verifying_mid or
                        now - ts > float(self.get_parameter('sighting_timeout').value)):
                    continue
                if now < self.approach_blocked_until[mid]:
                    continue
                max_attempts = int(self.get_parameter('max_approaches_per_marker').value)
                if max_attempts > 0 and self.approach_attempts[mid] >= max_attempts:
                    continue
                choices.append((dist, mid, xyz.copy()))
        return min(choices, key=lambda a: a[0]) if choices else None

    def mark_approach_finished(self, mid):
        with self.sighting_lock:
            self.approach_attempts[mid] += 1
            self.approach_blocked_until[mid] = (time.monotonic() + float(
                self.get_parameter('approach_cooldown_sec').value))
            self.pending_sightings.pop(mid, None)
            self.sighting_tracks[mid].clear()

    def cube_half_size(self):
        return (float(self.get_parameter('marker_size').value) / 2.0 +
                float(self.get_parameter('cube_keepout_margin').value))

    def inside_cube_zone(self, xy, center):
        return bool(np.max(np.abs(np.asarray(xy[:2], dtype=float) -
                                     np.asarray(center[:2], dtype=float))) <= self.cube_half_size())

    def known_cubes(self):
        with self.sighting_lock:
            return [(mid, tuple(map(float, xy)))
                    for mid, xy in self.marker_obstacles.items()]

    def marker_heading(self, marker_xy):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
        except TransformException:
            return None
        here = (tf.transform.translation.x, tf.transform.translation.y)
        yaw = self.yaw_from_rotation(tf.transform.rotation)
        return heading_delta(here, yaw, marker_xy)

    def latest_marker_hint(self, mid, fallback):
        with self.sighting_lock:
            item = self.pending_sightings.get(mid)
            if item is not None:
                ts, xyz, _ = item
                if time.monotonic() - ts < 3.0 and np.all(np.isfinite(xyz[:2])):
                    return np.asarray(xyz[:2], dtype=float).copy()
        return np.asarray(fallback[:2], dtype=float).copy()

    def recent_camera_range(self, mid):
        with self.sighting_lock:
            item = self.live_camera_ranges.get(mid)
        if item and time.monotonic() - item[0] < 1.2:
            return item[1]
        return None

    def close_visible_marker(self):
        now = time.monotonic()
        threshold = float(self.get_parameter('camera_emergency_range').value)
        with self.sighting_lock:
            return next((mid for mid, (stamp, rng) in self.live_camera_ranges.items()
                         if now - stamp < 0.65 and rng < threshold), None)

    def short_approach_pose(self, marker_xy):
        here = self.robot_position()
        if here is None or self.grid is None:
            return None
        goal_distance = float(self.get_parameter('approach_distance').value)
        max_step = float(self.get_parameter('approach_max_step').value)
        for factor in (1.0, 0.75, 0.50, 0.30):
            xy = incremental_step(here, marker_xy, goal_distance, max_step * factor)
            if xy is None:
                return None
            if not map_pose_safe(xy, self.free_cells, self.world_to_cell,
                                 self.world_map.info.resolution,
                                 clearance=float(self.get_parameter('clearance').value)):
                continue
            if any(self.inside_cube_zone(xy, pos) for _, pos in self.known_cubes()):
                continue
            return float(xy[0]), float(xy[1]), math.atan2(
                marker_xy[1] - xy[1], marker_xy[0] - xy[0])
        return None

    async def verify_sighting(self, sighting, goal, visited, total):
        _, mid, xyz = sighting
        if self.measurements[mid] or goal.is_cancel_requested:
            return
        self.mark_approach_finished(mid)
        self.verifying_mid = mid
        hint = np.asarray(xyz[:2], dtype=float).copy()
        try:

            for step_number in range(1, 5):
                if goal.is_cancel_requested or self.measurements[mid]:
                    return
                camera_distance = self.recent_camera_range(mid)
                if camera_distance is not None and camera_distance <= 2.9:
                    break
                hint = self.latest_marker_hint(mid, hint)
                pose = self.short_approach_pose(hint)
                if pose is None:
                    break
                x, y, heading = pose
                self.publish_progress(goal, visited, total,
                    f'Marker {mid}: cautious approach step {step_number}/4 '
                    f'({x:.1f}, {y:.1f}), target standoff ~2.1 m')
                result = await self.navigate(
                    x, y, goal, heading=heading, interrupt_for_marker=False,
                    timeout=float(self.get_parameter('approach_timeout_sec').value))
                if goal.is_cancel_requested or self.measurements[mid]:
                    return
                if result == 'near_marker':
                    self.get_logger().warn(
                        f'Marker {mid} optically too close; cancelling approach before contact')
                    break
                if result is not True:
                    self.get_logger().warn(
                        f'Marker {mid}: short approach step unreachable; stop approaching')
                    break
                await self.delay(0.35)

            if goal.is_cancel_requested or self.measurements[mid]:
                return

            hint = self.latest_marker_hint(mid, hint)
            camera_distance = self.recent_camera_range(mid)
            if camera_distance is not None and camera_distance < float(
                    self.get_parameter('camera_emergency_range').value):
                self.get_logger().warn(
                    f'Marker {mid}: only {camera_distance:.2f}m away; '
                    'stopping without unsafe in-place rotations')
                await self.delay(float(self.get_parameter('approach_wait_sec').value))
                return
            correction = self.marker_heading(hint)
            if correction is not None and abs(correction) > 0.10:
                await self.turn(correction, goal, interrupt_for_marker=False)
            if goal.is_cancel_requested or self.measurements[mid]:
                return
            self.publish_progress(goal, visited, total,
                                  f'Marker {mid}: stationary 3-frame scan')
            await self.delay(float(self.get_parameter('approach_wait_sec').value))
            if goal.is_cancel_requested or self.measurements[mid]:
                return

            sweep = float(self.get_parameter('camera_sweep_rad').value)
            for delta in (sweep, -2.0*sweep, sweep):
                if goal.is_cancel_requested or self.measurements[mid]:
                    return
                if self.close_visible_marker() is not None:
                    break
                await self.turn(delta, goal, interrupt_for_marker=False)
                await self.delay(1.0)
            if not self.measurements[mid]:
                self.get_logger().warn(
                    f'Marker {mid} not yet confirmed; returning to exploration '
                    'and retrying only after cooldown')
        finally:
            self.verifying_mid = None

    # Marker localization
    def add_marker_candidate(self, mid, point, distance):
        self.candidates[mid].append((point, float(distance)))
        observations = list(self.candidates[mid])
        radius = float(self.get_parameter('cluster_radius').value)
        minimum = max(1, int(self.get_parameter('min_confirmations').value))
        chosen = None
        chosen_score = float('inf')

        for seed, _ in observations:
            neighborhood = [(xyz, dist) for xyz, dist in observations
                            if np.linalg.norm(xyz[:2] - seed[:2]) <= radius]
            if len(neighborhood) < minimum:
                continue

            neighborhood.sort(key=lambda sample: sample[1])
            best = neighborhood[:max(minimum, min(15, len(neighborhood) // 2))]
            median = np.median(np.asarray([xyz for xyz, _ in best]), axis=0)
            inliers = [(xyz, dist) for xyz, dist in neighborhood
                       if np.linalg.norm(xyz[:2] - median[:2]) <= radius * 0.6]
            if len(inliers) < minimum:
                continue
            inliers.sort(key=lambda sample: sample[1])
            close = inliers[:max(minimum, min(15, len(inliers) // 2))]

            score = float(np.median([dist for _, dist in close]))
            if score < chosen_score:
                chosen_score = score
                chosen = close

        if chosen is None:
            return
        before = (np.median(np.asarray(self.measurements[mid]), axis=0)
                  if self.measurements[mid] else None)
        self.measurements[mid].clear()
        self.measurements[mid].extend(xyz for xyz, _ in chosen)
        after = np.median(np.asarray(self.measurements[mid]), axis=0)
        with self.sighting_lock:
            self.marker_obstacles[mid] = after[:2].copy()
        if before is None:
            self.get_logger().info(
                f'Confirmed marker {mid}: ({after[0]:.2f}, {after[1]:.2f}) '
                f'from {len(chosen)} samples, range ~{chosen_score:.2f} m')
        elif np.linalg.norm(before[:2] - after[:2]) > 0.20:
            self.get_logger().info(
                f'Refined marker {mid}: ({after[0]:.2f}, {after[1]:.2f}), '
                f'range ~{chosen_score:.2f} m')

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

            precise_tf = True
            try:
                tf = self.tf_buffer.lookup_transform(
                    'map', msg.header.frame_id, Time.from_msg(msg.header.stamp),
                    timeout=Duration(seconds=0.12))
            except TransformException:

                precise_tf = False
                tf = self.tf_buffer.lookup_transform(
                    'map', msg.header.frame_id, Time(),
                    timeout=Duration(seconds=0.04))
            if self.active and precise_tf:
                self.update_coverage(tf.transform)
            if ids is None:
                return
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, float(self.get_parameter('marker_size').value), self.k, self.d)
        except (TransformException, cv2.error, ValueError) as exc:
            self.get_logger().debug('Detector/TF: ' + str(exc))
            return

        marker_size = float(self.get_parameter('marker_size').value)
        half = marker_size / 2.0

        object_corners = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)
        min_side = float(self.get_parameter('min_marker_side_px').value)
        max_dist = float(self.get_parameter('max_marker_distance').value)
        max_error = float(self.get_parameter('max_reprojection_px').value)
        far_side = float(self.get_parameter('sighting_min_side_px').value)
        far_range = float(self.get_parameter('sighting_distance').value)
        far_error = float(self.get_parameter('sighting_reprojection_px').value)

        for index, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid not in self.measurements or self.measurements[mid]:
                continue
            pixel_corners = corners[index][0].astype(np.float32)
            side_lengths = np.linalg.norm(
                pixel_corners - np.roll(pixel_corners, 1, axis=0), axis=1)
            if min(side_lengths) < far_side:
                continue

            camera_xyz = np.asarray(tvecs[index][0], dtype=float)
            distance = float(np.linalg.norm(camera_xyz))
            if not np.isfinite(distance) or distance <= 0.2 or (far_range > 0 and distance > far_range):
                continue

            projected, _ = cv2.projectPoints(
                object_corners, rvecs[index][0], tvecs[index][0], self.k, self.d)
            error = float(np.mean(np.linalg.norm(
                projected.reshape(-1, 2) - pixel_corners, axis=1)))
            if not np.isfinite(error) or error > far_error:
                continue
            if self.active:
                with self.sighting_lock:
                    self.live_camera_ranges[mid] = (time.monotonic(), distance)

            x, y, z = camera_xyz

            point = self.transform_point(np.array([z, -x, -y]), tf.transform)
            if not np.all(np.isfinite(point)):
                continue

            if precise_tf:
                self.record_sighting(mid, point, distance)

            if (precise_tf and self.active and distance <= max_dist and
                    min(side_lengths) >= min_side and error <= max_error):
                self.add_marker_candidate(mid, point.copy(), distance)

    # Map coordinates and waypoints
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

    # Asynchronous navigation
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

    async def run_nav_action(self, client, message, timeout, goal,
                             interrupt_for_marker=False):
        if goal.is_cancel_requested:
            return False
        sent = client.send_goal_async(message)
        if not await self.wait_until(sent, 10.0, goal):
            return False
        handle = sent.result()
        if not handle.accepted:
            return False
        result_future = handle.get_result_async()
        started = time.monotonic()

        started_inside_zone = False
        started_camera_near = False
        if client is self.nav_client:
            started_camera_near = self.close_visible_marker() is not None
            initial_pos = self.robot_position()
            if initial_pos is not None:
                started_inside_zone = any(
                    self.inside_cube_zone(initial_pos, pos)
                    for _, pos in self.known_cubes())
        while rclpy.ok() and not result_future.done():
            timed_out = time.monotonic() - started > timeout
            completed = all(self.measurements.values())
            spotted = interrupt_for_marker and self.next_sighting() is not None
            too_close = False
            if client is self.nav_client and not started_inside_zone:
                here = self.robot_position()
                if here is not None:
                    too_close = any(self.inside_cube_zone(here, pos)
                                    for _, pos in self.known_cubes())
            camera_near = ((client is self.nav_client and not started_camera_near)
                           or client is self.spin_client) and self.close_visible_marker() is not None
            confirmed_approach = (client is self.nav_client and self.verifying_mid is not None
                                  and bool(self.measurements[self.verifying_mid]))
            if (goal.is_cancel_requested or completed or timed_out or
                    spotted or too_close or camera_near or confirmed_approach):
                try:
                    cancel_future = handle.cancel_goal_async()
                    await self.wait_until(cancel_future, 3.0, goal)

                    await self.wait_until(result_future, 3.0, goal)
                except Exception as exc:
                    self.get_logger().warn(f'Unable to cancel navigation: {exc}')
                if confirmed_approach:
                    return 'confirmed'
                if camera_near:
                    self.get_logger().warn('Live camera sees a cube <1.15m: stopping Nav2')
                    return 'near_marker'
                if too_close:
                    self.get_logger().warn('Too close to a confirmed cube; cancelling Nav2 goal')
                return 'sighting' if spotted and not goal.is_cancel_requested else False
            await self.delay(0.15)
        return (result_future.done() and
                result_future.result().status == GoalStatus.STATUS_SUCCEEDED)

    async def navigate(self, x, y, goal, heading=None,
                       interrupt_for_marker=False, timeout=None):

        start = self.robot_position()
        target = (float(x), float(y))
        segments = None
        if start is not None and self.world_map is not None:
            origin, yaw, resolution = self.map_parameters()
            segments = detour_waypoints(
                self.grid, start, target, self.known_cubes(), origin, yaw, resolution,
                cube_half_size=(self.cube_half_size() +
                    float(self.get_parameter('cube_robot_footprint').value)),
                robot_clearance=float(self.get_parameter('detour_robot_clearance').value))
        if segments == []:
            self.get_logger().warn(
                f'No safe route around a confirmed ArUco cube to ({x:.1f}, {y:.1f}); '
                'deferring waypoint (not visited)')
            return 'no_route'
        if segments is not None and len(segments) > 1:
            self.get_logger().info(
                f'Avoiding cube using {len(segments)-1} intermediate waypoint(s)')
            for wx, wy in segments[:-1]:
                state = await self._navigate_direct(
                    wx, wy, goal, interrupt_for_marker=interrupt_for_marker,
                    timeout=min(30.0, float(timeout or
                        self.get_parameter('navigate_timeout').value)))
                if state is not True:
                    return state
        return await self._navigate_direct(
            x, y, goal, heading=heading,
            interrupt_for_marker=interrupt_for_marker, timeout=timeout)

    async def _navigate_direct(self, x, y, goal, heading=None,
                               interrupt_for_marker=False, timeout=None):
        message = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        if heading is not None:
            theta = float(heading)
        else:
            current = self.robot_position()
            theta = (math.atan2(y - current[1], x - current[0])
                     if current is not None and
                     math.hypot(x - current[0], y - current[1]) > 0.05 else 0.0)
        pose.pose.orientation.z = math.sin(theta * 0.5)
        pose.pose.orientation.w = math.cos(theta * 0.5)
        message.pose = pose
        return await self.run_nav_action(
            self.nav_client, message,
            float(timeout if timeout is not None else
                  self.get_parameter('navigate_timeout').value), goal,
            interrupt_for_marker=interrupt_for_marker)

    async def turn(self, radians, goal, interrupt_for_marker=True):

        here = self.robot_position()
        map_cube_near = (here is not None and any(
            np.linalg.norm(here - np.asarray(pos[:2])) < 0.48
            for _, pos in self.known_cubes()))
        if self.close_visible_marker() is not None or map_cube_near:
            self.get_logger().warn('Skipping in-place rotation: cube too close')
            return False
        message = Spin.Goal()
        message.target_yaw = float(radians)
        message.time_allowance = Duration(seconds=float(
            self.get_parameter('spin_timeout').value)).to_msg()
        return await self.run_nav_action(self.spin_client, message,
                                         float(self.get_parameter('spin_timeout').value) + 5, goal,
                                         interrupt_for_marker=interrupt_for_marker)

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
            xyz = np.median(np.asarray(points), axis=0)
            p = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
            result.ids.append(mid)
            result.positions.append(p)
        return result

    # Search action
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
        self.get_logger().info(f'Generated {total} evenly distributed candidate stops (maximum {self.get_parameter("max_waypoints").value})')
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
        priority = set(coverage_anchors(
            points, start,
            count=int(self.get_parameter('priority_waypoints').value)))
        self.get_logger().info(
            f'Prioritizing {len(priority)} widely separated stops first, '
            'then filling coverage gaps')
        last_position_index = total
        visited = 0
        skipped = 0
        failed = 0
        temporarily_blocked = set()
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

            sighting = self.next_sighting()
            if sighting is not None:
                position_before = self.robot_position()
                await self.verify_sighting(sighting, goal, visited, total)

                here = self.robot_position()
                if here is not None:
                    if (position_before is not None and
                            np.linalg.norm(here - position_before) > 0.25):
                        temporarily_blocked.clear()
                    last_position_index = min(range(total + 1), key=lambda i:
                        np.linalg.norm(np.asarray((points + [tuple(start)])[i]) - here))
                continue

            redundant = [i for i in remaining if not self.needs_scan(*points[i])]
            for i in redundant:
                remaining.remove(i)
                priority.discard(i)
                temporarily_blocked.discard(i)
            skipped += len(redundant)
            if redundant:
                self.get_logger().info(f'Skipped {len(redundant)} already-seen stops '
                                       f'({skipped} total), {len(remaining)} left')
            if not remaining:
                break

            candidates_to_route = (priority & remaining) - temporarily_blocked
            if not candidates_to_route:
                candidates_to_route = remaining - temporarily_blocked
            if not candidates_to_route:
                self.get_logger().warn(
                    f'No navigable waypoints remain from the current pose: '
                    f'{len(temporarily_blocked)} deferred, {visited} actually reached. '
                    'Stopping instead of falsely marking all 24 points visited.')
                break
            route = tsp_path(candidates_to_route, last_position_index, costs)
            selected = route[0]
            x, y = points[selected]
            is_priority = selected in priority
            remaining.remove(selected)
            priority.discard(selected)

            self.publish_progress(goal, visited, total,
                                  f'Navigating to ({x:.1f}, {y:.1f}), {len(remaining)} candidates left')
            reached = await self.navigate(x, y, goal, interrupt_for_marker=True)
            if reached == 'sighting':

                remaining.add(selected)
                if is_priority:
                    priority.add(selected)
                self.get_logger().info('Detected a marker while driving; pausing route to approach it')
                continue
            if reached == 'near_marker':
                remaining.add(selected)
                if is_priority:
                    priority.add(selected)

                if self.next_sighting() is None:
                    temporarily_blocked.add(selected)
                self.get_logger().warn(
                    f'Waypoint ({x:.1f}, {y:.1f}) stopped for a nearby cube; '
                    'not counted as visited')
                continue
            if reached == 'no_route':

                remaining.add(selected)
                if is_priority:
                    priority.add(selected)
                temporarily_blocked.add(selected)
                self.get_logger().warn(
                    f'Waypoint ({x:.1f}, {y:.1f}) deferred; '
                    f'{len(temporarily_blocked)} temporarily blocked, '
                    f'{len(remaining)} still pending')
                continue
            if not reached:
                failed += 1
                self.get_logger().warn(
                    f'Nav2 could not reach ({x:.1f}, {y:.1f}); '
                    'counting as failed, not visited')
                continue
            visited += 1
            temporarily_blocked.clear()
            last_position_index = selected

            if all(self.measurements.values()):
                goal.succeed()
                self.publish_progress(goal, visited, total, 'Found all five markers')
                return self.make_result(True)

            if self.next_sighting() is not None:
                continue
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
            self.get_logger().warn(f'Finished: {visited} reached, {failed} failed, '
                                   f'{skipped} seen/skipped, '
                                   f'{len(temporarily_blocked)} blocked; not all markers found')
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
