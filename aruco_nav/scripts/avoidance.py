import heapq
import math


import cv2
import numpy as np


# Cube geometry
def intersects_cube_square(start, end, center, half_size):
    a = np.asarray(start[:2], dtype=float)
    b = np.asarray(end[:2], dtype=float)
    c = np.asarray(center[:2], dtype=float)
    delta = b - a
    t0, t1 = 0.0, 1.0
    for axis in range(2):
        lo, hi = c[axis] - half_size, c[axis] + half_size
        if abs(delta[axis]) < 1e-12:
            if a[axis] < lo or a[axis] > hi:
                return False
            continue
        enter = (lo - a[axis]) / delta[axis]
        leave = (hi - a[axis]) / delta[axis]
        if enter > leave:
            enter, leave = leave, enter
        t0 = max(t0, enter)
        t1 = min(t1, leave)
        if t0 > t1:
            return False
    return True


def near_route(start, end, markers, half_size):
    return any(intersects_cube_square(start, end, pos, half_size) for _, pos in markers)


# Grid pathfinding
def _cell(xy, origin, yaw, resolution):
    dx, dy = float(xy[0]) - origin[0], float(xy[1]) - origin[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return int(math.floor((-s * dx + c * dy) / resolution)), int(
        math.floor((c * dx + s * dy) / resolution)
    )


def _world(row, col, origin, yaw, resolution):
    x, y = (col + 0.5) * resolution, (row + 0.5) * resolution
    c, s = math.cos(yaw), math.sin(yaw)
    return float(origin[0] + c * x - s * y), float(origin[1] + s * x + c * y)


def _nearest_passable(passable, rc, max_radius):
    rows, cols = passable.shape
    r, c = rc
    for radius in range(max_radius + 1):
        y0, y1 = max(0, r - radius), min(rows, r + radius + 1)
        x0, x1 = max(0, c - radius), min(cols, c + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        cells = np.argwhere(passable[y0:y1, x0:x1])
        if len(cells):
            cells += [y0, x0]
            i = np.argmin((cells[:, 0] - r) ** 2 + (cells[:, 1] - c) ** 2)
            return tuple(map(int, cells[i]))
    return None


def _visible(a, b, allowed):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    h, w = allowed.shape
    for alpha in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) * 2.0) + 1)):
        r, c = np.rint(a + alpha * (b - a)).astype(int)
        if r < 0 or c < 0 or r >= h or c >= w or not allowed[r, c]:
            return False
    return True


def _astar(allowed, start, goal, max_expansions=170000):
    h, w = allowed.shape
    if start == goal:
        return [start]
    moves = [
        (-1, 0, 1),
        (1, 0, 1),
        (0, -1, 1),
        (0, 1, 1),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]
    queue = [(math.dist(start, goal), 0.0, start)]
    previous = {}
    costs = {start: 0.0}
    expansions = 0
    while queue and expansions < max_expansions:
        _, cost, cell = heapq.heappop(queue)
        if cost > costs.get(cell, float('inf')) + 1e-9:
            continue
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return list(reversed(path))
        expansions += 1
        r, c = cell
        for dr, dc, dist in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w and allowed[nr, nc]):
                continue
            if dr and dc and not (allowed[r + dr, c] and allowed[r, c + dc]):
                continue
            nxt = nr, nc
            new = cost + dist
            if new + 1e-9 < costs.get(nxt, float('inf')):
                costs[nxt] = new
                previous[nxt] = cell
                heapq.heappush(queue, (new + math.dist(nxt, goal), new, nxt))
    return None


# Virtual obstacle avoidance
def detour_waypoints(
    grid,
    start,
    end,
    markers,
    origin,
    yaw,
    resolution,
    cube_half_size=0.145,
    robot_clearance=0.24,
    max_leg=3.5,
):
    if not markers or not near_route(start, end, markers, cube_half_size):
        return None

    if any(
        np.max(
            np.abs(np.asarray(end[:2], dtype=float) - np.asarray(pos[:2], dtype=float))
        )
        <= cube_half_size
        for _, pos in markers
    ):
        return []
    grid = np.asarray(grid)
    free = ((grid >= 0) & (grid < 20)).astype(np.uint8)
    if not np.any(free):
        return []
    wall_dist = cv2.distanceTransform(np.pad(free, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    allowed = ((free > 0) & (wall_dist * resolution >= robot_clearance)).astype(
        np.uint8
    )

    edge = int(math.ceil((robot_clearance + 0.35) / resolution))
    allowed[:edge, :] = allowed[-edge:, :] = 0
    allowed[:, :edge] = allowed[:, -edge:] = 0

    for _, pos in markers:
        x, y = float(pos[0]), float(pos[1])

        h = float(cube_half_size) + resolution * 0.65
        polygon = []
        for xx, yy in ((x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)):
            row, col = _cell((xx, yy), origin, yaw, resolution)
            polygon.append((col, row))
        cv2.fillConvexPoly(allowed, np.asarray(polygon, dtype=np.int32), 0)
    inside_start = any(
        intersects_cube_square(start, start, pos, cube_half_size) for _, pos in markers
    )

    sr = _nearest_passable(
        allowed,
        _cell(start, origin, yaw, resolution),
        max(2, int((0.75 if inside_start else 0.35) / resolution)),
    )
    gr = _nearest_passable(
        allowed, _cell(end, origin, yaw, resolution), max(2, int(0.20 / resolution))
    )
    if sr is None or gr is None:
        return []
    path = _astar(allowed, sr, gr)
    if not path:
        return []

    out = []
    at = 0
    max_cells = max(2, int(max_leg / resolution))
    while at < len(path) - 1:
        far = at + 1
        for j in range(at + 2, len(path)):
            if math.dist(path[at], path[j]) > max_cells or not _visible(
                path[at], path[j], allowed
            ):
                break
            far = j
        out.append(_world(*path[far], origin, yaw, resolution))
        at = far
        if len(out) > 45:
            return []
    if inside_start:
        escape = _world(*sr, origin, yaw, resolution)

        for _, pos in markers:
            if intersects_cube_square(start, start, pos, cube_half_size):
                old_depth = np.max(np.abs(np.asarray(start[:2]) - np.asarray(pos[:2])))
                new_depth = np.max(np.abs(np.asarray(escape[:2]) - np.asarray(pos[:2])))
                if new_depth <= old_depth + 1e-6:
                    return []
        out.insert(0, escape)

    if out and math.dist(out[-1], end) < 0.22:
        out[-1] = tuple(map(float, end[:2]))
    else:
        out.append(tuple(map(float, end[:2])))

    legs = [tuple(map(float, start[:2]))] + out
    for i, (a, b) in enumerate(zip(legs, legs[1:])):
        if i == 0 and inside_start:

            continue
        if near_route(a, b, markers, cube_half_size):
            return []
    return out


# Camera heading
def heading_delta(robot_xy, robot_yaw, marker_xy):
    desired = math.atan2(marker_xy[1] - robot_xy[1], marker_xy[0] - robot_xy[0])
    return math.atan2(math.sin(desired - robot_yaw), math.cos(desired - robot_yaw))
