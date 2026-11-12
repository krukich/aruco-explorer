"""ROS-independent coverage waypoint and obstacle-aware open-TSP planner."""

import heapq
import math
import time

import cv2
import numpy as np


def _world_to_cell(x, y, origin, yaw, resolution):
    dx, dy = x - origin[0], y - origin[1]
    co, si = math.cos(yaw), math.sin(yaw)
    return ((-si * dx + co * dy) / resolution,
            (co * dx + si * dy) / resolution)


def _cell_to_world(row, col, origin, yaw, resolution):
    xx, yy = (col + 0.5) * resolution, (row + 0.5) * resolution
    co, si = math.cos(yaw), math.sin(yaw)
    return (origin[0] + co * xx - si * yy,
            origin[1] + si * xx + co * yy)


def evenly_spaced_waypoints(grid, start, origin, yaw, resolution,
                            clearance=0.35, count=50):
    """Farthest-point sampling over robot-reachable free map cells.

    The points are spatially dispersed, not an exact uniform square lattice.
    Only the connected free-space component nearest to the robot is used.
    """
    empty = np.asarray(grid)
    free = (empty >= 0) & (empty < 20)
    if not np.any(free) or count <= 0:
        return []

    # The robot has a radius: candidate navigation goals need wall clearance.
    dist_to_wall = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    clear_px = max(1, int(math.ceil(clearance / resolution)))
    good = free & (dist_to_wall >= clear_px)
    if not np.any(good):
        return []

    rr, cc = _world_to_cell(start[0], start[1], origin, yaw, resolution)
    good_rc = np.column_stack(np.nonzero(good))
    nearest = int(np.argmin((good_rc[:, 0] - rr) ** 2 +
                            (good_rc[:, 1] - cc) ** 2))
    seed_r, seed_c = good_rc[nearest]

    # Connectivity is evaluated with LESS clearance than navigation goals.
    # Otherwise a 0.6 m doorway may artificially split two reachable rooms.
    passable = free & (dist_to_wall >= max(1, int(math.ceil(0.10 / resolution))))
    _, components = cv2.connectedComponents(passable.astype(np.uint8), connectivity=8)
    group = good & (components == components[seed_r, seed_c])

    # A 0.25 m lattice limits candidate count but remains finer than stop spacing.
    stride = max(1, int(round(0.25 / resolution)))
    sample_mask = np.zeros_like(group, dtype=bool)
    sample_mask[stride // 2::stride, stride // 2::stride] = True
    rc = np.column_stack(np.nonzero(group & sample_mask))
    if len(rc) < count:
        rc = np.column_stack(np.nonzero(group))
        if len(rc) > 12000:
            rc = rc[::int(math.ceil(len(rc) / 12000))]
    if len(rc) == 0:
        return []

    xy = np.column_stack((rc[:, 1], rc[:, 0])).astype(np.float64) * resolution
    p_start = np.array([cc, rr]) * resolution
    nearest2 = np.sum((xy - p_start) ** 2, axis=1)
    selected = []
    min_spacing2 = max(0.45, clearance) ** 2
    for _ in range(min(count, len(rc))):
        i = int(np.argmax(nearest2))
        if nearest2[i] < min_spacing2:
            break
        selected.append(i)
        nearest2 = np.minimum(nearest2, np.sum((xy - xy[i]) ** 2, axis=1))
        nearest2[i] = -1.0
    return [_cell_to_world(int(rc[i, 0]), int(rc[i, 1]), origin, yaw, resolution)
            for i in selected]


def _closest_passable(passable, rc, radius=5):
    rr, cc = rc
    h, w = passable.shape
    r0, r1 = max(0, int(rr) - radius), min(h, int(rr) + radius + 1)
    c0, c1 = max(0, int(cc) - radius), min(w, int(cc) + radius + 1)
    cells = np.column_stack(np.nonzero(passable[r0:r1, c0:c1]))
    if len(cells) == 0:
        return None
    cells[:, 0] += r0
    cells[:, 1] += c0
    j = np.argmin((cells[:, 0] - rr) ** 2 + (cells[:, 1] - cc) ** 2)
    return tuple(int(v) for v in cells[j])


def _dijkstra_to_targets(passable, src, targets, resolution):
    """Shortest 8-connected grid distances, preventing diagonal corner cuts."""
    h, w = passable.shape
    distances = np.full((h, w), np.inf, dtype=np.float64)
    destinations = {}
    for idx, cell in enumerate(targets):
        if cell is not None:
            destinations.setdefault(cell, []).append(idx)
    out = np.full(len(targets), np.inf, dtype=np.float64)
    if src is None:
        return out
    distances[src] = 0.0
    todo = [(0.0, src[0], src[1])]
    steps = [(-1, 0, resolution), (1, 0, resolution),
             (0, -1, resolution), (0, 1, resolution),
             (-1, -1, resolution * math.sqrt(2)),
             (-1, 1, resolution * math.sqrt(2)),
             (1, -1, resolution * math.sqrt(2)),
             (1, 1, resolution * math.sqrt(2))]
    while todo and destinations:
        cost, row, col = heapq.heappop(todo)
        if cost > distances[row, col]:
            continue
        if (row, col) in destinations:
            for idx in destinations.pop((row, col)):
                out[idx] = cost
        for dr, dc, length in steps:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < h and 0 <= nc < w and passable[nr, nc]):
                continue
            if dr and dc and not (passable[row + dr, col] and passable[row, col + dc]):
                continue
            new_cost = cost + length
            if new_cost < distances[nr, nc]:
                distances[nr, nc] = new_cost
                heapq.heappush(todo, (new_cost, nr, nc))
    return out


def travel_matrix(grid, locations, origin, yaw, resolution, max_seconds=12.0):
    """Approximate Nav2 path distances for [waypoints..., initial robot pose].

    Dijkstra on a reduced occupancy grid; geometric fallback if reduced map
    disconnects a corridor or a computation budget is exceeded.
    """
    n = len(locations)
    pts = np.asarray(locations, dtype=np.float64)
    if n < 2:
        return np.zeros((n, n), dtype=np.float64), 'trivial'
    eucl = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    step = max(1, int(round(0.20 / resolution)))
    free = ((np.asarray(grid) >= 0) & (np.asarray(grid) < 20)).astype(np.uint8)
    h, w = free.shape
    coarse_h = max(1, math.ceil(h / step))
    coarse_w = max(1, math.ceil(w / step))
    # Mean free occupancy: solid walls block routes without filling corridors.
    coarse = cv2.resize(free.astype(np.float32), (coarse_w, coarse_h),
                        interpolation=cv2.INTER_AREA) >= 0.78
    coarse_res = resolution * step
    cells = []
    for x, y in locations:
        row, col = _world_to_cell(x, y, origin, yaw, resolution)
        cells.append(_closest_passable(coarse, (round(row / step), round(col / step))))

    matrix = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    started = time.monotonic()
    complete = True
    for i in range(n):
        if time.monotonic() - started > max_seconds:
            complete = False
            break
        row = _dijkstra_to_targets(coarse, cells[i], cells, coarse_res)
        matrix[i, :] = row
    # Pairwise distances must be symmetric for our 2-opt optimization.
    for i in range(n):
        for j in range(i + 1, n):
            a, b = matrix[i, j], matrix[j, i]
            if math.isfinite(a) and math.isfinite(b):
                distance = 0.5 * (a + b)
            elif math.isfinite(a):
                distance = a
            elif math.isfinite(b):
                distance = b
            else:
                # Discourage an unverified straight line through a wall.
                distance = 3.5 * eucl[i, j] + 2.0
            matrix[i, j] = matrix[j, i] = distance
    return matrix, ('grid' if complete else 'grid+fallback')


def route_length(route, start_index, costs):
    current = start_index
    total = 0.0
    for i in route:
        total += costs[current, i]
        current = i
    return total


def tsp_path(indices, start_index, costs, max_rounds=40):
    """Nearest-neighbor + 2-opt OPEN TSP (no return-to-origin)."""
    not_visited = set(indices)
    route = []
    current = start_index
    while not_visited:
        nxt = min(not_visited, key=lambda k: (costs[current, k], k))
        route.append(nxt)
        not_visited.remove(nxt)
        current = nxt

    for _ in range(max_rounds):
        improved = False
        for a in range(len(route) - 1):
            before = start_index if a == 0 else route[a - 1]
            first = route[a]
            for b in range(a + 1, len(route)):
                last = route[b]
                after = route[b + 1] if b + 1 < len(route) else None
                old = costs[before, first]
                new = costs[before, last]
                if after is not None:
                    old += costs[last, after]
                    new += costs[first, after]
                if new + 1e-6 < old:
                    route[a:b + 1] = reversed(route[a:b + 1])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return route
