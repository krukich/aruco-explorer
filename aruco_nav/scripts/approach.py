import numpy as np
import math


# Safe approach geometry
def incremental_step(robot_xy, marker_xy, stand_off, max_step):
    a = np.asarray(robot_xy[:2], dtype=float)
    b = np.asarray(marker_xy[:2], dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    length = float(np.linalg.norm(b - a))
    if length <= float(stand_off) + 0.10 or max_step <= 0:
        return None
    step = min(float(max_step), length - float(stand_off))
    if step <= 0.10:
        return None
    return a + (b - a) / length * step


def map_pose_safe(xy, free_cells, cell_fn, resolution, clearance=0.35):
    if free_cells is None or resolution <= 0:
        return False
    row, col = cell_fn(float(xy[0]), float(xy[1]))
    radius = max(1, int(math.ceil(clearance / resolution)))
    rows, cols = free_cells.shape
    if not (radius <= row < rows - radius and radius <= col < cols - radius):
        return False
    return bool(
        np.all(
            free_cells[row - radius : row + radius + 1, col - radius : col + radius + 1]
        )
    )
