"""
Map utilities — occupancy grid data structures, A* pathfinding, and frontiers.

Map format:
    - 2D NumPy array:  0 = free,  1 = obstacle,  -1 = unknown
    - Resolution: meters per cell
    - Origin: (x, y) world-coordinate of grid cell (0, 0)

Saved/loaded as .npz files with keys: grid, resolution, origin_x, origin_y.
"""

import heapq
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OccupancyMap:
    """A 2D occupancy grid map."""
    grid: np.ndarray         # (rows, cols) int8: 0=free, 1=obstacle, -1=unknown
    resolution: float        # Meters per cell
    origin_x: float          # World x-coordinate of grid cell (0, 0)
    origin_y: float          # World y-coordinate of grid cell (0, 0)

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]


def create_empty_map(
    size_cells: int = 200,
    resolution: float = 0.05,
) -> OccupancyMap:
    """
    Create an empty occupancy map centered on the origin.

    The robot starts at the center of the grid.
    """
    grid = np.full((size_cells, size_cells), -1, dtype=np.int8)
    # Origin is placed so that the center of the grid is world (0, 0).
    half = (size_cells * resolution) / 2.0
    return OccupancyMap(
        grid=grid,
        resolution=resolution,
        origin_x=-half,
        origin_y=-half,
    )


def save_map(omap: OccupancyMap, path: str) -> None:
    """Save an occupancy map to a .npz file."""
    np.savez_compressed(
        path,
        grid=omap.grid,
        resolution=np.array([omap.resolution]),
        origin_x=np.array([omap.origin_x]),
        origin_y=np.array([omap.origin_y]),
    )
    logger.info("Map saved to %s (%d x %d)", path, omap.rows, omap.cols)


def load_map(path: str) -> OccupancyMap:
    """Load an occupancy map from a .npz file."""
    data = np.load(path)
    omap = OccupancyMap(
        grid=data["grid"],
        resolution=float(data["resolution"][0]),
        origin_x=float(data["origin_x"][0]),
        origin_y=float(data["origin_y"][0]),
    )
    logger.info("Map loaded from %s (%d x %d)", path, omap.rows, omap.cols)
    return omap


# ------------------------------------------------------------------
# Coordinate conversion
# ------------------------------------------------------------------

def world_to_grid(omap: OccupancyMap, x: float, y: float) -> Tuple[int, int]:
    """Convert world coordinates (x, y) to grid cell (row, col)."""
    col = int((x - omap.origin_x) / omap.resolution)
    row = int((y - omap.origin_y) / omap.resolution)
    return (row, col)


def grid_to_world(omap: OccupancyMap, row: int, col: int) -> Tuple[float, float]:
    """Convert grid cell (row, col) to world coordinates (x, y)."""
    x = omap.origin_x + (col + 0.5) * omap.resolution
    y = omap.origin_y + (row + 0.5) * omap.resolution
    return (x, y)


def in_bounds(omap: OccupancyMap, row: int, col: int) -> bool:
    """Check if a grid cell is within the map bounds."""
    return 0 <= row < omap.rows and 0 <= col < omap.cols


# ------------------------------------------------------------------
# Obstacle inflation
# ------------------------------------------------------------------

def inflate_obstacles(
    grid: np.ndarray, radius_cells: int
) -> np.ndarray:
    """
    Inflate obstacles by a circular radius.

    Returns a new grid where cells within radius_cells of any obstacle
    are also marked as obstacles. Unknown cells (-1) are treated as
    obstacles for safety.
    """
    from scipy.ndimage import binary_dilation

    # Create a binary obstacle mask (obstacles + unknown = True).
    obstacle_mask = (grid >= 1) | (grid == -1)

    # Create a circular structuring element.
    size = 2 * radius_cells + 1
    y, x = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    struct = (x * x + y * y) <= (radius_cells * radius_cells)

    inflated_mask = binary_dilation(obstacle_mask, structure=struct)

    result = grid.copy()
    result[inflated_mask & (grid == 0)] = 1  # Mark newly inflated cells as obstacle
    return result


# ------------------------------------------------------------------
# A* pathfinding
# ------------------------------------------------------------------

# 8-connected neighbors: (delta_row, delta_col, cost).
_NEIGHBORS_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
]


def astar(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    A* search on a 2D occupancy grid with 8-connected neighbors.

    grid: 2D array where 0 = free, anything else = blocked.
    start: (row, col)
    goal:  (row, col)

    Returns a list of (row, col) waypoints from start to goal (inclusive),
    or None if no path exists.
    """
    rows, cols = grid.shape
    sr, sc = start
    gr, gc = goal

    # Quick sanity checks.
    if not (0 <= sr < rows and 0 <= sc < cols):
        return None
    if not (0 <= gr < rows and 0 <= gc < cols):
        return None
    if grid[sr, sc] != 0 or grid[gr, gc] != 0:
        return None

    def heuristic(r, c):
        return ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5

    # open_set: (f_score, counter, row, col)
    counter = 0
    open_set = [(heuristic(sr, sc), counter, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}
    closed = set()

    while open_set:
        _, _, r, c = heapq.heappop(open_set)

        if (r, c) in closed:
            continue
        closed.add((r, c))

        if r == gr and c == gc:
            # Reconstruct path.
            path = [(gr, gc)]
            node = (gr, gc)
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            return path

        for dr, dc, cost in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] != 0:
                continue
            if (nr, nc) in closed:
                continue

            tentative_g = g_score[(r, c)] + cost
            if tentative_g < g_score.get((nr, nc), float("inf")):
                came_from[(nr, nc)] = (r, c)
                g_score[(nr, nc)] = tentative_g
                f = tentative_g + heuristic(nr, nc)
                counter += 1
                heapq.heappush(open_set, (f, counter, nr, nc))

    return None  # No path found.


# ------------------------------------------------------------------
# Frontier detection
# ------------------------------------------------------------------

def find_frontiers(grid: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find frontier cells: free cells adjacent to unknown cells.

    Frontiers are the boundary between explored free space and
    unexplored territory — good targets for exploration.
    """
    rows, cols = grid.shape
    frontiers = []

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r, c] != 0:
                continue
            # Check 4-connected neighbors for unknown cells.
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if grid[nr, nc] == -1:
                    frontiers.append((r, c))
                    break

    return frontiers


# ------------------------------------------------------------------
# Bresenham ray casting (for map_builder.py)
# ------------------------------------------------------------------

def bresenham(
    r0: int, c0: int, r1: int, c1: int
) -> List[Tuple[int, int]]:
    """
    Bresenham's line algorithm.

    Returns a list of (row, col) cells from (r0, c0) to (r1, c1),
    NOT including the endpoint.
    """
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        if r == r1 and c == c1:
            break
        cells.append((r, c))
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return cells
