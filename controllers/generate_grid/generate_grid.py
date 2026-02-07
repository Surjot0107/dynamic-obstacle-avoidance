from controller import Supervisor
import numpy as np
from scipy.ndimage import binary_dilation
import os

# ================= CONFIG =================
RESOLUTION = 0.1
X_MIN, X_MAX = -10.0, 10.0
Z_MIN, Z_MAX = -10.0, 10.0

ROBOT_RADIUS = 0.105
SAFETY_MARGIN = 0.05
INFLATION_RADIUS = ROBOT_RADIUS + SAFETY_MARGIN
# =========================================

sup = Supervisor()
timestep = int(sup.getBasicTimeStep())

width  = int((X_MAX - X_MIN) / RESOLUTION)
height = int((Z_MAX - Z_MIN) / RESOLUTION)
grid = np.zeros((width, height), dtype=np.uint8)

def world_to_grid(x, z):
    return (
        int((x - X_MIN) / RESOLUTION),
        int((z - Z_MIN) / RESOLUTION)
    )

# ---------- MANUAL SHELF DEFINITIONS ----------
# (x, z, size_x, size_z)
SHELVES = [
    ( 4,  0, 6, 2),
    (-4,  0, 6, 2),
    ( 4,  4, 6, 2),
    (-4,  4, 6, 2),
    ( 4, -4, 6, 2),
    (-4, -4, 6, 2),
    ( 4,  7.5, 6, 1),
    (-4,  7.5, 6, 1),
    ( 4, -7.5, 6, 1),
    (-4, -7.5, 6, 1),
]

# Rasterize shelves
for x, z, sx, sz in SHELVES:
    x_min = x - sx / 2
    x_max = x + sx / 2
    z_min = z - sz / 2
    z_max = z + sz / 2

    gx0, gz0 = world_to_grid(x_min, z_min)
    gx1, gz1 = world_to_grid(x_max, z_max)

    for gx in range(max(0, gx0), min(width, gx1)):
        for gz in range(max(0, gz0), min(height, gz1)):
            grid[gx, gz] = 1

# Arena boundaries
grid[0, :] = 1
grid[-1, :] = 1
grid[:, 0] = 1
grid[:, -1] = 1

# Inflate obstacles
inflation_cells = int(INFLATION_RADIUS / RESOLUTION)
kernel = np.ones((2*inflation_cells+1, 2*inflation_cells+1))
grid = binary_dilation(grid, structure=kernel)

# Save
save_dir = os.path.abspath(os.path.join(os.getcwd(), "../../data/maps"))
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "aisle_grid.npy")
np.save(save_path, grid)

print("Occupancy grid generated successfully")
sup.step(timestep)
sup.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
