import numpy as np
import matplotlib.pyplot as plt

# ================= PATHS =================
GRID_PATH = "../../data/maps/aisle_grid.npy"
GLOBAL_PATH = "../../data/global_paths/a_star_path.npy"
TRACKING_PATH = "../../data/results/astar_tracking.npy"

ARENA_HALF = 10.0
RES = 0.1


# ================= LOAD DATA =================
grid = np.load(GRID_PATH)

global_path = np.load(GLOBAL_PATH, allow_pickle=True)

try:
    tracking = np.load(TRACKING_PATH)
    tracking_available = True
except:
    tracking_available = False


# ================= GRID -> WORLD =================
def grid_to_world(path):
    pts = []
    for z, x in path:
        wx = -ARENA_HALF + x * RES
        wy = -ARENA_HALF + z * RES
        pts.append((wx, wy))
    return np.array(pts)


global_world = grid_to_world(global_path)


# ================= PLOT =================
plt.figure(figsize=(8, 8))

# Plot occupancy grid (grid[z, x])
plt.imshow(
    grid,
    cmap="Blues",
    origin="lower",
    extent=[
        -ARENA_HALF,
        ARENA_HALF,
        -ARENA_HALF,
        ARENA_HALF,
    ],
)

# Plot global A* path
plt.plot(
    global_world[:, 0],
    global_world[:, 1],
    "r-",
    linewidth=2,
    label="A* Global Path",
)

# Plot robot tracking trajectory if available
if tracking_available:
    plt.plot(
        tracking[:, 0],
        tracking[:, 1],
        "g--",
        linewidth=2,
        label="Robot Tracking",
    )

# Mark start and goal
plt.scatter(
    global_world[0, 0],
    global_world[0, 1],
    c="green",
    s=100,
    marker="o",
    label="Start",
)

plt.scatter(
    global_world[-1, 0],
    global_world[-1, 1],
    c="blue",
    s=100,
    marker="x",
    label="Goal",
)

plt.title("A* Global Path and Robot Tracking")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()