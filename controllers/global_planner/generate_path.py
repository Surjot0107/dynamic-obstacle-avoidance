import numpy as np
import os
from a_star import AStarPlanner

GRID_PATH = "../../data/maps/aisle_grid.npy"
SAVE_PATH = "../../data/global_paths/a_star_path.npy"

# Ensure save directory exists
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Load grid
grid = np.load(GRID_PATH)

start = (40, 40)
goal  = (120, 120)

planner = AStarPlanner(grid)
path = planner.plan(start, goal)

if path is None:
    raise RuntimeError("No path found")

np.save(SAVE_PATH, np.array(path))

print("Global path saved successfully.")