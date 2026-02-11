import numpy as np
import matplotlib.pyplot as plt
from a_star import AStarPlanner


def plot_grid_and_path(grid, path, start, goal):
    plt.figure(figsize=(8, 8))

    # Plot occupancy grid (grid[z, x])
    plt.imshow(grid, cmap="Blues", origin="lower")

    # Plot path
    if path is not None:
        rows, cols = zip(*path)   # rows = z, cols = x
        plt.plot(cols, rows, 'r-', linewidth=2, label="A* Path")

    # Plot start and goal
    plt.scatter(start[1], start[0], c='green', s=100, marker='o', label="Start")
    plt.scatter(goal[1], goal[0], c='blue', s=100, marker='x', label="Goal")

    plt.legend()
    plt.xlabel("X cells")
    plt.ylabel("Z cells")
    plt.title("A* Path on Occupancy Grid")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load occupancy grid (already in [z, x] form)
    grid = np.load("../../data/maps/aisle_grid.npy")

    # Define start and goal (z, x)
    start = (40, 40)
    goal = (120, 120)

    # Plan directly on grid (NO transpose)
    planner = AStarPlanner(grid)
    path = planner.plan(start, goal)

    if path is None:
        print("No path found")
    else:
        print(f"Path found with {len(path)} points")
        plot_grid_and_path(grid, path, start, goal)
