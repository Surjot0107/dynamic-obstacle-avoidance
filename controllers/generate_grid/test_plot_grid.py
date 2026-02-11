import numpy as np
import matplotlib.pyplot as plt
import os

# Load occupancy grid
grid = np.load("../../data/maps/aisle_grid.npy")

# Create output directory if it doesn't exist
output_dir = "../../data/maps"
os.makedirs(output_dir, exist_ok=True)

# Plot
plt.figure(figsize=(6, 6))
plt.imshow(grid, origin="lower", cmap="Blues")
plt.title("Aisle Layout Occupancy Grid")
plt.xlabel("X cells")
plt.ylabel("Z cells")

# Save figure
output_path = os.path.join(output_dir, "aisle_occupancy_grid.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()