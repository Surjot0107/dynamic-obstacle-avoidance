"""
Supervisor that drives OBS1-OBS5 (Pedestrian nodes) as dynamic obstacles.

Each pedestrian gets a random velocity at startup and bounces off arena walls.
Only obstacles with z > 0 are moved — buried pedestrians (curriculum phases 1 & 2)
are left alone, which is how the TurtleBot env's curriculum system gates them.

Pedestrians are moved kinematically (no physics engine involvement) and
resetPhysics() is called each step to prevent the physics engine from
fighting the kinematic position updates.
"""

from controller import Supervisor
import math
import numpy as np

TIME_STEP = 64
ARENA_LIMIT = 4.5   # bounce boundary, slightly inside the 5m wall
PEDESTRIAN_Z = 1.25  # Pedestrian PROTO origin height — same as dynamic-obstacle-aisle.wbt
DT = TIME_STEP / 1000.0

robot = Supervisor()

# Discover all five pedestrian obstacle nodes and assign random velocities.
# getFromDef() uses the DEF name (uppercase), which differs from the 'name'
# field (lowercase) used for the Pedestrian's internal label.
obstacles = []
for i in range(1, 6):
    node = robot.getFromDef(f"OBS{i}")
    if not node:
        print(f"Warning: DEF OBS{i} not found in scene")
        continue

    speed = np.random.uniform(0.3, 0.7)
    angle = np.random.uniform(0, 2 * math.pi)
    obstacles.append({
        "node":  node,
        "trans": node.getField("translation"),
        "rot":   node.getField("rotation"),
        "vx":    speed * math.cos(angle),
        "vy":    speed * math.sin(angle),
    })

while robot.step(TIME_STEP) != -1:
    for obs in obstacles:
        pos = obs["trans"].getSFVec3f()

        # Skip pedestrians buried underground (inactive curriculum phase)
        if pos[2] < 0:
            continue

        new_x = pos[0] + obs["vx"] * DT
        new_y = pos[1] + obs["vy"] * DT

        # Bounce off arena walls — reverse only the offending velocity component
        if new_x > ARENA_LIMIT or new_x < -ARENA_LIMIT:
            obs["vx"] *= -1
            new_x = max(-ARENA_LIMIT, min(ARENA_LIMIT, new_x))
        if new_y > ARENA_LIMIT or new_y < -ARENA_LIMIT:
            obs["vy"] *= -1
            new_y = max(-ARENA_LIMIT, min(ARENA_LIMIT, new_y))

        obs["trans"].setSFVec3f([new_x, new_y, PEDESTRIAN_Z])

        # Face the pedestrian in the direction of travel, same as dynamic_obstacles.py
        heading = math.atan2(obs["vy"], obs["vx"])
        obs["rot"].setSFRotation([0, 0, 1, heading])

        # resetPhysics() keeps the bounding object in sync with the visual position.
        # Without it, the physics engine can lag behind kinematic updates, causing
        # invisible collision geometry to drift away from where the pedestrian appears.
        obs["node"].resetPhysics()
