from controller import Supervisor
import numpy as np
import math
import os

# ================= PATH HANDLING =================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
GLOBAL_PATH_FILE = os.path.join(BASE_DIR, "data", "global_paths", "a_star_path.npy")
RESULT_PATH_FILE = os.path.join(BASE_DIR, "data", "results", "astar_tracking.npy")

# ================= CONSTANTS =================

TIME_STEP = 64
MAX_SPEED = 6.28

ARENA_HALF = 10.0
RES = 0.1
HEIGHT_Z = 0.0

WAYPOINT_TOL = 0.25

# Faster parameters
FWD_SPEED = 6.0
TURN_GAIN = 3.0

# ================= LOAD GLOBAL PATH =================

if not os.path.exists(GLOBAL_PATH_FILE):
    raise RuntimeError("Run generate_path.py first.")

path = np.load(GLOBAL_PATH_FILE, allow_pickle=True)

# ================= PATH SPARSIFICATION =================

def sparsify_path(path, step=5):
    return path[::step]

path = sparsify_path(path, step=5)

# ================= GRID -> WORLD =================

def grid_to_world(path):
    pts = []
    for z, x in path:
        wx = -ARENA_HALF + x * RES
        wy = -ARENA_HALF + z * RES
        pts.append((wx, wy))
    return pts

waypoints = grid_to_world(path)
wp = 0

# ================= SUPERVISOR =================

robot = Supervisor()

node = robot.getFromDef("TURTLEBOT3")
if node is None:
    raise RuntimeError("DEF TURTLEBOT3 not found")

translation = node.getField("translation")
rotation = node.getField("rotation")

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0)
right_motor.setVelocity(0)

# ================= INITIAL POSE =================

sx, sy = waypoints[0]
translation.setSFVec3f([sx, sy, HEIGHT_Z])
rotation.setSFRotation([0, 0, 1, 0])

# ================= HELPERS =================

def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def get_pose():
    p = translation.getSFVec3f()
    r = rotation.getSFRotation()
    yaw = r[3] if r[2] >= 0 else -r[3]
    return p[0], p[1], yaw

# ================= TRACKING LOG =================

log = []

# ================= CONTROL LOOP =================

while robot.step(TIME_STEP) != -1:

    if wp >= len(waypoints):
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)

        os.makedirs(os.path.dirname(RESULT_PATH_FILE), exist_ok=True)
        np.save(RESULT_PATH_FILE, np.array(log))

        print("GOAL REACHED")
        break

    x, y, yaw = get_pose()
    log.append([x, y])

    wx, wy = waypoints[wp]

    dx = wx - x
    dy = wy - y
    dist = math.hypot(dx, dy)

    angle_to_wp = math.atan2(dy, dx)
    heading_error = wrap(angle_to_wp - yaw)

    # Continuous motion control
    v = FWD_SPEED
    w = TURN_GAIN * heading_error

    # Slow down if turning very sharply
    if abs(heading_error) > 1.0:
        v = 0.5 * FWD_SPEED

    vl = v - w
    vr = v + w

    vl = max(-MAX_SPEED, min(MAX_SPEED, vl))
    vr = max(-MAX_SPEED, min(MAX_SPEED, vr))

    left_motor.setVelocity(vl)
    right_motor.setVelocity(vr)

    # Move to next waypoint
    if dist < WAYPOINT_TOL:
        wp += 1