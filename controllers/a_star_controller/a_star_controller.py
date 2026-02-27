from controller import Supervisor
import numpy as np
import math
from a_star import AStarPlanner

# ================= CONSTANTS =================
TIME_STEP = 64
MAX_SPEED = 6.28

ARENA_HALF = 10.0
RES = 0.1
HEIGHT_Z = 0.0

WAYPOINT_TOL = 0.6

ROT_SPEED = 3.5
FWD_SPEED = 6.0
TURN_GAIN = 1.0

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

# ================= LOAD GRID =================
# Grid is already in [z, x] form — DO NOT transpose
grid = np.load("../../data/maps/aisle_grid.npy")

# Start and goal are (z, x)
start = (40, 40)
goal  = (120, 120)

planner = AStarPlanner(grid)
path = planner.plan(start, goal)
if path is None:
    raise RuntimeError("A* failed")

# ================= GRID → WORLD =================
def grid_to_world(path):
    pts = []
    for z, x in path:
        wx = -ARENA_HALF + x * RES
        wy = -ARENA_HALF + z * RES
        pts.append((wx, wy))
    return pts

waypoints = grid_to_world(path)
wp = 0

# ================= INITIAL POSE =================
sx, sy = waypoints[0]
translation.setSFVec3f([sx, sy, HEIGHT_Z])
rotation.setSFRotation([0, 0, 1, 0])

# ================= HELPERS =================
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def get_pose():
    p = translation.getSFVec3f()
    r = rotation.getSFRotation()
    yaw = r[3] if r[2] >= 0 else -r[3]
    return p[0], p[1], yaw

# ================= CONTROL LOOP =================
while robot.step(TIME_STEP) != -1:

    if wp >= len(waypoints):
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        print("GOAL REACHED")
        break

    x, y, yaw = get_pose()
    wx, wy = waypoints[wp]

    dx = wx - x
    dy = wy - y
    dist = math.hypot(dx, dy)

    # Robot-frame projection
    ex =  math.cos(yaw) * dx + math.sin(yaw) * dy
    ey = -math.sin(yaw) * dx + math.cos(yaw) * dy

    # ---------- CONTROL LOGIC ----------
    if ex < 0.05:
        # Waypoint not in front → rotate only
        w = ROT_SPEED if ey > 0 else -ROT_SPEED
        left_motor.setVelocity(-w)
        right_motor.setVelocity(w)
    else:
        # Waypoint in front → drive forward + correction
        v = FWD_SPEED
        w = TURN_GAIN * ey

        vl = v - w
        vr = v + w

        vl = max(-MAX_SPEED, min(MAX_SPEED, vl))
        vr = max(-MAX_SPEED, min(MAX_SPEED, vr))

        left_motor.setVelocity(vl)
        right_motor.setVelocity(vr)

    if dist < WAYPOINT_TOL:
        wp += 1
