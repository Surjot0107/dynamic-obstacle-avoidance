from controller import Supervisor
import math

TIME_STEP = 64
robot = Supervisor()

# Get obstacle nodes by DEF name
obs1 = robot.getFromDef("OBS1")
obs2 = robot.getFromDef("OBS2")
obs3 = robot.getFromDef("OBS3")

# Movement parameters
OBS1_SPEED, OBS2_SPEED, OBS3_SPEED = 0.5, 0.6, 0.4

# Patrol bounds
OBS1_Y_MIN, OBS1_Y_MAX = -7.0, -5.0
OBS2_X_MIN, OBS2_X_MAX = -1.5,  1.5
OBS3_Y_MIN, OBS3_Y_MAX = -5.0,  1.5

# Directions
dir1, dir2, dir3 = 1, 1, -1
dt = TIME_STEP / 1000.0

while robot.step(TIME_STEP) != -1:
    # --- OBS1: Y-axis Patrol ---
    pos1 = obs1.getField("translation").getSFVec3f()
    new_y1 = pos1[1] + dir1 * OBS1_SPEED * dt
    if new_y1 >= OBS1_Y_MAX or new_y1 <= OBS1_Y_MIN:
        dir1 *= -1
    obs1.getField("translation").setSFVec3f([pos1[0], new_y1, 1.25])
    obs1.getField("rotation").setSFRotation([0, 0, 1, math.pi/2 if dir1 > 0 else -math.pi/2])
    obs1.resetPhysics() # Forces kinematic stability

    # --- OBS2: X-axis Crossing ---
    pos2 = obs2.getField("translation").getSFVec3f()
    new_x2 = pos2[0] + dir2 * OBS2_SPEED * dt
    if new_x2 >= OBS2_X_MAX or new_x2 <= OBS2_X_MIN:
        dir2 *= -1
    obs2.getField("translation").setSFVec3f([new_x2, pos2[1], 1.25])
    obs2.getField("rotation").setSFRotation([0, 0, 1, 0 if dir2 > 0 else math.pi])
    obs2.resetPhysics()

    # --- OBS3: Y-axis Head-on ---
    pos3 = obs3.getField("translation").getSFVec3f()
    new_y3 = pos3[1] + dir3 * OBS3_SPEED * dt
    if new_y3 <= OBS3_Y_MIN or new_y3 >= OBS3_Y_MAX:
        dir3 *= -1
    obs3.getField("translation").setSFVec3f([pos3[0], new_y3, 1.25])
    obs3.getField("rotation").setSFRotation([0, 0, 1, math.pi/2 if dir3 > 0 else -math.pi/2])
    obs3.resetPhysics()
