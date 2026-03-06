from controller import Supervisor
import numpy as np
import math

from dwa_core import DWAPlanner
from debug_plot import plot_dwa_debug


TIME_STEP = 64
MAX_SPEED = 6.28

WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.16


robot = Supervisor()


left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


lidar = robot.getDevice("LDS-01")
lidar.enable(TIME_STEP)


robot_node = robot.getFromDef("TURTLEBOT3")

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")


plan = np.load("../../data/global_paths/a_star_path.npy")


RES = 0.1
ARENA_HALF = 10.0

path_world = []

for z, x in plan:

    wx = x * RES - ARENA_HALF
    wy = z * RES - ARENA_HALF

    path_world.append([wx, wy])

path_world = np.array(path_world)


# print("First path points:")
# print(path_world[:5])


# TELEPORT ROBOT TO START
start = path_world[0]
next_pt = path_world[1]

dx = next_pt[0] - start[0]
dy = next_pt[1] - start[1]

yaw = math.atan2(dy, dx)

translation_field.setSFVec3f([start[0], start[1], 0.01])
rotation_field.setSFRotation([0, 0, 1, 0])

robot.step(TIME_STEP)


path_index = 0


dwa = DWAPlanner()

current_v = 0.0
current_w = 0.0


def get_robot_pose():

    pos = robot_node.getPosition()
    rot = robot_node.getOrientation()

    x = pos[0]
    y = pos[1]

    # orientation matrix from Webots
    r00 = rot[0]
    r01 = rot[1]
    r10 = rot[3]
    r11 = rot[4]

    yaw = math.atan2(r10, r00)

    return x, y, yaw


def get_lidar_points(x, y, yaw):

    ranges = lidar.getRangeImage()

    fov = lidar.getFov()
    angle_increment = fov / (len(ranges) - 1)

    obstacles = []

    for i, r in enumerate(ranges):

        if r == float("inf") or r > 3.5:
            continue

        angle = fov / 2 - i * angle_increment

        world_angle = yaw + angle

        ox = x + r * math.cos(world_angle)
        oy = y + r * math.sin(world_angle)

        obstacles.append([ox, oy])

    if len(obstacles) == 0:
        return np.empty((0, 2))

    return np.array(obstacles)


def compute_wheel_speeds(v, w):

    left = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS

    return left, right


while robot.step(TIME_STEP) != -1:

    x, y, yaw = get_robot_pose()
    # print("POSE:", round(x,3), round(y,3), round(yaw,3))

    obstacles = get_lidar_points(x, y, yaw)
    # if len(obstacles) > 0:
    #     print("FIRST OBSTACLE:", obstacles[0])

    LOOKAHEAD = 10

    goal_index = min(path_index + LOOKAHEAD, len(path_world) - 1)

    goal = path_world[goal_index]
    print("GOAL:", goal, "INDEX:",goal_index)

    dist = math.hypot(goal[0] - x, goal[1] - y)

    if dist < 0.35 and path_index < len(path_world) - 1:
        path_index += 1

    state = [x, y, yaw, current_v, current_w]

    v_cmd, w_cmd, trajectories = dwa.plan(state, goal, obstacles)
    # print("CMD:", round(v_cmd,2), round(w_cmd,2))

    plot_dwa_debug([x, y], goal, obstacles, trajectories)

    current_v = v_cmd
    current_w = w_cmd

    left_speed, right_speed = compute_wheel_speeds(v_cmd, w_cmd)

    left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)