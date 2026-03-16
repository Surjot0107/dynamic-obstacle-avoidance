from controller import Supervisor
import numpy as np
import math

from dwa_core import DWAPlanner
from debug_plot import plot_dwa_debug


TIME_STEP = 64
MAX_SPEED = 6.28

WHEEL_RADIUS = 0.033
WHEEL_BASE   = 0.16
LOOKAHEAD_DIST = 1.5  # metres ahead on path to place the carrot


robot = Supervisor()
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

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


# ── Load and convert global path ──────────────────────────────────────────────
plan = np.load("../../data/global_paths/a_star_path.npy")

RES = 0.1
ARENA_HALF = 10.0

path_world = np.array([
    [x * RES - ARENA_HALF, z * RES - ARENA_HALF]
    for z, x in plan
])

# Precompute cumulative arc-length along path
seg_lengths = np.hypot(
    np.diff(path_world[:, 0]),
    np.diff(path_world[:, 1])
)
arc_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])


# TELEPORT ROBOT TO START
start = path_world[0]
next_pt = path_world[1]

yaw = math.atan2(next_pt[1] - start[1], next_pt[0] - start[0])

translation_field.setSFVec3f([start[0], start[1], 0.01])
rotation_field.setSFRotation([0, 0, 1, 0])
robot.step(TIME_STEP)


# ── State ─────────────────────────────────────────────────────────────────────
closest_index = 0
dwa = DWAPlanner()
current_v = 0.0
current_w = 0.0
collision_count = 0
in_collision = False  # tracks if we are currently in a collision event


def get_robot_pose():
    pos = robot_node.getPosition()
    rot = robot_node.getOrientation()
    return pos[0], pos[1], math.atan2(rot[3], rot[0])


def get_lidar_points(x, y, yaw):
    ranges = lidar.getRangeImage()
    fov = lidar.getFov()
    angle_increment = fov / (len(ranges) - 1)
    obstacles = []
    for i, r in enumerate(ranges):
        if r == float("inf") or r > 3.0:
            continue
        angle = fov / 2 - i * angle_increment
        world_angle = yaw + angle
        obstacles.append([x + r * math.cos(world_angle), y + r * math.sin(world_angle)])
    return np.array(obstacles) if obstacles else np.empty((0, 2))


def compute_wheel_speeds(v, w):
    left = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return left, right


def get_carrot(x, y):
    global closest_index
    end = min(closest_index + 150, len(path_world))
    dists = np.hypot(path_world[closest_index:end, 0] - x,
                     path_world[closest_index:end, 1] - y)
    closest_index = closest_index + int(np.argmin(dists))

    target_arc   = arc_length[closest_index] + LOOKAHEAD_DIST
    carrot_index = closest_index
    while carrot_index < len(path_world) - 1 and arc_length[carrot_index] < target_arc:
        carrot_index += 1

    return path_world[carrot_index], carrot_index


while robot.step(TIME_STEP) != -1:

    x, y, yaw = get_robot_pose()

    obstacles = get_lidar_points(x, y, yaw)

    goal, goal_index = get_carrot(x, y)
    print("CLOSEST:", closest_index, "  CARROT:", goal_index,
          " GOAL:", np.round(goal, 2), " COLLISIONS:", collision_count)

    if goal_index >= len(path_world) - 1:
        if math.hypot(path_world[-1][0] - x, path_world[-1][1] - y) < 0.3:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            print("GOAL REACHED")
            break

    # Collision detection — count distinct events, not frames.
    # A new collision is only counted when contact starts (not while dragging).
    contact_points = robot_node.getContactPoints(includeDescendants=True)
    currently_touching = any(cp.point[2] > 0.01 for cp in contact_points)
    if currently_touching and not in_collision:
        collision_count += 1  # new collision event started
    in_collision = currently_touching
    state = [x, y, yaw, current_v, current_w]
    v_cmd, w_cmd, trajectories = dwa.plan(state, goal, obstacles)

    plot_dwa_debug([x, y], goal, obstacles, trajectories)

    current_v = v_cmd
    current_w = w_cmd

    left_speed, right_speed = compute_wheel_speeds(v_cmd, w_cmd)
    left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)