from controller import Supervisor
import numpy as np
import math
import json
import time
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dwa_core import DWAPlanner
from debug_plot import plot_dwa_debug

DEBUG_GRAPH = False

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

# ── Logging ───────────────────────────────────────────────────────────────────
RUN_NOTES = ""  # set before each run e.g. "3 static boxes, run 1"
BASE_LOG_DIR = "../../results/dwa"
os.makedirs(BASE_LOG_DIR, exist_ok=True)

# Auto-increment run number
existing = [int(d[-3:]) for d in os.listdir(BASE_LOG_DIR) if d.startswith("run_")]
if existing != []:
    run_number = int(max(existing)) + 1
else:
    run_number = 1
LOG_DIR = os.path.join(BASE_LOG_DIR, f"run_{run_number:03d}")
os.makedirs(LOG_DIR, exist_ok=True)

run_start_time = robot.getTime()
path_taken = []
distance_travelled = 0.0
prev_x, prev_y = None, None
goal_reached = False


def save_log(success):
    elapsed = round(robot.getTime() - run_start_time, 2)
    log = {
        "run_number": run_number,
        "success": success,
        "time_to_goal_s": elapsed if success else None,
        "total_distance_m": round(distance_travelled, 3),
        "collision_count": collision_count,
        "notes": RUN_NOTES,
        "path_taken": [[round(p[0], 3), round(p[1], 3)] for p in path_taken],
    }
    json_path = os.path.join(LOG_DIR, "log.json")
    with open(json_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved: {json_path}")
    fig = Figure(figsize=(8, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(path_world[:, 0], path_world[:, 1], "b--", linewidth=1, label="A* path")
    if len(path_taken) > 1:
        taken = np.array(path_taken)
        ax.plot(taken[:, 0], taken[:, 1], "r-", linewidth=1.5, label="Travelled path")
    ax.plot(path_world[0, 0], path_world[0, 1], "go", markersize=8, label="Start")
    ax.plot(path_world[-1, 0], path_world[-1, 1], "r*", markersize=12, label="Goal")
    ax.set_title(f"DWA Run {run_number:03d} | {'SUCCESS' if success else 'FAILED'} | "
                 f"Collisions: {collision_count} | Dist: {round(distance_travelled,2)}m")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True)
    plot_path = os.path.join(LOG_DIR, "path_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")


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


try:
    while robot.step(TIME_STEP) != -1:

        x, y, yaw = get_robot_pose()
        path_taken.append([x, y])
        if prev_x is not None:
            distance_travelled += math.hypot(x - prev_x, y - prev_y)
        prev_x, prev_y = x, y

        obstacles = get_lidar_points(x, y, yaw)

        goal, goal_index = get_carrot(x, y)
        print("CLOSEST:", closest_index, "  CARROT:", goal_index,
              " GOAL:", np.round(goal, 2), " COLLISIONS:", collision_count)

        if goal_index >= len(path_world) - 1:
            if math.hypot(path_world[-1][0] - x, path_world[-1][1] - y) < 0.3:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                print("GOAL REACHED")
                goal_reached = True
                save_log(success=True)
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

        if DEBUG_GRAPH == True:
            plot_dwa_debug([x, y], goal, obstacles, trajectories)

        current_v = v_cmd
        current_w = w_cmd

        left_speed, right_speed = compute_wheel_speeds(v_cmd, w_cmd)
        left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

finally:
    if not goal_reached:
        save_log(success=False)