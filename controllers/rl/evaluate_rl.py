"""
Evaluate a trained RL agent by following a global path.

Mirrors the execution loop and logging from controllers/dwa/dwa.py:
same get_carrot() logic, same collision counting, same JSON + plot output.

Configuration: edit the paths below, or pass them as controller arguments.
"""

from controller import Supervisor
import numpy as np
import math
import json
import os
import glob
import sys
import argparse
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from stable_baselines3 import PPO

# ── Configuration ─────────────────────────────────────────────────────────
# Edit these directly, or override via Webots controller arguments.
PATH_FILE      = "../../data/global_paths/a_star_path.npy"
BASE_MODEL_DIR = "models"          # parent of all run_XXX folders
OUTPUT_DIR     = "../../results/rl"
LOOKAHEAD_DIST = 1.5

parser = argparse.ArgumentParser(description="Evaluate RL agent on a global path")
parser.add_argument("--path",      default=PATH_FILE)
parser.add_argument("--model",     default=None,
                    help="Explicit model file (without .zip). "
                         "Defaults to ppo_final from the latest run in BASE_MODEL_DIR.")
parser.add_argument("--model-dir", default=BASE_MODEL_DIR,
                    help="Parent directory containing run_XXX folders.")
parser.add_argument("--output",    default=OUTPUT_DIR)
parser.add_argument("--lookahead", type=float, default=LOOKAHEAD_DIST)
args, _ = parser.parse_known_args()  # parse_known_args tolerates Webots injected args

PATH_FILE      = args.path
MODEL_EXPLICIT = args.model
BASE_MODEL_DIR = args.model_dir
OUTPUT_DIR     = args.output
LOOKAHEAD_DIST = args.lookahead

# ── Constants ─────────────────────────────────────────────────────────────
TIME_STEP     = 64
MAX_SPEED     = 6.28
WHEEL_RADIUS  = 0.033
WHEEL_BASE    = 0.16
MAX_LIDAR_RANGE = 3.0
NUM_LIDAR_BUCKETS = 72
FRAME_SKIP    = 3

ACTION_TABLE = [
    (0.8,  0.0),    # 0: Fast Forward
    (0.4,  0.0),    # 1: Slow Forward
    (0.4,  0.75),   # 2: Soft Left
    (0.4, -0.75),   # 3: Soft Right
    (0.2,  1.5),    # 4: Tight Left  (dodge maneuver)
    (0.2, -1.5),    # 5: Tight Right (dodge maneuver)
    (0.0,  1.5),    # 6: Hard Left   (pivot)
    (0.0, -1.5),    # 7: Hard Right  (pivot)
    (-0.2, 0.0),    # 8: Reverse
]

# ── Robot setup ───────────────────────────────────────────────────────────
robot = Supervisor()

camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

left_motor  = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

lidar = robot.getDevice("LDS-01")
lidar.enable(TIME_STEP)

robot_node = robot.getFromDef("TURTLEBOT3")
translation_field = robot_node.getField("translation")
rotation_field    = robot_node.getField("rotation")

# ── Load and convert global path (identical to dwa.py) ────────────────────
plan = np.load(PATH_FILE)

RES = 0.1
ARENA_HALF = 10.0

path_world = np.array([
    [x * RES - ARENA_HALF, z * RES - ARENA_HALF]
    for z, x in plan
])

seg_lengths = np.hypot(
    np.diff(path_world[:, 0]),
    np.diff(path_world[:, 1]),
)
arc_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])

# Teleport to start
start   = path_world[0]
next_pt = path_world[1]
yaw     = math.atan2(next_pt[1] - start[1], next_pt[0] - start[0])

translation_field.setSFVec3f([start[0], start[1], 0.01])
rotation_field.setSFRotation([0, 0, 1, yaw])
robot_node.resetPhysics()
robot.step(TIME_STEP)

# ── Load model ────────────────────────────────────────────────────────────

def find_latest_model(base_dir):
    """
    Walk models/run_XXX/ subfolders, find the latest run, and return the
    path to its ppo_final model. Falls back to any .zip if ppo_final is
    missing (e.g. training was interrupted before completion).
    """
    run_dirs = sorted(
        [d for d in glob.glob(os.path.join(base_dir, "run_*"))
         if os.path.isdir(d)]
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No run_XXX folders found in {base_dir!r}. "
            "Have you trained a model yet?"
        )

    # Walk from the most recent run backwards until we find a usable model
    for run_dir in reversed(run_dirs):
        preferred = os.path.join(run_dir, "ppo_final.zip")
        if os.path.isfile(preferred):
            return preferred.replace(".zip", "")
        # If the run crashed before saving ppo_final, take whatever .zip is newest
        fallback = sorted(glob.glob(os.path.join(run_dir, "*.zip")),
                          key=os.path.getmtime)
        if fallback:
            return fallback[-1].replace(".zip", "")

    raise FileNotFoundError(f"No model .zip files found under {base_dir!r}")


if MODEL_EXPLICIT:
    model_path = MODEL_EXPLICIT
else:
    model_path = find_latest_model(BASE_MODEL_DIR)

print(f"Loading model: {model_path}")
model = PPO.load(model_path)

# ── State ─────────────────────────────────────────────────────────────────
closest_index  = 0
current_v      = 0.0
current_w      = 0.0
collision_count = 0
in_collision   = False

RUN_NOTES    = ""
BASE_LOG_DIR = OUTPUT_DIR
os.makedirs(BASE_LOG_DIR, exist_ok=True)

existing = [int(d[-3:]) for d in os.listdir(BASE_LOG_DIR) if d.startswith("run_")]
run_number = (max(existing) + 1) if existing else 1
LOG_DIR = os.path.join(BASE_LOG_DIR, f"run_{run_number:03d}")
os.makedirs(LOG_DIR, exist_ok=True)

run_start_time   = robot.getTime()
path_taken       = []
distance_travelled = 0.0
prev_x, prev_y  = None, None
goal_reached     = False


# ── Helpers ───────────────────────────────────────────────────────────────

def compute_wheel_speeds(v, w):
    left  = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return left, right


def get_robot_pose():
    pos = robot_node.getPosition()
    rot = robot_node.getOrientation()
    return pos[0], pos[1], math.atan2(rot[3], rot[0])


def get_carrot(x, y):
    global closest_index
    end = min(closest_index + 150, len(path_world))
    dists = np.hypot(
        path_world[closest_index:end, 0] - x,
        path_world[closest_index:end, 1] - y,
    )
    closest_index = closest_index + int(np.argmin(dists))

    target_arc   = arc_length[closest_index] + LOOKAHEAD_DIST
    carrot_index = closest_index
    while carrot_index < len(path_world) - 1 and arc_length[carrot_index] < target_arc:
        carrot_index += 1

    return path_world[carrot_index], carrot_index


def build_observation(x, y, yaw, carrot):
    """Build the same 76-dim vector the agent was trained on."""
    ranges = lidar.getRangeImage()
    n_rays = len(ranges)
    rays_per_bucket = max(1, n_rays // NUM_LIDAR_BUCKETS)
    buckets = np.ones(NUM_LIDAR_BUCKETS, dtype=np.float32)

    for b in range(NUM_LIDAR_BUCKETS):
        start = b * rays_per_bucket
        end   = start + rays_per_bucket
        for r in ranges[start:end]:
            if r == float("inf") or r > MAX_LIDAR_RANGE:
                d = 1.0
            else:
                d = r / MAX_LIDAR_RANGE
            if d < buckets[b]:
                buckets[b] = d

    dx = carrot[0] - x
    dy = carrot[1] - y
    dist_to_carrot  = math.hypot(dx, dy)
    angle_to_carrot = math.atan2(dy, dx) - yaw
    angle_to_carrot = (angle_to_carrot + math.pi) % (2 * math.pi) - math.pi

    obs = np.concatenate([
        buckets,
        [min(dist_to_carrot / 7.0, 1.0)],
        [angle_to_carrot / math.pi],
        [np.clip(current_v / 0.8, -1.0, 1.0)],
        [np.clip(current_w / 1.5, -1.0, 1.0)],
    ]).astype(np.float32)

    return obs


def save_log(success):
    elapsed = round(robot.getTime() - run_start_time, 2)
    log = {
        "run_number":       run_number,
        "model_used":       model_path,
        "success":          success,
        "time_to_goal_s":   elapsed if success else None,
        "total_distance_m": round(distance_travelled, 3),
        "collision_count":  collision_count,
        "notes":            RUN_NOTES,
        "path_taken":       [[round(p[0], 3), round(p[1], 3)] for p in path_taken],
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
    ax.plot(path_world[0, 0],  path_world[0, 1],  "go", markersize=8,  label="Start")
    ax.plot(path_world[-1, 0], path_world[-1, 1], "r*", markersize=12, label="Goal")

    ax.set_title(
        f"RL Run {run_number:03d} | {'SUCCESS' if success else 'FAILED'} | "
        f"Collisions: {collision_count} | Dist: {round(distance_travelled, 2)}m"
    )
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True)

    plot_path = os.path.join(LOG_DIR, "path_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")


# ── Main loop ─────────────────────────────────────────────────────────────

step_count = 0
try:
    while robot.step(TIME_STEP) != -1:

        x, y, yaw = get_robot_pose()
        path_taken.append([x, y])
        if prev_x is not None:
            distance_travelled += math.hypot(x - prev_x, y - prev_y)
        prev_x, prev_y = x, y

        goal, goal_index = get_carrot(x, y)
        print(
            "CLOSEST:", closest_index, "  CARROT:", goal_index,
            " GOAL:", np.round(goal, 2), " COLLISIONS:", collision_count,
        )

        # Check if we've reached the end of the path
        if goal_index >= len(path_world) - 1:
            if math.hypot(path_world[-1][0] - x, path_world[-1][1] - y) < 0.3:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                print("GOAL REACHED")
                goal_reached = True
                save_log(success=True)
                break

        # Collision counting — distinct events, not individual frames
        contact_points = robot_node.getContactPoints(includeDescendants=True)
        currently_touching = any(cp.point[2] > 0.01 for cp in contact_points)
        if currently_touching and not in_collision:
            collision_count += 1
        in_collision = currently_touching

        # RL decision — only update every FRAME_SKIP steps to match training dynamics
        if step_count % FRAME_SKIP == 0:
            obs = build_observation(x, y, yaw, goal)
            action, _ = model.predict(obs, deterministic=True)
            v_cmd, w_cmd = ACTION_TABLE[int(action)]

            current_v = v_cmd
            current_w = w_cmd

            left_speed, right_speed = compute_wheel_speeds(v_cmd, w_cmd)
            left_speed  = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
            right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        step_count += 1

finally:
    if not goal_reached:
        save_log(success=False)
