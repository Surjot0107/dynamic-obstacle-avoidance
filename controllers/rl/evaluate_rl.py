"""
Evaluate a trained RL agent by following a global A* path.

Must match rl_env.py exactly:
  - 24 LiDAR sectors
  - Continuous action space → [v, w]
  - Same normalisation constants
"""

from controller import Supervisor
import numpy as np
import math
import json
import os
import glob
import argparse
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from stable_baselines3 import PPO

# ── Configuration ─────────────────────────────────────────────────────────
PATH_FILE      = "../../data/global_paths/a_star_path.npy"
BASE_MODEL_DIR = "models"
OUTPUT_DIR     = "../../results/rl"
LOOKAHEAD_DIST = 2.0  # Shorter lookahead for smoother carrot transitions around corners

parser = argparse.ArgumentParser()
parser.add_argument("--path",      default=PATH_FILE)
parser.add_argument("--model",     default=None)
parser.add_argument("--model-dir", default=BASE_MODEL_DIR)
parser.add_argument("--output",    default=OUTPUT_DIR)
parser.add_argument("--lookahead", type=float, default=LOOKAHEAD_DIST)
args, _ = parser.parse_known_args()

PATH_FILE      = args.path
MODEL_EXPLICIT = args.model
BASE_MODEL_DIR = args.model_dir
OUTPUT_DIR     = args.output
LOOKAHEAD_DIST = args.lookahead

# ── Constants (MUST match rl_env.py) ──────────────────────────────────────
TIME_STEP       = 64
MAX_SPEED       = 6.28
WHEEL_RADIUS    = 0.033
WHEEL_BASE      = 0.16
MAX_LIDAR_RANGE = 3.5
N_LIDAR_SECTORS = 24
FRAME_SKIP      = 3

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

# ── Load global path ──────────────────────────────────────────────────────
plan = np.load(PATH_FILE)
RES = 0.1
ARENA_HALF = 10.0

path_world = np.array([
    [x * RES - ARENA_HALF, z * RES - ARENA_HALF]
    for z, x in plan
])

seg_lengths = np.hypot(np.diff(path_world[:, 0]), np.diff(path_world[:, 1]))
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
    run_dirs = sorted([
        d for d in glob.glob(os.path.join(base_dir, "run_*"))
        if os.path.isdir(d)
    ])
    if not run_dirs:
        raise FileNotFoundError(f"No run_XXX folders in {base_dir!r}")

    for run_dir in reversed(run_dirs):
        preferred = os.path.join(run_dir, "ppo_final.zip")
        if os.path.isfile(preferred):
            return preferred.replace(".zip", "")
        fallback = sorted(glob.glob(os.path.join(run_dir, "*.zip")),
                          key=os.path.getmtime)
        if fallback:
            return fallback[-1].replace(".zip", "")
    raise FileNotFoundError(f"No model .zip files in {base_dir!r}")


model_path = MODEL_EXPLICIT if MODEL_EXPLICIT else find_latest_model(BASE_MODEL_DIR)
print(f"Loading model: {model_path}")
model = PPO.load(model_path)

# ── State ─────────────────────────────────────────────────────────────────
closest_index   = 0
collision_count  = 0
in_collision     = False
run_start_time   = robot.getTime()
path_taken       = []
distance_travelled = 0.0
prev_x, prev_y  = None, None
goal_reached     = False

BASE_LOG_DIR = OUTPUT_DIR
os.makedirs(BASE_LOG_DIR, exist_ok=True)
existing = [int(d[-3:]) for d in os.listdir(BASE_LOG_DIR) if d.startswith("run_")]
run_number = (max(existing) + 1) if existing else 1
LOG_DIR = os.path.join(BASE_LOG_DIR, f"run_{run_number:03d}")
os.makedirs(LOG_DIR, exist_ok=True)


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
    """Build the same 26-dim vector the agent was trained on."""
    ranges = lidar.getRangeImage()
    n_rays = len(ranges)
    rpt = max(1, n_rays // N_LIDAR_SECTORS)
    sectors = np.ones(N_LIDAR_SECTORS, dtype=np.float32)
    for s in range(N_LIDAR_SECTORS):
        start = s * rpt
        for r in ranges[start:start + rpt]:
            d = 1.0 if (r == float("inf") or r > MAX_LIDAR_RANGE) else r / MAX_LIDAR_RANGE
            if d < sectors[s]:
                sectors[s] = d

    dx = carrot[0] - x
    dy = carrot[1] - y
    dist = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) - yaw
    angle = (angle + math.pi) % (2 * math.pi) - math.pi

    obs = np.concatenate([
        sectors,
        [min(dist / 10.0, 1.0)],
        [(angle / math.pi + 1.0) / 2.0],
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
        "notes":            "",
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
left_speed = 0.0
right_speed = 0.0

# ── Action smoothing (EMA) ────────────────────────────────────────────────
# Smooths rapid left↔right oscillations from the policy.
# alpha=0.5 → 50% new action, 50% previous.
EMA_ALPHA = 0.5
smooth_v = 0.0
smooth_w = 0.0

# ── Stagnation recovery ──────────────────────────────────────────────────
# If the robot doesn't move for ~5 seconds, the RL policy is stuck.
# Switch to a simple proportional controller that points at the carrot
# and drives forward — exactly what the user does manually.
STAG_CHECK_INTERVAL = 40     # steps (~5.1 sec at 64ms)
STAG_DIST_THRESHOLD = 0.3    # must move at least this far
RECOVERY_STEPS      = 20     # drive with P-controller for ~4 sec
stag_x, stag_y = None, None
stag_step = 0
recovery_mode = False
recovery_countdown = 0

try:
    while robot.step(TIME_STEP) != -1:

        x, y, yaw = get_robot_pose()
        path_taken.append([x, y])
        if prev_x is not None:
            distance_travelled += math.hypot(x - prev_x, y - prev_y)
        prev_x, prev_y = x, y

        goal, goal_index = get_carrot(x, y)
        mode_str = " RECOVERY" if recovery_mode else ""
        print(
            "CLOSEST:", closest_index, "  CARROT:", goal_index,
            " GOAL:", np.round(goal, 2), " COLLISIONS:", collision_count,
            mode_str,
        )

        # Goal check
        if goal_index >= len(path_world) - 1:
            if math.hypot(path_world[-1][0] - x, path_world[-1][1] - y) < 0.3:
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                print("GOAL REACHED")
                goal_reached = True
                save_log(success=True)
                break

        # Collision counting
        contact_points = robot_node.getContactPoints(includeDescendants=True)
        currently_touching = any(cp.point[2] > 0.01 for cp in contact_points)
        if currently_touching and not in_collision:
            collision_count += 1
        in_collision = currently_touching

        # Stagnation detection
        if stag_x is None:
            stag_x, stag_y = x, y
        stag_step += 1
        if stag_step >= STAG_CHECK_INTERVAL:
            moved = math.hypot(x - stag_x, y - stag_y)
            if moved < STAG_DIST_THRESHOLD and not recovery_mode:
                recovery_mode = True
                recovery_countdown = RECOVERY_STEPS
                smooth_v = 0.0
                smooth_w = 0.0
                print(">>> STAGNATION DETECTED — switching to P-controller recovery")
            elif not recovery_mode:
                pass  # all good
            stag_x, stag_y = x, y
            stag_step = 0

        # Recovery countdown
        if recovery_mode:
            recovery_countdown -= 1
            if recovery_countdown <= 0:
                recovery_mode = False
                print(">>> Recovery complete — back to RL policy")

        # RL decision every FRAME_SKIP steps
        if step_count % FRAME_SKIP == 0:

            if recovery_mode:
                # ── P-controller fallback: point at carrot and drive ──
                dx = goal[0] - x
                dy = goal[1] - y
                target_yaw = math.atan2(dy, dx)
                angle_err = target_yaw - yaw
                angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi

                use_w = float(np.clip(2.0 * angle_err, -1.5, 1.5))  # P-gain = 2.0
                # Drive forward proportional to alignment (slow if misaligned)
                use_v = 0.2 * max(0.0, 1.0 - abs(angle_err) / (math.pi / 2))
            else:
                # ── Normal RL policy + EMA smoothing ──
                obs = build_observation(x, y, yaw, goal)
                action, _ = model.predict(obs, deterministic=True)

                v_raw = float((np.clip(action[0], -1, 1) + 1.0) / 2.0 * 0.4)
                w_raw = float(np.clip(action[1], -1, 1) * 1.5)

                smooth_v = EMA_ALPHA * v_raw + (1 - EMA_ALPHA) * smooth_v
                smooth_w = EMA_ALPHA * w_raw + (1 - EMA_ALPHA) * smooth_w
                use_v, use_w = smooth_v, smooth_w

            left_speed, right_speed = compute_wheel_speeds(use_v, use_w)
            left_speed  = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
            right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        step_count += 1

finally:
    if not goal_reached:
        save_log(success=False)
