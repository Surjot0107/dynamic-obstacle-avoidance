"""
Gymnasium environment for RL training in a corridor-based Webots world.

Design based on Tai et al. (2017) "Virtual-to-real Deep RL: Continuous
Control of Mobile Robots for Mapless Navigation" and standard SB3 practices.

KEY DESIGN PRINCIPLES:
  1. Train in the SAME corridor geometry used for evaluation.
  2. Use continuous actions (v, w) so the agent can fine-tune its steering.
  3. CARROT ADVANCES during the episode — matches evaluation exactly.
     In eval, the carrot slides forward along the A* path every step.
     In training, when the robot reaches a carrot, a new one appears ahead.
     Without this, the agent learns to aim at a static point and stop — useless.
  4. Episode NEVER terminates on carrot reach (only on collision or timeout).
     This matches evaluation where no single mid-path carrot ends the run.
  5. Carrot placed at ±60° offset — agent trains on full angular range seen in eval.
  6. Spawns cover both corridor (59%) AND open area (41%) of the A* path.

Observation: 26-dim
  - 24 LiDAR sectors (360°/15° each, normalised [0,1])
  - carrot distance (normalised over 10m)
  - carrot relative angle (normalised [0,1], 0.5=dead ahead)

Action: Box(2) continuous
  - action[0] → linear velocity  in [0.0, 0.4] m/s
  - action[1] → angular velocity in [-1.5, 1.5] rad/s
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from controller import Supervisor

# ── Simulation ────────────────────────────────────────────────────────────────
TIME_STEP       = 64             # ms — MUST match evaluate_rl.py
MAX_SPEED       = 6.28           # motor limit (rad/s)
WHEEL_RADIUS    = 0.033          # m   — MUST match evaluate_rl.py
WHEEL_BASE      = 0.16           # m   — MUST match evaluate_rl.py
MAX_LIDAR_RANGE = 3.5            # m   — MUST match evaluate_rl.py
N_LIDAR_SECTORS = 24             # — MUST match evaluate_rl.py
FRAME_SKIP      = 3              # hold action for 192ms — MUST match evaluate_rl.py
MAX_STEPS       = 300            # episode time limit

# ── Velocity limits ───────────────────────────────────────────────────────────
V_MAX = 0.4                      # m/s — MUST match evaluate_rl.py action mapping
W_MAX = 1.5                      # rad/s — MUST match evaluate_rl.py action mapping

# ── Carrot / goal ─────────────────────────────────────────────────────────────
CARROT_REACH_RADIUS = 0.8        # m — when robot gets this close, advance the carrot
                                  # Larger than eval GOAL_RADIUS because carrot keeps moving
CARROT_DIST_MIN     = 1.5        # m — eval lookahead is 2m, so train must include that
CARROT_DIST_MAX     = 4.0        # m — still some longer ones for generalization
CARROT_ANGLE_MAX    = math.pi / 2  # ±90° — covers full forward hemisphere
                                    # The A* path has a 90° turn at corridor entrance

# ── Reward tuning ─────────────────────────────────────────────────────────────
R_CARROT_REACH  = 20.0           # small reward for reaching each chained carrot
R_COLLISION     = -100.0
R_PROGRESS      = 30.0           # per metre of progress toward current carrot
R_TIME          = -0.3           # per step — 300 steps × -0.3 = -90
R_SPIN_PENALTY  = -0.5           # spinning in place (v≈0, high |w|)
R_OSCILLATION   = -0.3           # rapid left↔right direction flip

# ── Spawn positions ───────────────────────────────────────────────────────────
# Covers both corridor (x∈[-1,1]) and open-area (41% of A* path, x∈[-6,-1]).
SPAWNS = [
    # ── Inside corridor ───────────────────────────────────────────────────
    (0.0, -7.0, -math.pi/2),
    (0.0, -5.0, -math.pi/2),
    (0.0, -3.0, -math.pi/2),
    (0.0, -1.0, -math.pi/2),
    (0.0,  1.0,  math.pi/2),
    (0.0,  3.0,  math.pi/2),
    (0.0,  5.0,  math.pi/2),
    (0.0,  7.0,  math.pi/2),
    # Reverse direction
    (0.0, -5.0,  math.pi/2),
    (0.0, -3.0,  math.pi/2),
    (0.0, -1.0,  math.pi/2),
    (0.0,  1.0, -math.pi/2),
    (0.0,  3.0, -math.pi/2),
    (0.0,  5.0, -math.pi/2),
    # Slightly off-center
    (-0.5, -6.0, -math.pi/2),
    ( 0.5, -4.0, -math.pi/2),
    (-0.3, -2.0, -math.pi/2),
    ( 0.3,  2.0,  math.pi/2),
    (-0.5,  4.0,  math.pi/2),
    ( 0.5,  6.0,  math.pi/2),
    # ── Open area (matches 41% of A* path outside the corridor) ───────────
    (-6.0, -6.0,  math.pi/4),    # A* start heading NE
    (-5.0, -5.9,  0.0),
    (-4.0, -5.9,  0.0),
    (-3.0, -5.9,  0.0),
    (-2.0, -5.6,  math.pi/4),
    (-1.5, -5.6,  0.0),
    ( 1.0,  1.5,  math.pi/4),    # exiting corridor
    ( 1.5,  1.7,  math.pi/4),
    (-5.0, -5.5,  math.pi/2),
    (-4.0, -5.5, -math.pi/4),
    (-3.0, -5.0,  math.pi/2),
]


def compute_wheel_speeds(v, w):
    """MUST match evaluate_rl.py identically."""
    left  = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return left, right


def _new_carrot(rx, ry, base_yaw, prev_carrot=None):
    """
    Place a new carrot 3-5m ahead.
    - base_yaw: current robot yaw (used as direction baseline).
    - prev_carrot: if given AND far enough away, bias toward that direction.
      This provides continuity when chaining carrots during an episode.
      If prev_carrot is very close (robot just reached it), atan2 is noisy,
      so we fall back to base_yaw instead.
    """
    if prev_carrot is not None:
        pdx = prev_carrot[0] - rx
        pdy = prev_carrot[1] - ry
        if math.hypot(pdx, pdy) > 0.3:          # only use if direction is reliable
            base_yaw = math.atan2(pdy, pdx)     # continue roughly toward prev carrot
        # else: keep base_yaw (current robot heading) as direction

    dist = float(np.random.uniform(CARROT_DIST_MIN, CARROT_DIST_MAX))
    offset = float(np.random.uniform(-CARROT_ANGLE_MAX, CARROT_ANGLE_MAX))
    angle = base_yaw + offset
    cx = float(np.clip(rx + dist * math.cos(angle), -9.0, 9.0))
    cy = float(np.clip(ry + dist * math.sin(angle), -9.0, 9.0))
    return np.array([cx, cy])


class TurtleBotEnv(gym.Env):
    """
    Obs: Box(26,) — 24 LiDAR sectors + carrot_dist + carrot_angle
    Act: Box(2,)  — [v_norm, w_norm] in [-1, 1] → mapped to [0,0.4] and [-1.5,1.5]
    """
    metadata = {"render_modes": []}

    def __init__(self, curriculum_phase=1):
        super().__init__()
        self.curriculum_phase = curriculum_phase

        # ── Webots ────────────────────────────────────────────────────────
        self.robot = Supervisor()

        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(TIME_STEP)

        self.left_motor  = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.robot_node  = self.robot.getFromDef("TURTLEBOT3")
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field   = self.robot_node.getField("rotation")

        # CardboardBox obstacle references (CB1-CB6)
        self.cardboard_obs = []
        for i in range(1, 7):
            node = self.robot.getFromDef(f"CB{i}")
            if node:
                self.cardboard_obs.append({
                    "node":  node,
                    "trans": node.getField("translation"),
                })

        # Dynamic pedestrian references (OBS1-OBS5)
        self.dynamic_obs = []
        for i in range(1, 6):
            node = self.robot.getFromDef(f"OBS{i}")
            if node:
                self.dynamic_obs.append({
                    "node":  node,
                    "trans": node.getField("translation"),
                })

        # ── Gymnasium spaces ──────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=np.zeros(N_LIDAR_SECTORS + 2, dtype=np.float32),
            high=np.ones(N_LIDAR_SECTORS + 2, dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
        )

        # ── State ─────────────────────────────────────────────────────────
        self.carrot    = np.array([0.0, 0.0])
        self.prev_dist = 0.0
        self.step_count = 0
        self.prev_w     = 0.0

        self.robot.step(TIME_STEP)

    def set_curriculum_phase(self, phase):
        self.curriculum_phase = phase

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_pose(self):
        """MUST match evaluate_rl.py get_robot_pose() exactly."""
        pos = self.robot_node.getPosition()
        rot = self.robot_node.getOrientation()
        yaw = math.atan2(rot[3], rot[0])
        return pos[0], pos[1], yaw

    def _build_observation(self):
        """
        Build 26-dim observation — MUST be bit-for-bit identical to
        evaluate_rl.py build_observation().
        """
        x, y, yaw = self._get_pose()

        # LiDAR → N_LIDAR_SECTORS buckets (min reading per sector)
        ranges = self.lidar.getRangeImage()
        n_rays = len(ranges)
        rpt = max(1, n_rays // N_LIDAR_SECTORS)
        sectors = np.ones(N_LIDAR_SECTORS, dtype=np.float32)
        for s in range(N_LIDAR_SECTORS):
            start = s * rpt
            for r in ranges[start:start + rpt]:
                d = 1.0 if (r == float("inf") or r > MAX_LIDAR_RANGE) else r / MAX_LIDAR_RANGE
                if d < sectors[s]:
                    sectors[s] = d

        # Carrot bearing (same formula as evaluate_rl.py build_observation)
        dx = self.carrot[0] - x
        dy = self.carrot[1] - y
        dist  = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) - yaw
        angle = (angle + math.pi) % (2 * math.pi) - math.pi   # wrap to [-π, π]

        obs = np.concatenate([
            sectors,
            [min(dist / 10.0, 1.0)],          # normalised distance [0,1]
            [(angle / math.pi + 1.0) / 2.0],  # normalised angle [0,1], 0.5=ahead
        ]).astype(np.float32)

        return obs, dist

    def _place_obstacles(self, rx, ry, cx, cy):
        """Place cardboard boxes along the robot→carrot line."""
        bury = [0.0, 0.0, -10.0]

        if self.curriculum_phase >= 2 and len(self.cardboard_obs) > 0:
            dx = cx - rx
            dy = cy - ry
            dist = math.hypot(dx, dy)
            ux = dx / max(dist, 1e-4)
            uy = dy / max(dist, 1e-4)
            px, py = -uy, ux  # perpendicular

            n_boxes = min(len(self.cardboard_obs), max(2, int(dist / 1.5)))

            for i, obs in enumerate(self.cardboard_obs):
                if i < n_boxes:
                    frac = (i + 1) / (n_boxes + 1)
                    d_along = max(dist * frac, 1.0 + i * 0.5)
                    lat = float(np.random.uniform(-0.6, 0.6))
                    bx = float(np.clip(rx + ux * d_along + px * lat, -9.0, 9.0))
                    by = float(np.clip(ry + uy * d_along + py * lat, -9.0, 9.0))
                    obs["trans"].setSFVec3f([bx, by, 0.3])
                    obs["node"].resetPhysics()
                else:
                    obs["trans"].setSFVec3f(bury)
                    obs["node"].resetPhysics()
        else:
            for obs in self.cardboard_obs:
                obs["trans"].setSFVec3f(bury)
                obs["node"].resetPhysics()

        if self.curriculum_phase >= 3:
            mid_x = (rx + cx) / 2.0
            mid_y = (ry + cy) / 2.0
            for obs in self.dynamic_obs:
                pos = [
                    float(np.clip(np.random.normal(mid_x, 0.3), -0.8, 0.8)),
                    float(np.clip(np.random.normal(mid_y, 1.5), -8.5, 8.5)),
                    1.25,
                ]
                obs["trans"].setSFVec3f(pos)
                obs["node"].resetPhysics()
        else:
            for obs in self.dynamic_obs:
                obs["trans"].setSFVec3f(bury)
                obs["node"].resetPhysics()

    # ── Gym interface ─────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Random spawn
        idx = np.random.randint(len(SPAWNS))
        rx, ry, ryaw = SPAWNS[idx]
        rx += float(np.random.uniform(-0.2, 0.2))
        ry += float(np.random.uniform(-0.3, 0.3))

        self.trans_field.setSFVec3f([rx, ry, 0.01])
        self.rot_field.setSFRotation([0, 0, 1, ryaw])
        self.robot_node.resetPhysics()

        # Initial carrot: 3-5m away at ±60° from spawn heading
        self.carrot = _new_carrot(rx, ry, ryaw, prev_carrot=None)

        self._place_obstacles(rx, ry, self.carrot[0], self.carrot[1])

        # Settle physics
        for _ in range(3):
            self.robot.step(TIME_STEP)

        self.step_count = 0
        self.prev_w     = 0.0
        obs, dist = self._build_observation()
        self.prev_dist = dist
        return obs, {}

    def step(self, action):
        # Action → velocity — MUST match evaluate_rl.py
        v = float((np.clip(action[0], -1, 1) + 1.0) / 2.0 * V_MAX)
        w = float(np.clip(action[1], -1, 1) * W_MAX)

        l, r = compute_wheel_speeds(v, w)
        self.left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, l)))
        self.right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, r)))

        for _ in range(FRAME_SKIP):
            if self.robot.step(TIME_STEP) == -1:
                obs, _ = self._build_observation()
                return obs, 0.0, True, False, {}

        self.step_count += 1
        obs, curr_dist = self._build_observation()
        x, y, _ = self._get_pose()

        # ── Reward ────────────────────────────────────────────────────────
        terminated = False
        truncated  = False

        # Collision check (same threshold as evaluate_rl.py)
        contacts = self.robot_node.getContactPoints(includeDescendants=True)
        hit = any(cp.point[2] > 0.01 for cp in contacts)

        if hit:
            reward     = R_COLLISION
            terminated = True

        elif curr_dist < CARROT_REACH_RADIUS:
            # ── Robot reached the current carrot ─────────────────────────
            # Give intermediate reward and advance the carrot.
            # Pass current yaw as fallback when prev_carrot direction is noisy.
            reward = R_CARROT_REACH
            x, y, curr_yaw = self._get_pose()
            self.carrot = _new_carrot(x, y, curr_yaw, prev_carrot=self.carrot)
            self._place_obstacles(x, y, self.carrot[0], self.carrot[1])
            obs, curr_dist = self._build_observation()  # rebuild with new carrot

        else:
            # Normal step: progress + shaping
            reward  = R_PROGRESS * (self.prev_dist - curr_dist)
            reward += R_TIME

            # Spinning-in-place penalty
            if v < 0.05 and abs(w) > 0.5:
                reward += R_SPIN_PENALTY

            # Rapid oscillation penalty (left↔right direction flip)
            if self.prev_w * w < 0 and abs(w) > 0.3:
                reward += R_OSCILLATION

            # Hard floor — guards against physics glitch position jumps
            reward = max(reward, -15.0)

        self.prev_w    = w
        self.prev_dist = curr_dist

        if not terminated and self.step_count >= MAX_STEPS:
            truncated = True

        return obs, reward, terminated, truncated, {}
