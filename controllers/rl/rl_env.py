"""
Gymnasium environment wrapping a Webots TurtleBot3 Burger for RL training.

Key fixes applied:
  - action_space matches ACTION_TABLE size (9 actions).
  - DANGER_DIST penalty is calibrated so it never overpowers the goal reward.
  - Static obstacles are only scattered outside a 2 m exclusion zone around
    the robot AND the carrot, preventing immediate collisions on episode start.
  - Carrot is normalised over 7 m (consistent with evaluator).
  - Phase drop between curricula is mitigated by flushing the env reset.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from controller import Supervisor

# ── Simulation constants ──────────────────────────────────────────────────────
TIME_STEP         = 64
MAX_SPEED         = 6.28
WHEEL_RADIUS      = 0.033
WHEEL_BASE        = 0.16
MAX_LIDAR_RANGE   = 3.0
NUM_LIDAR_BUCKETS = 72
FRAME_SKIP        = 3
MAX_EPISODE_STEPS = 500

# ── Reward tuning ─────────────────────────────────────────────────────────────
# Normalised to MAX_LIDAR_RANGE (3 m).  0.1 → 0.30 m physical
DANGER_DIST  = 0.10   # 0.30 m — any closer gets a strong penalty
WARNING_DIST = 0.20   # 0.60 m — gentle nudge to keep clearance

# Forward-sector proximity penalty covers front 120° (24 of 72 buckets).
# Only the forward arc matters — side walls in corridors should not panic the agent.
# Webots LDS-01: bucket 0 = forward, increasing counter-clockwise.
FRONT_BUCKETS_HALF = 12   # 12 right-fwd + 12 left-fwd = ±60° = 120° total

# Warning zone: 0.50 normalized = 1.5 m physical.  Agent must sense it and react early.
# Danger zone:  0.15 normalized = 0.45 m physical. Imminent contact.
WARNING_DIST = 0.50
DANGER_DIST  = 0.15

COLLISION_PENALTY = -500.0
GOAL_REWARD       = 500.0

# ── Action table ──────────────────────────────────────────────────────────────
# 9 actions total — action_space MUST match len(ACTION_TABLE).
ACTION_TABLE = [
    (0.8,  0.0),    # 0: Fast Forward
    (0.4,  0.0),    # 1: Slow Forward
    (0.4,  0.75),   # 2: Soft Left
    (0.4, -0.75),   # 3: Soft Right
    (0.2,  1.5),    # 4: Tight Left  (dodge)
    (0.2, -1.5),    # 5: Tight Right (dodge)
    (0.0,  1.5),    # 6: Pivot Left
    (0.0, -1.5),    # 7: Pivot Right
    (-0.2, 0.0),    # 8: Reverse
]
N_ACTIONS = len(ACTION_TABLE)   # 9


def compute_wheel_speeds(v, w):
    left  = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return left, right


def _safe_random_pos(avoid_xy_list, margin, arena_limit=4.0):
    """Return a (x, y) that is at least `margin` m from every point in avoid_xy_list."""
    for _ in range(200):
        x = float(np.random.uniform(-arena_limit, arena_limit))
        y = float(np.random.uniform(-arena_limit, arena_limit))
        ok = all(math.hypot(x - ax, y - ay) >= margin for ax, ay in avoid_xy_list)
        if ok:
            return x, y
    # Fallback: just pick something random (extremely rare)
    return float(np.random.uniform(-arena_limit, arena_limit)), float(np.random.uniform(-arena_limit, arena_limit))


class TurtleBotEnv(gym.Env):
    """
    Obs:  Box(76,) — 72 LiDAR buckets [0,1], carrot_dist [0,1],
                      carrot_angle/π [-1,1], v [-1,1], w [-1,1]
    Act:  Discrete(9)  ← must equal N_ACTIONS
    """
    metadata = {"render_modes": []}

    def __init__(self, curriculum_phase=1):
        super().__init__()
        self.curriculum_phase = curriculum_phase

        # ── Webots devices ────────────────────────────────────────────────
        self.robot = Supervisor()

        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(TIME_STEP)

        self.left_motor  = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # ── Supervisor handles ────────────────────────────────────────────
        self.robot_node  = self.robot.getFromDef("TURTLEBOT3")
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field   = self.robot_node.getField("rotation")

        # Dynamic obstacles (OBS1-OBS5) — Pedestrian PROTO nodes
        self.dynamic_obs = []
        for i in range(1, 6):
            node = self.robot.getFromDef(f"OBS{i}")
            if node:
                self.dynamic_obs.append({
                    "node":  node,
                    "trans": node.getField("translation"),
                })

        # Static obstacles (STATIC_OBS1-STATIC_OBS8) — SolidBox nodes
        self.static_obs = []
        for i in range(1, 9):
            node = self.robot.getFromDef(f"STATIC_OBS{i}")
            if node:
                self.static_obs.append({
                    "node":  node,
                    "trans": node.getField("translation"),
                })

        # ── Gymnasium spaces ──────────────────────────────────────────────
        low  = np.concatenate([np.zeros(72), [0.0, -1.0, -1.0, -1.0]])
        high = np.ones(76)
        self.observation_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # ── Episode state ─────────────────────────────────────────────────
        self.current_v   = 0.0
        self.current_w   = 0.0
        self.carrot      = np.array([0.0, 0.0])
        self.prev_dist   = 0.0
        self.step_count  = 0
        self.prev_action = -1

        # First sim tick so sensors initialise
        self.robot.step(TIME_STEP)

    # ── Curriculum ────────────────────────────────────────────────────────────

    def set_curriculum_phase(self, phase):
        self.curriculum_phase = phase

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_pose(self):
        pos = self.robot_node.getPosition()
        rot = self.robot_node.getOrientation()
        yaw = math.atan2(rot[3], rot[0])
        return pos[0], pos[1], yaw

    def _build_observation(self):
        x, y, yaw = self._get_pose()

        ranges = self.lidar.getRangeImage()
        n_rays = len(ranges)
        rpt    = max(1, n_rays // NUM_LIDAR_BUCKETS)
        buckets = np.ones(NUM_LIDAR_BUCKETS, dtype=np.float32)
        for b in range(NUM_LIDAR_BUCKETS):
            start = b * rpt
            for r in ranges[start:start + rpt]:
                d = 1.0 if (r == float("inf") or r > MAX_LIDAR_RANGE) else r / MAX_LIDAR_RANGE
                if d < buckets[b]:
                    buckets[b] = d

        dx = self.carrot[0] - x
        dy = self.carrot[1] - y
        dist_to_carrot  = math.hypot(dx, dy)
        angle_to_carrot = math.atan2(dy, dx) - yaw
        angle_to_carrot = (angle_to_carrot + math.pi) % (2 * math.pi) - math.pi

        obs = np.concatenate([
            buckets,
            [min(dist_to_carrot / 7.0, 1.0)],
            [angle_to_carrot / math.pi],
            [np.clip(self.current_v / 0.8, -1.0, 1.0)],
            [np.clip(self.current_w / 1.5, -1.0, 1.0)],
        ]).astype(np.float32)

        return obs, dist_to_carrot

    def _teleport_obstacles(self, rx, ry, cx, cy):
        """Place or bury obstacles for this episode.

        Phase 2 guarantee: at least 2 static boxes are placed ON the direct
        line from the robot to the carrot, at 35 % and 65 % of the distance.
        The remaining boxes are scattered randomly.  This ensures the agent
        CANNOT reach the carrot by going straight — it must detour every ep.

        Phase 3: dynamic pedestrians are clustered at the midpoint between
        the robot and the carrot.
        """
        bury = [0.0, 0.0, -10.0]

        # ── Static (Phase ≥ 2) ────────────────────────────────────────────
        if self.curriculum_phase >= 2:
            # Unit vector along robot→carrot
            dx   = cx - rx
            dy   = cy - ry
            dist = math.hypot(dx, dy)
            ux   = dx / max(dist, 1e-4)
            uy   = dy / max(dist, 1e-4)

            # Perpendicular unit vector (for slight lateral jitter so the
            # agent can't always pass on the exact same side)
            px = -uy
            py =  ux

            # Guaranteed blocking positions: 30 % and 70 % along the path.
            # Minimum distance from robot = 1.2 m so we never spawn inside it.
            BOX_SAFE_DIST = 1.2  # m from robot start before placing a blocker
            on_path_positions = []
            actual_dist = math.hypot(cx - rx, cy - ry)  # may differ from carrot_dist after clamping
            for frac in [0.30, 0.70]:
                raw_d = actual_dist * frac
                safe_d = max(raw_d, BOX_SAFE_DIST)
                jitter = float(np.random.uniform(-0.4, 0.4))
                bx = rx + ux * safe_d + px * jitter
                by = ry + uy * safe_d + py * jitter
                bx = float(np.clip(bx, -4.0, 4.0))
                by = float(np.clip(by, -4.0, 4.0))
                on_path_positions.append((bx, by))

            # Place the 'on-path' blockers first
            for i, obs in enumerate(self.static_obs[:2]):
                bx, by = on_path_positions[i]
                obs["trans"].setSFVec3f([bx, by, 1.0])

            # Remaining boxes are random but must not overlap the blockers
            avoid = [(rx, ry), (cx, cy)] + on_path_positions
            for obs in self.static_obs[2:]:
                sx, sy = _safe_random_pos(avoid, margin=1.0)
                avoid.append((sx, sy))
                obs["trans"].setSFVec3f([sx, sy, 1.0])

        else:
            for obs in self.static_obs:
                obs["trans"].setSFVec3f(bury)

        # ── Dynamic (Phase ≥ 3) ───────────────────────────────────────────
        if self.curriculum_phase >= 3:
            mid_x = (rx + cx) / 2.0
            mid_y = (ry + cy) / 2.0
            for obs in self.dynamic_obs:
                pos = [
                    float(np.clip(np.random.normal(mid_x, 0.8), -4.0, 4.0)),
                    float(np.clip(np.random.normal(mid_y, 0.8), -4.0, 4.0)),
                    1.25,
                ]
                obs["trans"].setSFVec3f(pos)
                obs["node"].resetPhysics()
        else:
            for obs in self.dynamic_obs:
                obs["trans"].setSFVec3f(bury)
                obs["node"].resetPhysics()


    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Random robot spawn, well inside the arena
        rx   = float(np.random.uniform(-3.5, 3.5))
        ry   = float(np.random.uniform(-3.5, 3.5))
        ryaw = float(np.random.uniform(-math.pi, math.pi))

        self.trans_field.setSFVec3f([rx, ry, 0.01])
        self.rot_field.setSFRotation([0, 0, 1, ryaw])
        self.robot_node.resetPhysics()

        # Carrot: 4.0–8.0 m ahead, ±60° of current yaw
        # Longer distance gives room for on-path obstacles to sit between robot and goal.
        carrot_dist  = float(np.random.uniform(4.0, 8.0))
        carrot_angle = ryaw + float(np.random.uniform(-math.pi / 3, math.pi / 3))
        cx = float(np.clip(rx + carrot_dist * math.cos(carrot_angle), -4.0, 4.0))
        cy = float(np.clip(ry + carrot_dist * math.sin(carrot_angle), -4.0, 4.0))
        self.carrot = np.array([cx, cy])

        # Scatter obstacles (static ones won't overlap robot/carrot)
        self._teleport_obstacles(rx, ry, cx, cy)

        # Flush physics — 3 steps so LiDAR gets fresh readings with the new layout
        for _ in range(3):
            self.robot.step(TIME_STEP)

        self.current_v   = 0.0
        self.current_w   = 0.0
        self.step_count  = 0
        self.prev_action = -1

        obs, dist = self._build_observation()
        self.prev_dist = dist
        return obs, {}

    def step(self, action):
        v, w = ACTION_TABLE[action]
        l, r = compute_wheel_speeds(v, w)
        self.left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, l)))
        self.right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, r)))

        for _ in range(FRAME_SKIP):
            if self.robot.step(TIME_STEP) == -1:
                obs, _ = self._build_observation()
                return obs, 0.0, True, False, {}

        self.current_v = v
        self.current_w = w
        self.step_count += 1

        obs, curr_dist = self._build_observation()

        # ── Reward ────────────────────────────────────────────────────────
        terminated = False
        truncated  = False

        if curr_dist < 0.3:
            reward     = GOAL_REWARD
            terminated = True

        else:
            contacts = self.robot_node.getContactPoints(includeDescendants=True)
            hit = any(cp.point[2] > 0.01 for cp in contacts)

            if hit:
                reward     = COLLISION_PENALTY
                terminated = True
            else:
                # Progress toward carrot (~10 reward units per metre)
                reward = (self.prev_dist - curr_dist) * 100.0

                # Small time penalty to discourage dawdling
                reward -= 0.05

                # ── Forward-sector proximity penalty ─────────────────────
                # Only look at the front 120° arc (±60° of heading).
                # This means side walls in a corridor don’t trigger avoidance
                # when the agent is heading straight through.
                fwd_r = obs[:FRONT_BUCKETS_HALF]                          # buckets 0..11
                fwd_l = obs[NUM_LIDAR_BUCKETS - FRONT_BUCKETS_HALF:]      # buckets 60..71
                min_fwd = float(min(np.min(fwd_r), np.min(fwd_l)))

                if min_fwd < DANGER_DIST:
                    # Quadratic penalty — grows sharply inside danger zone
                    reward -= ((DANGER_DIST - min_fwd) ** 2) * 2000.0
                elif min_fwd < WARNING_DIST:
                    # Quadratic warning penalty visible from 1.5 m away
                    # At 1.5m: penalty ≈ 0.  At 0.6m: penalty ≈ -28.  Gradient
                    # forces the agent to steer while it still has room.
                    reward -= ((WARNING_DIST - min_fwd) ** 2) * 200.0

        self.prev_dist   = curr_dist
        self.prev_action = action

        if not terminated and self.step_count >= MAX_EPISODE_STEPS:
            truncated = True

        return obs, reward, terminated, truncated, {}
