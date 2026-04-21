"""
Gymnasium environment wrapping a Webots TurtleBot3 Burger for RL training.

The robot is a Supervisor so it can teleport itself and obstacles.
Supports curriculum learning: empty → static obstacles → dynamic obstacles.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from controller import Supervisor

TIME_STEP = 64
MAX_SPEED = 6.28          # rad/s motor limit (same as DWA)
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.16
MAX_LIDAR_RANGE = 3.0
NUM_LIDAR_BUCKETS = 72    # 360° / 5° per bucket
FRAME_SKIP = 3            # repeat each action for 3 sim steps → 192ms per decision
MAX_EPISODE_STEPS = 500   # ~96s of sim time, prevents infinite wandering

# Discrete action → (linear_vel, angular_vel)
ACTION_TABLE = [
    (0.8,  0.0),    # 0: Fast Forward
    (0.4,  0.0),    # 1: Slow Forward
    (0.4,  0.75),   # 2: Soft Left
    (0.4, -0.75),   # 3: Soft Right
    (0.0,  1.5),    # 4: Hard Left
    (0.0, -1.5),    # 5: Hard Right
    (-0.2, 0.0),    # 6: Reverse
]


def compute_wheel_speeds(v, w):
    """Differential drive kinematics — identical to dwa.py."""
    left  = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return left, right


class TurtleBotEnv(gym.Env):
    """
    Obs:  Box(76,) — 72 LiDAR buckets, carrot distance, carrot angle, v, w
    Act:  Discrete(7)
    """
    metadata = {"render_modes": []}

    def __init__(self, curriculum_phase=1):
        super().__init__()
        self.curriculum_phase = curriculum_phase

        # ── Webots devices ────────────────────────────────────────────────
        self.robot = Supervisor()

        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(TIME_STEP)

        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # ── Supervisor handles ────────────────────────────────────────────
        self.robot_node = self.robot.getFromDef("TURTLEBOT3")
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")

        # Dynamic obstacle references (OBS1-OBS5) — Pedestrian PROTO nodes
        self.dynamic_obs = []
        for i in range(1, 6):
            node = self.robot.getFromDef(f"OBS{i}")
            if node:
                self.dynamic_obs.append({
                    "node": node,
                    "trans": node.getField("translation"),
                })

        # Static obstacle references (STATIC_OBS1-STATIC_OBS3) — SolidBox nodes
        self.static_obs = []
        for i in range(1, 4):
            node = self.robot.getFromDef(f"STATIC_OBS{i}")
            if node:
                self.static_obs.append({
                    "node": node,
                    "trans": node.getField("translation"),
                })

        # ── Spaces ────────────────────────────────────────────────────────
        # LiDAR buckets are [0,1], dist is [0,1], angle/v/w are [-1,1]
        low  = np.concatenate([np.zeros(72), [0.0, -1.0, -1.0, -1.0]])
        high = np.ones(76)
        self.observation_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
        )
        self.action_space = spaces.Discrete(7)

        # ── Internal state ────────────────────────────────────────────────
        self.current_v = 0.0
        self.current_w = 0.0
        self.carrot = np.array([0.0, 0.0])
        self.prev_dist = 0.0
        self.step_count = 0

        # One sim step so sensors produce their first readings
        self.robot.step(TIME_STEP)

    # ── Curriculum control ────────────────────────────────────────────────

    def set_curriculum_phase(self, phase):
        self.curriculum_phase = phase

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_pose(self):
        pos = self.robot_node.getPosition()
        rot = self.robot_node.getOrientation()
        # getOrientation() returns a 3×3 rotation matrix (row-major).
        # For a Z-up rotation: rot[0]=cos(θ), rot[3]=sin(θ).
        yaw = math.atan2(rot[3], rot[0])
        return pos[0], pos[1], yaw

    def _build_observation(self):
        x, y, yaw = self._get_pose()

        # ── LiDAR bucketing ───────────────────────────────────────────
        # Split the raw 360-ray scan into 72 five-degree sectors.
        # Each bucket holds the *minimum* reading in its sector, which
        # highlights the nearest obstacle — more useful than the mean
        # for collision avoidance.
        ranges = self.lidar.getRangeImage()
        n_rays = len(ranges)
        rays_per_bucket = max(1, n_rays // NUM_LIDAR_BUCKETS)
        buckets = np.ones(NUM_LIDAR_BUCKETS, dtype=np.float32)

        for b in range(NUM_LIDAR_BUCKETS):
            start = b * rays_per_bucket
            end = start + rays_per_bucket
            for r in ranges[start:end]:
                if r == float("inf") or r > MAX_LIDAR_RANGE:
                    d = 1.0
                else:
                    d = r / MAX_LIDAR_RANGE
                if d < buckets[b]:
                    buckets[b] = d

        # ── Carrot-relative features ──────────────────────────────────
        dx = self.carrot[0] - x
        dy = self.carrot[1] - y
        dist_to_carrot = math.hypot(dx, dy)

        angle_to_carrot = math.atan2(dy, dx) - yaw
        # Wrap to [-π, π]
        angle_to_carrot = (angle_to_carrot + math.pi) % (2 * math.pi) - math.pi

        obs = np.concatenate([
            buckets,
            [min(dist_to_carrot / 5.0, 1.0)],
            [angle_to_carrot / math.pi],
            [np.clip(self.current_v / 0.8, -1.0, 1.0)],
            [np.clip(self.current_w / 1.5, -1.0, 1.0)],
        ]).astype(np.float32)

        return obs, dist_to_carrot

    def _teleport_obstacles(self):
        """Bury or activate obstacles based on the current curriculum phase."""
        bury = [0.0, 0.0, -10.0]

        # Phase ≥ 2: scatter static boxes in the arena.
        # z=1.0 = half of the 2m SolidBox height — sits flush on the floor.
        if self.curriculum_phase >= 2:
            for obs in self.static_obs:
                pos = [
                    float(np.random.uniform(-3.0, 3.0)),
                    float(np.random.uniform(-3.0, 3.0)),
                    1.0,
                ]
                obs["trans"].setSFVec3f(pos)
        else:
            for obs in self.static_obs:
                obs["trans"].setSFVec3f(bury)

        # Phase ≥ 3: scatter dynamic pedestrians in the arena.
        # z=1.25 matches the Pedestrian PROTO origin used in dynamic-obstacle-aisle.wbt.
        if self.curriculum_phase >= 3:
            for obs in self.dynamic_obs:
                pos = [
                    float(np.random.uniform(-3.5, 3.5)),
                    float(np.random.uniform(-3.5, 3.5)),
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

        # Random spawn within the arena, well inside the walls
        rx = float(np.random.uniform(-4.0, 4.0))
        ry = float(np.random.uniform(-4.0, 4.0))
        ryaw = float(np.random.uniform(-math.pi, math.pi))

        self.trans_field.setSFVec3f([rx, ry, 0.01])
        self.rot_field.setSFRotation([0, 0, 1, ryaw])
        self.robot_node.resetPhysics()

        # Place a random carrot 1.5–2.5 m away, clamped inside the arena
        carrot_dist = float(np.random.uniform(1.5, 2.5))
        carrot_angle = float(np.random.uniform(-math.pi, math.pi))
        cx = np.clip(rx + carrot_dist * math.cos(carrot_angle), -4.5, 4.5)
        cy = np.clip(ry + carrot_dist * math.sin(carrot_angle), -4.5, 4.5)
        self.carrot = np.array([float(cx), float(cy)])

        # Reset motion
        self.current_v = 0.0
        self.current_w = 0.0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.step_count = 0

        self._teleport_obstacles()

        # Advance one step so sensors catch up after the teleport
        self.robot.step(TIME_STEP)

        obs, dist = self._build_observation()
        self.prev_dist = dist
        return obs, {}

    def step(self, action):
        v, w = ACTION_TABLE[action]
        left_speed, right_speed = compute_wheel_speeds(v, w)
        left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        # Frame skip — hold the same motor command for 3 sim ticks.
        # This gives the robot 192 ms of real motion per RL decision,
        # which is important: at 64 ms the robot barely moves, making
        # the progress reward too noisy to learn from.
        for _ in range(FRAME_SKIP):
            if self.robot.step(TIME_STEP) == -1:
                obs, _ = self._build_observation()
                return obs, 0.0, True, False, {}

        self.current_v = v
        self.current_w = w
        self.step_count += 1

        obs, curr_dist = self._build_observation()

        # ── Reward ────────────────────────────────────────────────────
        terminated = False
        truncated = False

        # Goal reached
        if curr_dist < 0.3:
            reward = 100.0
            terminated = True

        else:
            # Collision — same check as dwa.py, ignoring floor contacts
            contacts = self.robot_node.getContactPoints(includeDescendants=True)
            hit = any(cp.point[2] > 0.01 for cp in contacts)

            if hit:
                reward = -100.0
                terminated = True
            else:
                # Dense progress signal: closer → positive, farther → negative
                reward = (self.prev_dist - curr_dist) * 100.0
                reward -= 0.05  # small time penalty to discourage dawdling

                # Clearance penalty: discourage getting too close to walls/obstacles
                # buckets are normalized to 3.0m. 0.15 = 0.45m
                min_clearance = np.min(obs[:NUM_LIDAR_BUCKETS])
                if min_clearance < 0.15:
                    reward -= (0.15 - min_clearance) * 20.0

        # Episode length cap
        if not terminated and self.step_count >= MAX_EPISODE_STEPS:
            truncated = True

        self.prev_dist = curr_dist
        return obs, reward, terminated, truncated, {}
