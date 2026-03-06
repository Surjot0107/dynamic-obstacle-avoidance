import numpy as np

from trajectory_simulator import simulate_trajectory
from cost_functions import compute_total_cost


class DWAPlanner:

    def __init__(
        self,
        max_speed=0.8,
        min_speed=0.0,
        max_yaw_rate=1.5,
        max_accel=0.4,
        max_yaw_accel=1.5,
        v_res=0.1,
        w_res=0.15,
        dt=0.1,
        predict_time=2.5,
    ):
        """
        DWA parameter initialization
        """

        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_yaw_rate = max_yaw_rate

        self.max_accel = max_accel
        self.max_yaw_accel = max_yaw_accel

        self.v_res = v_res
        self.w_res = w_res

        self.dt = dt
        self.predict_time = predict_time


    def compute_dynamic_window(self, v, w):
        """
        Compute the dynamic window based on current velocity.
        """

        v_min = max(self.min_speed, v - self.max_accel * self.dt)
        v_max = min(self.max_speed, v + self.max_accel * self.dt)

        w_min = max(-self.max_yaw_rate, w - self.max_yaw_accel * self.dt)
        w_max = min(self.max_yaw_rate, w + self.max_yaw_accel * self.dt)

        return v_min, v_max, w_min, w_max


    def plan(self, state, goal, obstacles):
        """
        Main DWA step.

        state = [x, y, yaw, v, w]
        goal  = [gx, gy]
        obstacles = Nx2 obstacle points
        """

        x, y, yaw, v, w = state
        trajectories = []

        # Compute dynamic window
        v_min, v_max, w_min, w_max = self.compute_dynamic_window(v, w)

        # ensure forward samples exist
        if v_max < 0.1:
            v_max = 0.3

        best_cost = float("inf")
        best_v = 0.0
        best_w = 0.0

        # Sample velocities
        for v_sample in np.arange(v_min, v_max + self.v_res, self.v_res):

            for w_sample in np.arange(w_min, w_max + self.w_res, self.w_res):

                # Simulate trajectory
                trajectory = simulate_trajectory(
                    [x, y, yaw],
                    v_sample,
                    w_sample,
                    self.dt,
                    self.predict_time,
                )
                trajectories.append(trajectory)

                # Compute cost
                cost = compute_total_cost(
                    trajectory,
                    goal,
                    obstacles,
                    v_sample,
                    self.max_speed,
                )

                if cost < best_cost:

                    best_cost = cost
                    best_v = v_sample
                    best_w = w_sample

        return best_v, best_w, trajectories