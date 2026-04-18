import numpy as np


def simulate_trajectory(state, v, w, dt=0.1, predict_time=2.0):

    x, y, yaw = state

    trajectory = []

    time = 0.0

    while time <= predict_time:

        trajectory.append([x, y, yaw])

        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += w * dt

        time += dt

    return np.array(trajectory)