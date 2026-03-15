import numpy as np
import math


def heading_cost(trajectory, goal):
    last_x = trajectory[-1, 0]
    last_y = trajectory[-1, 1]
    last_yaw = trajectory[-1, 2]

    angle_to_goal = math.atan2(goal[1] - last_y, goal[0] - last_x)
    error=math.atan2(math.sin(angle_to_goal-last_yaw),
                       math.cos(angle_to_goal-last_yaw))
    return abs(error)


def obstacle_cost(trajectory, obstacles, robot_radius=0.18):
    if len(obstacles) == 0:
        return 0.0

    min_dist = float("inf")

    for point in trajectory:
        dx = obstacles[:, 0] - point[0]
        dy = obstacles[:, 1] - point[1]
        closest = float(np.min(np.hypot(dx, dy)))
        if closest < min_dist:
            min_dist = closest

    if min_dist <= robot_radius:
        return 20.0

    min_dist = max(min_dist, 0.3)
    return 1.0 / min_dist


def velocity_cost(v, max_speed):
    return max_speed - v


def compute_total_cost(
    trajectory,
    goal,
    obstacles,
    v,
    max_speed,
    path_world=None,
    w_heading=2.0,
    w_obs=1.2,
    w_vel=1.0,
):
    h_cost = heading_cost(trajectory, goal)
    o_cost = obstacle_cost(trajectory, obstacles)
    v_cost = velocity_cost(v, max_speed)

    return w_heading * h_cost + w_obs * o_cost + w_vel * v_cost