import numpy as np


def goal_cost(trajectory, goal):
    """
    Cost based on distance between trajectory endpoint and goal.
    Lower is better.
    """

    last_x = trajectory[-1, 0]
    last_y = trajectory[-1, 1]

    dx = goal[0] - last_x
    dy = goal[1] - last_y

    return np.hypot(dx, dy)


def obstacle_cost(trajectory, obstacles, robot_radius=0.2):
    """
    Penalizes trajectories that get too close to obstacles.
    """

    if len(obstacles) == 0:
        return 0.0

    min_dist = float("inf")

    for point in trajectory:

        px = point[0]
        py = point[1]

        dx = obstacles[:, 0] - px
        dy = obstacles[:, 1] - py

        d = np.hypot(dx, dy)

        closest = np.min(d)

        if closest < min_dist:
            min_dist = closest

    # Collision check
    if min_dist <= robot_radius:
        return float("inf")

    # Clamp distance so obstacle cost does not explode
    min_dist = max(min_dist, 0.3)

    return 1.0 / min_dist


def velocity_cost(v, max_speed):
    """
    Encourages higher forward speeds.
    """

    return 0.5 * (max_speed - v)


def compute_total_cost(
    trajectory,
    goal,
    obstacles,
    v,
    max_speed,
    w_goal=2.0,
    w_obs=0.4,
    w_vel=0.2,
):
    """
    Combined cost function for DWA.
    """

    g_cost = goal_cost(trajectory, goal)
    o_cost = obstacle_cost(trajectory, obstacles)
    v_cost = velocity_cost(v, max_speed)

    total_cost = (
        w_goal * g_cost
        + w_obs * o_cost
        + w_vel * v_cost
    )

    return total_cost