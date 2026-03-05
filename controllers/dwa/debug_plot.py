import matplotlib.pyplot as plt


def plot_dwa_debug(robot_pos, goal, obstacles, trajectories):

    plt.clf()

    # robot
    plt.scatter(robot_pos[0], robot_pos[1], s=100)

    # goal
    plt.scatter(goal[0], goal[1], s=100)

    # obstacles
    if len(obstacles) > 0:
        plt.scatter(obstacles[:, 0], obstacles[:, 1], s=5)

    # trajectories
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1])

    plt.axis("equal")
    plt.pause(0.001)