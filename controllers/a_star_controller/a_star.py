import heapq
import math

class AStarPlanner:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape

        self.neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def in_bounds(self, node):
        r, c = node
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, node):
        r, c = node
        return self.grid[r, c] == 0

    def plan(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from = {}
        g_cost = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for dr, dc in self.neighbors:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self.in_bounds(neighbor):
                    continue
                if not self.is_free(neighbor):
                    continue

                step_cost = math.hypot(dr, dc)
                new_cost = g_cost[current] + step_cost

                if new_cost < g_cost.get(neighbor, float("inf")):
                    g_cost[neighbor] = new_cost
                    f_cost = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_cost, neighbor))
                    came_from[neighbor] = current

        return None

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
