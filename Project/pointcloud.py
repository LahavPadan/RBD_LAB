import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from threading import Timer, Lock

show_animation = True


class PointCloud(object):
    def __init__(self, app):
        self._app = app
        self._kdTree = None
        self.px, self.py = None, None
        self._points = None
        self._lock_points = Lock()
        self._lock_kdTree = Lock()
        # start and goal position
        self.start = (10.0, 10.0)  # [cm]
        self.goal = (50.0, 50.0)  # [cm]
        self.grid_size = 2.0  # [cm]
        self.robot_radius = 1.0  # [cm]
        self.update_map()

    def update_map(self):
        def read_map():
            # https://stackoverflow.com/questions/40769691/prevent-pandas-read-csv-treating-first-row-as-header-of-column-names
            df = pd.read_csv('../data/pointData.csv', header=None)
            df.columns = ['x', 'y', 'z']
            # https://stackoverflow.com/questions/17241004/how-do-i-convert-a-pandas-series-or-index-to-a-numpy-array
            self.px, self.py = df['x'].to_numpy(), df['y'].to_numpy()
            self.px *= 1000
            self.py *= 1000
            self.px = np.round_(self.px)
            self.py = np.round_(self.py)
            with self._lock_points:
                self._points = np.column_stack((self.px, self.py)).tolist()

        """
        while self._app.request_from_cpp("isWall"):  # NOT CLEAR ENOUGH
            pass
        """

        print("[POINTCLOUD] updating pointcloud...")
        """
        self._app.request_from_cpp("SaveMap")
        self._app.request_from_cpp("isMapSaved")
        """
        # shutil.move("/tmp/pointData.csv", "../data/pointData.csv")
        read_map()
        with self._lock_kdTree and self._lock_points:
            self._kdTree = KDTree(self._points, leaf_size=2)
        # schedule next map update
        Timer(30, self.update_map)

    def plot(self):
        plt.plot(self.px, self.py, ".k")
        plt.plot(*self.start, "og")
        plt.plot(*self.goal, "xb")
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    """
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()

    return list(zip(rx, ry))[::-1]
    """

    # https://stackoverflow.com/questions/48126771/nearest-neighbour-search-kdtree
    def eval_cost_map(self, local_map) -> dict:
        with self._lock_points:
            points_numpy = np.array(self._points)
        # PROBABLY A SCALE ISSUE HERE

        cost_map = {}
        # NN = nearest neighbors
        print("READY FOR query_radius")
        #nn_indices = self._kdTree.query_radius([query_point], r=30)  # NNs within distance of 30 of point
        #all_nns = [np.array(self.points)[idx] for idx in nn_indices]
        with self._lock_kdTree:
            all_nn_indices = self._kdTree.query_radius([*local_map.values()], r=30)  # NNs within distance of 30 of point
        all_nns = [[points_numpy[idx] for idx in nn_indices] for nn_indices in all_nn_indices]
        print("all nns: ", all_nns)
        for nns in all_nns:
            print(nns)

        for i, key in enumerate(local_map.keys()):
            print(key)
            print(local_map[key])
            cost_map[key] = -len(all_nns[i])
            print(cost_map[key])

        return cost_map
