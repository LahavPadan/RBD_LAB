import shutil
from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay
from threading import Timer, Lock
from utils import tri_area, tri_centroid


class PointCloud(object):
    def __init__(self, app):
        self._app = app
        self._kdTree = None
        self._points: np.array = np.array([])
        self._lock_points = Lock()
        self._lock_kdTree = Lock()
        self.map_update = None
        self.update_map()

    def end(self):
        while self.map_update is not None:
            try:
                # cancel previous schedule
                self.map_update.cancel()
                self.map_update = None
            except AttributeError:
                pass

    def update_map(self):
        def read_map():
            # https://stackoverflow.com/questions/40769691/prevent-pandas-read-csv-treating-first-row-as-header-of-column-names
            df = pd.read_csv('../data/pointData.csv', header=None)
            df.columns = ['x', 'y', 'z']
            # https://stackoverflow.com/questions/17241004/how-do-i-convert-a-pandas-series-or-index-to-a-numpy-array
            px, py = df['x'].to_numpy(), df['y'].to_numpy()
            px *= 1000
            py *= 1000
            px = np.round_(px)
            py = np.round_(py)
            with self._lock_points:
                self._points = np.column_stack((px, py))

        while self._app.request_from_cpp("isTrackingLost") and self._app.running:
            pass

        print("[POINTCLOUD] updating pointcloud...")
        self._app.request_from_cpp("SAVEMAP")
        self._app.request_from_cpp("isMapSaved")
        # shutil.move("/tmp/pointData.csv", "../data/pointData.csv")
        read_map()
        with self._lock_kdTree and self._lock_points:
            self._kdTree = KDTree(self._points, leaf_size=2)
        print("[POINTCLOUD] pointcloud updated.")

        # maybe there is not need for this check
        if self._app.running:
            print("App is running, so map update is scheduled")
            # schedule next map update
            self.map_update = Timer(30, self.update_map)
            self.map_update.start()

    def plot(self):
        with self._lock_points:
            plt.plot(self._points[:, 0], self._points[:, 1], ".k")
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    def eval_movement_vector(self, point: np.array) -> np.array:
        print("[POINTCLOUD][eval_movement_vector] evaluating movement from ", point)
        point = point[:-1]  # all but last element
        print("point of dimension drop: ", point)
        # PROBABLY A SCALE ISSUE HERE

        # https://stackoverflow.com/questions/48126771/nearest-neighbour-search-kdtree
        # NN = nearest neighbors
        with self._lock_kdTree:
            nn_indices = self._kdTree.query_radius([point], r=60)  # NNs within distance of 60 of point
        with self._lock_points:
            all_nns = [self._points[idx] for idx in nn_indices]
        all_nns = np.array(*all_nns)
        """
        cost_map = {}
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
        """
        # https://gamedev.stackexchange.com/questions/95869/find-the-largest-empty-space-inside-a-cube-populated-with-a-point-cloud
        # https://stackoverflow.com/questions/62608710/delaunay-triangulation-simplices-scipy
        try:
            triangles = Delaunay(all_nns, qhull_options="QJ")
        except scipy.spatial.qhull.QhullError:
            while self._lock_points.acquire() and len(self._points) < 5:
                self._lock_points.release()
                if not self._app.running:
                    return point
                sleep(10)
            self._lock_points.release()

            # Edge case: 'point' itself is on pointcloud, resulting in an identity point. thus k=5, not k=4
            # do the previous query, but now, its possible to take points from arbitrary range
            nn_dist, nn_indices = self._kdTree.query([point], k=5)
            with self._lock_points:
                all_nns = [self._points[idx] for idx in nn_indices]
            all_nns = np.array(*all_nns)
            triangles = Delaunay(all_nns, qhull_options="QJ")

        # https://stackoverflow.com/questions/18296755/python-max-function-using-key-and-lambda-expression
        tri = max(all_nns[triangles.simplices], key=tri_area)
        centroid = tri_centroid(tri)
        print("centroid: ", centroid)
        """
        plt.triplot(all_nns[:, 0], all_nns[:, 1], triangles.simplices)
        plt.plot(all_nns[:, 0], all_nns[:, 1], '.k')
        plt.show()
        self.plot()
        """
        print(f'before movement_vector: centroid: {centroid}, point: {point}')
        movement_vector = np.subtract(centroid, point)  # vector subtraction
        print("after vector subtraction: ", movement_vector)
        movement_vector = np.append(movement_vector, 0)  # make it a 3D vector by adding dummy z coordinate
        movement_vector /= np.linalg.norm(movement_vector)  # normalize vector
        return movement_vector
