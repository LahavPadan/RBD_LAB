import djitellopy
from threading import Event, Thread, Timer, Lock
from time import sleep, time
import random
import numpy as np
from math import cos, sin, radians
from pointcloud import PointCloud
from utils import calc_angle
import cv2
import color_tracker

FRAME_SIZE = (640, 480)
X = np.array([1, 0, 0])
Y = np.array([0, 1, 0])
Z = np.array([0, 0, 1])


class TelloCV(object):
    auto_movement_bool = True

    def __init__(self, app):
        self.app = app
        self.drone = djitellopy.Tello()
        self.speed = 10  # 10 cm/sec
        self._angle = 90  # drone facing forward = 90 degrees on the unit circle
        self.momentum_vec = np.array([cos(radians(self._angle)), sin(radians(self._angle))])  # 2D vector: (0, 1)

        self.slam_to_real_world_scale = 1
        self.slam_pose: np.array = np.array([])
        self.pointcloud = None

        self.ack = Event()  # signify acknowledgement from user after drone sends an authorization request

        self.move_lock = Lock()
        self.move_to_color_thread = None
        self.need_stay_in_air = True
        self.stay_in_air = None

        self.start_time_stay_in_air = 0

        # IM CREATING TON OF THREADS AND JOINING NONE
        Thread(target=self._init_drone).start()

    def _init_drone(self):
        def calibrate_map_scale():
            self.move(Z, 50)
            # *OR* height_drone = 50
            height_drone = self.drone.get_height()  # Get current height in cm
            # get coordinates from slam
            slam_height = self.app.request_from_cpp("listPose")[2]  # Z coordinate
            print("slam_height is ", slam_height)
            # calculate the real world scale
            self.slam_to_real_world_scale = height_drone / slam_height

        # self.drone.connect()
        # self.drone.streamon()
        # self.drone.set_speed(self.speed)

        while not self.app.request_from_cpp("isSlamInitialized") and self.app.running:
            print("[TELLO][INIT_DRONE] Waiting for ORB_SLAM2 to initialize...")
            # scan_env handles self.app.running check internally
            self.scan_env(exit_clause=lambda: self.app.request_from_cpp("isSlamInitialized"))
        print("[TELLO][INIT_DRONE] ORB_SLAM2 initialized.")
        # calibrate_map_scale()

        self.stay_in_air = Timer(15, self.move, args=[Z, 10 if TelloCV.auto_movement_bool else -10])

        self.start_time_stay_in_air = time()

        self.stay_in_air.start()
        self.slam_pose = np.array(self.app.request_from_cpp("listPose")) * self.slam_to_real_world_scale
        self.pointcloud = PointCloud(self.app)

        self.move_to_color_thread = Thread(target=self.move_to_color)
        self.move_to_color_thread.start()

    def end(self):
        print("[TELLO][END] Ending TelloCV...")
        # release any move authorization request
        # self.ack.set()  # the bit stays turned on

        print("[TELLO][END] Ending stay_in_air...")
        self.need_stay_in_air = False
        self.stay_in_air.cancel()
        while self.stay_in_air is not None:
            try:
                # cancel previous schedule
                self.stay_in_air.cancel()
                self.stay_in_air.join()
                self.stay_in_air = None
            except AttributeError:
                pass
        print("[TELLO][END] stay_in_air ended.")
        # wait for the previous stay_in_air schedule to finish
        # sleep(15)

        self.pointcloud.end()

        print("About to join move_to_color")
        self.move_to_color_thread.join()
        print("move_to_color joined.")
        # self.drone.land()
        # self.drone.end()

    def _authorization_request(self, direction, cm):
        if self.app.running:
            # clear previous/accidental authorization bit
            self.ack.clear()
        while True:
            print(f'[TELLO][authorization request] {direction}, {cm} cm. Press s to move')
            ack_accepted = self.ack.wait(timeout=13)  # wait about a quarter of a minute
            if ack_accepted:
                break

        if not self.app.running:
            self.ack.set()  # release all waiting threads if app is not running

    def _handle_stay_in_air(function):
        def wrapper(self, *args, **kwargs):
            with self.move_lock:  # lock also prevents scheduled call and actual call, to enter 'move' simultaneously
                print("time passed since last schedule: ", time() - self.start_time_stay_in_air)
                try:
                    # cancel previous schedule
                    self.stay_in_air.cancel()
                except AttributeError:
                    pass
                # call wrapped function
                res = function(self, *args, **kwargs)

                # schedule next
                if self.need_stay_in_air:
                    TelloCV.auto_movement_bool = not TelloCV.auto_movement_bool
                    self.stay_in_air = Timer(15, self.move, args=[Z, 10 if TelloCV.auto_movement_bool else -10])
                    self.start_time_stay_in_air = time()
                    self.stay_in_air.start()
            return res
        return wrapper

    def __update_movement_vector(self, diff_angle):
        # self._angle is only used inside cosine and sine, thus, no problem that self._angle will become
        # arbitrarily large
        self._angle = self._angle + diff_angle
        self.momentum_vec = np.array([cos(radians(self._angle)), sin(radians(self._angle))])

    @_handle_stay_in_air
    def scan_env(self, exit_clause, *args, **kwargs):
        print("[TELLO][SCAN_ENV] scanning environment...")
        degree_per_iteration = 30
        for _ in range(0, 360, degree_per_iteration):

            if exit_clause() or not self.app.running:
                break

            self.__update_movement_vector(diff_angle=degree_per_iteration)
            # self.drone.rotate_clockwise(degree_per_iteration)
            # wait for drone to rotate
            sleep(4)

        print("[TELLO][SCAN_ENV] done.")

    @_handle_stay_in_air
    def move(self, vector: np.array, cm, *args, **kwargs) -> bool:
        """
        :param vector: (3D vector, normalized) direction to move.
        :param cm: magnitude of vector in cm.
        :return: True if and only if drone actually traveled.
        """
        vector = vector * cm  # it also switches vector-entries-sign if cm < 0
        cm = abs(cm)
        print(f'[TELLO][MOVE] Method was called with; vector: {vector}, cm: {cm}')
        # change drone angle, then, check if safe to continue in that direction
        vector_2d = vector[:-1]
        angle = calc_angle(self.momentum_vec, vector_2d)  # calculate angle between two 2D vectors
        self.__update_movement_vector(diff_angle=angle)
        """
        if angle >= 0:
            self.drone.rotate_counter_clockwise(angle)
        elif angle < 0:
            self.drone.rotate_clockwise(-1 * angle)
        """
        print(f'[TELLO] rotating in angle of {angle}... ')
        # wait for the drone to rotate
        sleep(10)

        X_movement = np.dot(vector, X) * X
        X_norm = np.linalg.norm(X_movement)
        Y_movement = np.dot(vector, Y) * Y
        Y_norm = np.linalg.norm(Y_movement)
        Z_movement = np.dot(vector, Z) * Z
        Z_norm = np.linalg.norm(Z_movement)

        X_movement = 'right' if X_movement[0] > 0 else 'left'
        Y_movement = 'forward' if Y_movement[1] > 0 else 'back'
        Z_movement = 'up' if Z_movement[2] > 0 else 'down'

        isWall = self.app.request_from_cpp("isWall")
        print("isWall= ", isWall)
        if isWall:
            return False

        for axis in ["X_", "Y_", "Z_"]:
            # https://stackoverflow.com/questions/9437726/how-to-get-the-value-of-a-variable-given-its-name-in-a-string
            movement = locals()[axis + "movement"]
            norm = locals()[axis + "norm"]
            if norm != 0:
                # self._authorization_request(movement, norm)
                if not self.app.running:
                    break
                # self.drone.move(movement, norm)
                # wait for the drone to move
                # sleep((norm / self.speed) + 1)
                sleep(5)

        # update pose variable
        self.slam_pose = np.array(self.app.request_from_cpp("listPose")) * self.slam_to_real_world_scale
        print("self.slam_pose is ", self.slam_pose)
        # for the future: maybe track drone pose as given by commands throughout the program
        return True

    def wander_around(self, exit_clause):
        # point = 0.3, 0.3
        while not exit_clause() and self.app.running:
            print("[TELLO][WANDER_AROUND] searching where to go...")
            cm = 20
            moved = self.move(self.pointcloud.eval_movement_vector(self.slam_pose, self.momentum_vec), cm)
            while self.app.running and not moved:
                print("[TELLO][WANDER_AROUND] attempting to move away from wall...")
                # move away from wall, choose randomly another target to go to
                axis = random.choice([X, Y])
                direction = random.choice([cm, -cm])
                moved = self.move(axis, direction)
            self.scan_env(exit_clause)

    """
    the hue range in
    opencv is 180, normally it is 360
    green_lower = (50, 50, 50)
    green_upper = (70, 255, 255)
    red_lower = (0, 50, 50)
    red_upper = (20, 255, 255)
    blue_lower = (110, 50, 50)
    upper_blue = (130, 255, 255)
    """

    def move_to_color(self):
        tracker = color_tracker.ColorTracker(max_nb_of_objects=1, max_nb_of_points=20, debug=True)
        tracker.set_tracking_callback(self.tracker_callback)
        # Define your custom Lower and Upper HSV values
        tracking_thread = Thread(target=tracker.track, args=[self.app.factory(), [155, 103, 82], [178, 255, 255]])
        tracking_thread.start()

        def get_tracked_object():
            obj = None

            def is_object_found():
                nonlocal obj
                try:
                    x, y, x_w, y_h = tracker.tracked_objects[-1].last_bbox
                    width = (x_w - x)
                    height = (y_h - y)
                    area = width * height
                    center = x + width / 2, y + height / 2  # ABOUT THIS ONE IM NOT TOO SURE - ITS THE CENTER IN FRAME
                    # NOT NECESSARILY IN REAL WORLD
                    if area > 30000:  # filter-out false positives
                        obj = {'area': area, 'center': center}
                        return True
                except IndexError:
                    pass
                return False

            self.wander_around(exit_clause=is_object_found)
            if not self.app.running:
                print("Returning that dummy object found dict")
                # returning dummy object_found dict
                obj = {'area': 0, 'center': (FRAME_SIZE[0] / 2, FRAME_SIZE[1] / 2)}

            return obj

        res = get_tracked_object()
        print(f"[TELLO][TRACKER] OBJECT FOUND; center:{res['center']}, area:{res['area']}."
              f" Frame-center is: {(FRAME_SIZE[0] / 2, FRAME_SIZE[1] / 2)}")
        # repose the drone to face tracked object
        xoffset = res['center'][0] - FRAME_SIZE[
            0] / 2  # if object_center_x > FRAME_SIZE_x/2, you should move to the right
        yoffset = res['center'][1] - FRAME_SIZE[1] / 2
        self.move(X, xoffset)
        self.move(Z, yoffset)  # y axis in frame = z axis in real world

        print("move forward until object is close enough")
        # move forward until object is close enough
        cm = 10
        self.move(Y, cm)
        res = get_tracked_object()
        while res['area'] < 180000 and self.app.running:
            self.move(Y, cm)
            res = get_tracked_object()

        print("[TELLO][TRACKER] Stopping tracker...")
        tracker.stop_tracking()
        tracking_thread.join()
        print("[TELLO][TRACKER] Tracker thread joined.")

    def tracker_callback(self, t: color_tracker.ColorTracker):
        frame = t.debug_frame
        # Show the direction vector output in the cv2 window
        cv2.arrowedLine(frame, (int(FRAME_SIZE[0] / 2), int(FRAME_SIZE[1] / 2)),
                        (int(FRAME_SIZE[0] / 2 + 100 * cos(radians(self._angle))),
                         int(FRAME_SIZE[1] / 2 - 100 * sin(radians(self._angle)))),
                        (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BACKWARD", (int(FRAME_SIZE[0] / 2) - 60, FRAME_SIZE[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "FORWARD", (int(FRAME_SIZE[0] / 2) - 60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT", (FRAME_SIZE[0] - 100, int(FRAME_SIZE[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(frame, "LEFT", (0, int(FRAME_SIZE[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.imshow("debug", frame)
        cv2.waitKey(1)
