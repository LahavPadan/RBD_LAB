import djitellopy
from threading import Event, Thread, Timer, Lock
from time import sleep
from pointcloud import PointCloud
from face_recognition import Tracker
from utils import current_and_next
import numpy as np
from math import atan2, pi, cos, sin, radians, sqrt
import cv2
import color_tracker

FRAME_SIZE = (640, 480)


class TelloCV(object):
    auto_movement_bool = True

    def __init__(self, app):
        self.app = app
        self.drone = djitellopy.Tello()
        self.angle = 0

        self.slam_to_real_world_scale = 1
        self.slam_pose = None
        self.pointcloud = None

        self.ack = Event()  # signify acknowledgement from user after drone sends an authorization request

        self.move_lock = Lock()
        """
        self.auto_navigator = None
        """
        self.need_stay_in_air = True
        self.stay_in_air = None

        # IM CREATING TON OF THREADS AND JOINING NONE
        init_thread = Thread(target=self.init_drone)
        init_thread.start()

    def init_drone(self):
        def calibrate_map_scale():
            self.drone.move('Z', 50)
            # *OR* height_drone = 50
            height_drone = self.drone.get_height()  # Get current height in cm
            # get coordinates from slam
            slam_height = self.app.request_from_cpp("Pose")[2]  # Z coordinate
            print("slam_height is ", slam_height)
            # calculate the real world scale
            self.slam_to_real_world_scale = height_drone / slam_height

        #self.drone.connect()
        #self.drone.streamon()
        while not self.app.request_from_cpp("isSlamInitialized"):
            # self.scan_env()
            pass
        # calibrate_map_scale()

        self.stay_in_air = Timer(15, self.move,
                                 args=['down' if TelloCV.auto_movement_bool else 'up', 10])
        self.stay_in_air.start()
        self.slam_pose = np.array(self.app.request_from_cpp("Pose")) * self.slam_to_real_world_scale
        self.pointcloud = PointCloud(self.app)
        """
        self.auto_navigator = Thread(target=self.auto_navigation)
        self.auto_navigator.start()
        """
        move_to_ball_thread = Thread(target=self.move_to_ball)
        move_to_ball_thread.start()

    def end(self):
        self.need_stay_in_air = False
        # wait for the previous stay_in_air schedule to finish
        sleep(15)
        self.drone.land()
        """
        self.auto_navigator.join()  # NOT SURE I CAN EVEN JOIN IT
        """
        self.drone.end()

    def authorization_request(self):
        # clear previous/accidental authorization bit
        self.ack.clear()
        while True:
            print("[TELLO] [authorization request] Press s to move")
            ack_accepted = self.ack.wait(timeout=30)  # wait for half a minute
            if ack_accepted:
                break

    def scan_env(self):
        degree_slice = 10
        print("[Drone][Scan] scanning environment for ORB_SLAM2...")
        with self.move_lock:
            for i in range(0, 360, degree_slice):
                """
                self.drone.rotate_clockwise(degree_slice)
                self.angle = (self.angle + degree_slice) % 360  # IM CONCERNED ABOUT THIS
                # wait for drone to rotate and orbslam to initialize
                """
                sleep(1)

        print("[Drone][Scan] Done.")

    def move(self, axis, cm):
        print(f'[TELLO] move method was called, axis: {axis}, cm: {cm}')
        with self.move_lock:  # lock prevents scheduled call and actual call, to enter 'move' simultaneously

            # cancel previous schedule
            self.stay_in_air.cancel()

            # change drone angle, then, check if safe to continue in that direction
            x1, y1 = cos(radians(self.angle)), sin(radians(self.angle))
            x2, y2 = 0, 0
            direction = None
            if axis == 'X':
                if cm < 0:
                    direction = 'left'
                    x2, y2 = (-1, 0)
                else:
                    direction = 'right'
                    x2, y2 = (1, 0)

            elif axis == 'Y':
                if cm < 0:
                    direction = 'back'
                    x2, y2 = (0, -1)
                else:
                    direction = 'forward'
                    x2, y2 = (0, 1)

            elif axis == 'Z':
                if cm < 0:
                    direction = 'down'
                else:
                    direction = 'up'
                x2, y2 = x1, y1

            dot = x1 * x2 + y1 * y2  # dot product
            det = x1 * y2 - y1 * x2  # determinant
            angle = int(atan2(det, dot) * (180 / pi))  # atan2(y, x) or atan2(sin, cos)
            self.angle = self.angle + angle
            """
            if angle >= 0:
                self.drone.rotate_counter_clockwise(angle)
            elif angle < 0:
                self.drone.rotate_clockwise(-1 * angle)
            """
            print(f'[TELLO] rotating in angle of {angle}... ')
            # wait for the drone to rotate
            sleep(2)

            isWall = self.app.request_from_cpp("IsWall")
            print("isWall= ", isWall)
            """
            if not isWall:
                self.authorization_request()
                if direction == 'right' or direction == 'left' or direction == 'back':
                    self.drone.move_forward(cm)
                    pass
                else:
                    self.drone.move(direction, cm)
                    pass
                # wait for the drone to move
                sleep(1)
            """

            # update pose variables
            self.slam_pose = np.array(self.app.request_from_cpp("Pose")) * self.slam_to_real_world_scale
            print("self.slam_pose is ", self.slam_pose)
            # get drone pose
            self.drone_pose = None
            # get drone pose

            # schedule next
            if self.need_stay_in_air:
                TelloCV.auto_movement_bool = not TelloCV.auto_movement_bool
                self.stay_in_air = Timer(15, self.move, args=['down' if TelloCV.auto_movement_bool else 'up', 10])
                self.stay_in_air.start()

    def print_measurement_variance(self):
        dr_paw, dr_pitch, dr_roll = self.drone.get_yaw(), self.drone.get_pitch(), self.drone.get_roll()
        # get roll, yaw and pitch from slam
        pass

    def auto_navigation(self):
        targets = [(x, x) for x in range(10, 20)]
        # targets = run_AStar()
        # targets.append(0, [0, 0])  # append [0, 0] to start, index 0
        targets = current_and_next(targets)
        for curr_coordinate, next_coordinate in targets:
            print(f'curr_coordinate = {curr_coordinate}, next_coordinate = {next_coordinate}')
            # https://stackoverflow.com/questions/20250396/converting-string-that-looks-like-a-list-into-a-real-list-python
            pose = self.app.request_from_cpp("Pose")
            self.move_to_location(*curr_coordinate, *next_coordinate)

    def move_to_location(self, x1, y1, x2, y2):
        """given initial x1,y1 position, drone moves to x2,y2"""
        x2, y2 = (x2 - x1, y2 - y1)  # new vector of movement
        x1, y1 = cos(radians(self.angle)), sin(radians(self.angle))  # previous vector of movement

        dot = x1 * x2 + y1 * y2  # dot product
        det = x1 * y2 - y1 * x2  # determinant
        angle = (atan2(det, dot) * (180 / pi))  # atan2(y, x) or atan2(sin, cos)
        self.angle = self.angle + angle
        print(f'[TELLO] rotating in angle of {angle}... ')
        """
        if angle >= 0:
            self.drone.rotate_counter_clockwise(angle)
        elif angle < 0:
            self.drone.rotate_clockwise(-1 * angle)
        """
        # clear previous/accidental authorization bit
        self.ack.clear()
        while True:
            print("[TELLO] [authorization request] Press s to move_forward")
            ack_accepted = self.ack.wait(timeout=30)  # wait for half a minute
            if ack_accepted:
                break
        d = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        """
        self.drone.move_forward(d)
        """
        print("[TELLO] moving... ")

    def wander_around(self):
        cm = 20
        #point = self.slam_pose
        point = 0.3, 0.3
        x, y = point[0] * 1000, point[1] * 1000
        local_map = {
                                        ('Y', cm): (x, y + cm),
            ('X', -cm): (x - cm, y),    ('X', 0): (x, y),           ('X', cm): (x + cm, y),
                                        ('Y', -cm): (x, y - cm)
        }
        cost_map = self.pointcloud.eval_cost_map(local_map)
        print("[TELLO][WANDER AROUND] cost map says: ", max(cost_map, key=cost_map.get))
        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        self.move(*max(cost_map, key=cost_map.get))
        self.scan_env()

    """
    Based on the tutorial:
    https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
    +Y                 (0,240)
    Y  (-320, 0)        (0,0)               (320,0)
    -Y                 (0,-240)
    -X                    X                    +X
    the hue range in
    opencv is 180, normally it is 360
    green_lower = (50, 50, 50)
    green_upper = (70, 255, 255)
    red_lower = (0, 50, 50)
    red_upper = (20, 255, 255)
    blue_lower = (110, 50, 50)
    upper_blue = (130, 255, 255)
    """

    def move_to_ball(self):
        tracker = color_tracker.ColorTracker(max_nb_of_objects=1, max_nb_of_points=20, debug=True)
        tracker.set_tracking_callback(tracker_callback)
        # Define your custom Lower and Upper HSV values
        tracking_thread = Thread(target=tracker.track, args=[self.app.factory(), [155, 103, 82], [178, 255, 255]])
        tracking_thread.start()

        # IS_RUNNING HAS TO STOP THIS INFINITE LOOP
        def get_tracked_object():
            while True:
                try:
                    x, y, x_w, y_h = tracker.tracked_objects[-1].last_bbox
                    width = (x_w - x)
                    height = (y_h - y)
                    area = width * height
                    center = x + width / 2, y + height / 2  # ABOUT THIS ONE IM NOT TOO SURE
                    return {'area': area, 'center': center}
                except IndexError:
                    self.wander_around()

        res = get_tracked_object()
        """
        while res is None:
            self.wander_around()  # the infinite loop at get_tracked_object ruined this
            res = get_tracked_object()
        """
        print(f"[TELLO][TRACKER] OBJECT FOUND; center:{res['center']}, area:{res['area']}")
        print("FRAME CENTER: ", (FRAME_SIZE[0] / 2, FRAME_SIZE[1] / 2))
        # repose the drone to face tracked ball
        xoffset = res['center'][0] - FRAME_SIZE[
            0] / 2  # if ball_center_x > FRAME_SIZE_x/2, you should move to the right
        yoffset = res['center'][1] - FRAME_SIZE[1] / 2
        self.move('X', xoffset)
        self.move('Z', yoffset)  # y axis in frame = z axis in real world

        print("determine in which direction to approach ball")
        # determine in which direction to approach ball
        self.move('Y', 20)
        res_forward = get_tracked_object()
        self.move('Y', -40)
        res_back = get_tracked_object()
        self.move('Y', 20)
        cm = 10 if res_forward['area'] >= res_back['area'] else -10

        print("move in that direction until ball is close enough")
        # move in that direction until ball is close enough
        self.move('Y', cm)
        res = get_tracked_object()
        while res['area'] < 700:
            self.move('Y', cm)
            res = get_tracked_object()

        tracker.stop_tracking()
        tracking_thread.join()


def tracker_callback(t: color_tracker.ColorTracker):
    cv2.imshow("debug", t.debug_frame)
    cv2.waitKey(1)