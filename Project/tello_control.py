import djitellopy
from threading import Thread, Event
from time import sleep
import numpy as np
from math import cos, sin, radians
from Project.pointcloud import PointCloud
from Project.utils import find_relative_3D_space, round_up
import cv2
import color_tracker
from queue import Queue, Empty

FRAME_SIZE = (640, 480)


class TelloCV(object):
    """
     attributes:
            speed: movement speed of drone, both rotating and moving. Set internally once _init_drone is called.
            q: queue of actions specified by djitellopy's API
            momentum_vec: direction vector of drone
            X, Y, Z: floats - relative X,Y,Z coordinates. that is, Y axis is always set to the direction vector of drone
            slam_pose: np.array - 3D coordinate of drone in ORB_SLAM2's coordinate space
            ack_accepted: threading.Event Object - signify acknowledgement from user after drone sends an authorization request
    """
    def __init__(self, app):
        """
        @param app: reference to main app object. to access CppCommunication's API
        """
        self.app = app
        self.drone = djitellopy.Tello()
        self.speed = 10  # 10 cm/sec
        self._angle = 90  # drone facing forward = 90 degrees on the unit circle

        self.q = Queue()

        self.momentum_vec = np.array([0, 1, 0])  # 3D vector: (0, 1, 0)
        self.X, self.Y, self.Z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])

        self.slam_pose: np.array = np.array([])
        self.pointcloud = None

        self.ack_accepted = Event()

        self.move_to_obj_thread = None

        # IM CREATING TON OF THREADS AND JOINING NONE
        Thread(target=self._init_drone).start()

    def submit_action(self, action_str, args: list, blocking=False):
        """
        @summary: submit djitellopy actions to queue for execution

        @param action_str: djitellopy action specified by djitellopy's API
        @param args: arguments for action
        @param blocking: whether it is asynchronous/synchronous request.
        If synchronous, caller will be stalled util action is handled
        @return: None
        """
        done_indicator = Event()
        self.q.put( {'action_str': action_str, 'args': args, 'done_indicator': done_indicator} )
        if blocking:
            done_indicator.wait()

    def worker(self):
        """
        @summary: responsiable for executing actions from queue as well as keeping the drone "busy"
        so it will stay in air and not auto-land
        @return: None
        """
        i = 0
        staying_in_air = ['up', 'down']
        print("worker in loop")
        while self.app.running:
            try:
                item = self.q.get(block=False)
            except Exception:
                # for some reason, sometimes queue.Empty doesn't catch that exception
                print("QUEUE IS EMPTY")
                item = None
                self.drone.move(staying_in_air[i], 20)
                i = (i + 1) % 2
            else:
                print(f'item = {item}')
                try:
                    getattr(self.drone, item['action_str'])(*item['args'])
                except AttributeError:
                    print("Invalid argument: Cannot execute passed function")
                    continue
                except KeyError:
                    print("Please submit to queue via submit_action method")
                    continue

            # wait for the action to be executed
            sleep(3)
            if isinstance(item, dict):
                print("Setting item")
                item['done_indicator'].set()

    def _init_drone(self):
        """
        @summary: connect to drone, takeoff, start searching for obj in move_to_obj_thread.
        initialize ORB_SLAM2, and a pointcloud object
        @return: None
        """
        self.drone.connect()
        self.drone.streamon()
        print("Battery percentage: ", self.drone.get_battery())
        self.drone.set_speed(self.speed)
        self.drone.takeoff()

        # start listening to commands
        Thread(target=self.worker).start()
        # Drone Camera is open. wait for ORB_SLAM2 to initialize

        while not self.app.request_from_cpp("isSlamInitialized") and self.app.running:
            # Assuming that Esc will be pressed after ORB_SLAM2 - opens - (request_from_cpp is blocking operation)
            print("[TELLO][INIT_DRONE] Waiting for ORB_SLAM2 to initialize...")
        print("[TELLO][INIT_DRONE] ORB_SLAM2 initialized.")
        
        # from this point onwards, ORB_SLAM2 is initialized (but can lose localization)
        self.slam_pose = np.array(self.app.request_from_cpp("listPose"))
        self.pointcloud = PointCloud(self.app)

        self.move_to_obj_thread = Thread(target=self.move_to_obj)
        self.move_to_obj_thread.start()

        # complete 360 degrees without exit clause
        self.scan_env(blocking=False)


    def end(self):
        print("[TELLO][END] Ending TelloCV...")
        # release any move authorization request
        self.ack_accepted.set()  # the bit stays turned on

        self.pointcloud.end()

        print("About to join move_to_obj")
        self.move_to_obj_thread.join()
        print("move_to_obj joined.")
        self.drone.land()
        self.drone.end()

    def _authorization_request(self, direction, cm):
        """
        @summary: block until user authorizes movement by a key stroke
        @param direction: movement direction
        @param cm: movement magnitude in cm
        @return: None
        """
        if self.app.running:
            # clear previous/accidental authorization bit
            self.ack_accepted.clear()
        while self.app.running:
            print(f'[TELLO][authorization request] {direction}, {cm} cm. Press s to move')
            accepted = self.ack_accepted.wait(15)  # wait about a quarter of a minute
            if accepted:
                break

    def __update_movement_vector(self, diff_angle):
        """
        used internally by scan_env to keep drone's positional attributes intact
        @param diff_angle: how the angle has changed from previous angle
        @return: None
        """
        # self._angle is only used inside cosine and sine, thus, no problem that self._angle will become
        # arbitrarily large
        self._angle = self._angle + diff_angle
        self.momentum_vec = np.array([cos(radians(self._angle)), sin(radians(self._angle))])
        self.momentum_vec = np.append(self.momentum_vec, 0)  # make it a 3D vector by adding dummy z coordinate
        self.X, self.Y, self.Z = find_relative_3D_space(self.momentum_vec)

    def scan_env(self, pause_sec=0, exit_clause=(lambda: False), blocking=True):
        """
        @summary: rotate 360 degress unless exit_clause is met at some point
        @param pause_sec: time to wait between individual rotation steps in scan
        @param exit_clause: the scan will be stopped mid way through if exit clause is met
        @param blocking: whether it is asynchronous/synchronous request.
        If synchronous, caller will be stalled util action is handled
        """
        print("[TELLO][SCAN_ENV] scanning environment...")
        degree_per_iteration = 40
        for _ in range(0, 360, degree_per_iteration):

            if exit_clause() or not self.app.running:
                break

            self.__update_movement_vector(diff_angle=degree_per_iteration)
            self.submit_action(action_str="rotate_clockwise", args=[degree_per_iteration], blocking=blocking)

            # wait for ORB_SLAM2 to initialize and scan
            if pause_sec:
                sleep(pause_sec)

        print("[TELLO][SCAN_ENV] done.")

    def move(self, vector: np.array, cm, blocking=True) -> bool:
        """
        :param vector: (3D vector, normalized) direction to move.
        :param cm: magnitude of vector in cm.
        :return: True if and only if drone actually traveled.
        """
        vector = vector * cm  # it also switches vector-entries-sign if cm < 0
        cm = abs(cm)

        X_movement = np.dot(vector, self.X) * self.X
        # redundant, the norm is the abs(dot)
        X_norm = round_up(np.linalg.norm(X_movement))  # round above 20 if close enough (can't move less than 20 cm)
        Y_movement = np.dot(vector, self.Y) * self.Y
        Y_norm = round_up(np.linalg.norm(Y_movement))
        Z_movement = np.dot(vector, self.Z) * self.Z
        Z_norm = round_up(np.linalg.norm(Z_movement))

        X_movement = 'right' if X_movement[0]/self.X[0] > 0 else 'left'
        Y_movement = 'forward' if Y_movement[1]/self.Y[1] > 0 else 'back'
        Z_movement = 'up' if Z_movement[2]/self.Z[2] > 0 else 'down'

        while self.app.request_from_cpp("isWall"):
            pass

        for axis in ["X_", "Y_", "Z_"]:
            # https://stackoverflow.com/questions/9437726/how-to-get-the-value-of-a-variable-given-its-name-in-a-string
            movement = locals()[axis + "movement"]
            norm = locals()[axis + "norm"]
            print(f'axis: {axis}, norm: {norm}')
            if norm > 0:
                self._authorization_request(movement, int(norm))
                if not self.app.running:
                    break
                self.submit_action(action_str="move_"+movement, args=[int(norm)], blocking=blocking)
                print("Aware that action is done")

        # update pose variable
        self.slam_pose = np.array(self.app.request_from_cpp("listPose"))
        print("self.slam_pose is ", self.slam_pose)
        # for the future: maybe track drone pose as given by commands throughout the program
        return True

    def wander_around(self, exit_clause=(lambda: False)):
        # point = 0.3, 0.3
        while not (exit_clause() and not self.app.request_from_cpp("isWall")) and self.app.running:
            print("[TELLO][WANDER_AROUND] searching where to go...")
            cm = 40
            self.slam_pose = np.array(self.app.request_from_cpp("listPose"))
            moved = self.move(self.pointcloud.eval_movement_vector(self.slam_pose, self.momentum_vec), cm)
            self.scan_env(pause_sec=3, exit_clause=exit_clause)

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

    def move_to_obj(self):
        def get_tracked_object():
            obj = None

            def is_object_found():
                nonlocal obj
                try:
                    x, y, x_w, y_h = tracker.tracked_objects[-1].last_bbox
                    width = (x_w - x)
                    height = (y_h - y)
                    area = width * height
                    center = x + width / 2, y + height / 2  #ITS THE CENTER IN FRAME NOT NECESSARILY IN REAL WORLD
                    obj = {'area': area, 'center': center}
                    return True
                except (AttributeError, TypeError, IndexError):
                    pass
                return False
            self.wander_around(exit_clause=is_object_found)
            if not self.app.running:
                print("Returning that dummy object found dict")
                # returning dummy object_found dict
                obj = {'area': 0, 'center': (FRAME_SIZE[0] / 2, FRAME_SIZE[1] / 2)}

            return obj

        tracker = color_tracker.ColorTracker(max_nb_of_objects=1, max_nb_of_points=20, debug=True)
        tracker.set_tracking_callback(self.tracker_callback)
        # Define your custom Lower and Upper HSV values
        # [148, 192, 0], [255, 255, 147]: more like the color of sweatshirt
        tracking_thread = Thread(target=tracker.track, args=[self.app.factory(), [155, 103, 82], [178, 255, 255]],
                                 kwargs={'min_contour_area': 1000, 'max_track_point_distance': 1000})
        tracking_thread.start()

        # wait for tracking to start
        sleep(40)

        res = get_tracked_object()
        print(f"[TELLO][TRACKER] OBJECT FOUND; center:{res['center']}, area:{res['area']}."
              f" Frame-center is: {(FRAME_SIZE[0] / 2, FRAME_SIZE[1] / 2)}")
        # repose the drone to face tracked object
        xoffset = res['center'][0] - (FRAME_SIZE[0] / 2) # if object_center_x > FRAME_SIZE_x/2, you should move to the right
        yoffset = res['center'][1] - (FRAME_SIZE[1] / 2)
        # NEED TO TWEAK THE CM - OFFSET RATIO
        print("First call")
        self.move(self.X, xoffset/5)
        print("Second call")
        self.move(self.Z, yoffset/5)  # y axis in frame = z axis in real world

        print("move forward until object is close enough")
        # move forward until object is close enough
        cm = 40
        res = get_tracked_object()
        while res['area'] < 60000 and self.app.running:
            self.move(self.momentum_vec, cm)
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
