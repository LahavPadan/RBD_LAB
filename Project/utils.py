import os
import sys
import cv2
import numpy as np

# CppCommunication imports
import subprocess
from threading import Thread, Lock, Event
from pynput import keyboard
import socket
import struct
import ast

# generators_factory imports
import collections

# calc_angle imports
from math import atan2, pi


PORT, HOST_IP = 8080, '127.0.0.1'
key = 4


class CppCommunication(object):
    def __init__(self, subproc_path=None, commands_to_cpp=None, queries_from_cpp=None, controls=None):

        self.lock_running = Lock()
        self._running = True
        # keyboard control
        self.controls = None
        self.key_listener = None
        self.keydown = False
        # list of possible queries to be fetched from cpp
        self.queries_from_cpp = queries_from_cpp
        self.can_stopListening = Event()
        # dict of current queries from cpp
        self.request_from_stdout = {}
        # commands dict
        self.commands_to_cpp = commands_to_cpp
        self.lock_send_command = Lock()
        self.last_command = None
        self.command_pending = Event()  # event initially false

        self.subproc_path = subproc_path
        if self.subproc_path is not None:
            self.subproc = subprocess.Popen(self.subproc_path,
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=-1,
                                            universal_newlines=True, shell=False)
            self.output_listener = Thread(target=self.__listen_to_output)
            self.output_listener.start()
            self.command_sender = Thread(target=self.__send_command_to_cpp)
            self.command_sender.start()

        self.controls = controls
        if self.controls is not None:
            self.key_listener = keyboard.Listener(on_press=self.on_press,
                                                  on_release=self.on_release)
            self.key_listener.start()

    @property
    def running(self):
        with self.lock_running:
            running = self._running
        return running

    @running.setter
    def running(self, value):
        with self.lock_running:
            self._running = value

    def __listen_to_output(self):
        """
        Wrong initialization, reseting...
        System Reseting
        Reseting Local Mapper... done
        Reseting Loop Closing... done
        Reseting Database... done
        New Map created with
        """
        while True:
            self.lock_running.acquire()
            if not self._running:
                self.lock_running.release()
                break
            self.lock_running.release()
            nextline = self.subproc.stdout.readline()
            if nextline == '' and self.subproc.poll() is not None:
                break
            """
            # https://stackoverflow.com/questions/26598322/find-a-list-of-patterns-in-a-list-of-string-in-python
            if not any(q in nextline for q in self.queries_from_cpp):
                sys.stdout.write(nextline)
                sys.stdout.flush()
            """
            # https://stackoverflow.com/questions/11941817/how-to-avoid-runtimeerror-dictionary-changed-size-during-iteration-error
            for query in list(self.request_from_stdout):
                if query in nextline:
                    try:
                        retrieved, retrieval = self.request_from_stdout[query]
                    except KeyError:
                        break
                    retrieval = nextline.split(query, 1)[1].rstrip()
                    self.request_from_stdout[query] = (retrieved, retrieval)
                    retrieved.set()
        # cleaning up
        print("Cleaning up requests1")
        while not self.can_stopListening.is_set():
            # print("Cleaning up requests2")
            # release all awaiting requests
            for query in list(self.request_from_stdout):
                try:
                    retrieved, retrieval = self.request_from_stdout[query]
                except KeyError:
                    break
                if "is" in query:
                    retrieval = "0"  # equivalent to printing false in cpp
                elif "list" in query:
                    retrieval = "[]"
                else:
                    retrieval = "None"
                self.request_from_stdout[query] = (retrieved, retrieval)
                retrieved.set()

    def __send_command_to_cpp(self):
        received_end = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # https://stackoverflow.com/questions/6380057/python-binding-socket-address-already-in-use
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST_IP, PORT))
            s.listen()
            print("starting to listen")
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while not received_end:
                    self.command_pending.wait()
                    with self.lock_send_command:
                        if self.last_command == 'END':
                            received_end = True
                        # send command
                        d = struct.pack('I', self.commands_to_cpp[self.last_command])
                        conn.sendall(d)
                        # clear command pending event bit
                        self.command_pending.clear()

    def request_from_cpp(self, request):
        """API function to send requests"""
        if request in self.commands_to_cpp.keys():
            with self.lock_send_command:
                self.last_command = request
                self.command_pending.set()
        elif request in self.queries_from_cpp:
            query = request
            retrieved = Event()
            self.request_from_stdout[query] = (retrieved, None)
            retrieved.wait()
            retrieved.clear()
            retrieval = self.request_from_stdout[query][1]
            del self.request_from_stdout[query]
            retrieval = ast.literal_eval(retrieval)
            return retrieval

    def on_press(self, keyname):
        """handler for keyboard listener"""
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            print('+' + keyname)
            if keyname == 'Key.esc':
                self.end()
                # return False from a callback to stop the listener.
                # source: https://pynput.readthedocs.io/en/latest/keyboard.html
                return False
            elif keyname in self.controls:
                self.controls[keyname]()
        except AttributeError:
            print('special key {0} pressed'.format(keyname))

    def on_release(self, keyname):
        """Reset on key up from keyboard listener"""
        self.keydown = False

    def end(self):
        if self.subproc_path is not None:
            self.request_from_cpp("END")
            return_code = self.subproc.wait()
            print("C++ script finished with exit code: ", return_code)
            self.running = False
        # no need to join key_listener
        # as listener is joined once we return False from on_press
        # see on_press for source link

########################################################################################################################

# https://stackoverflow.com/questions/55674212/shared-python-generator/55762535
def generators_factory(iterable, size=None):
    it = iter(iterable)
    deques = []
    already_gone = collections.deque(maxlen=size)
    lock = Lock()

    def new_generator():
        new_deque = collections.deque()
        new_deque.extend(already_gone)
        deques.append(new_deque)

        def gen(mydeque):
            while True:
                if not mydeque:  # when the local deque is empty
                    with lock:
                        try:
                            newval = next(it)  # fetch a new value and
                            already_gone.append(newval)
                            for d in deques:  # load it to all the deques
                                d.append(newval)
                        except StopIteration:
                            break
                yield mydeque.popleft()

        return gen(new_deque)

    return new_generator


########################################################################################################################


TRAINING_IMAGE_SIZE = (640, 480)

"""from https://www.bytefish.de/blog/validating_algorithms.html"""


# USAGE: images,labels = read_images("/path/to/some/folder")
#        images, labels, names = read_images("/path/to/some/folder", ret_folder_names=True)


def read_images(path, image_size=None, ret_folder_names=False):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    Args:
        path: Path to a folder with subfolders representing the subjects.
        image_size: Size to which Resizes
        ret_folder_names: whether or not to return the image as well as her parent folder name
    Returns:
        A list [images, identifiers, names (-optional)]
            images: The images, which is a Python list of numpy arrays.
            identifiers: The corresponding identifiers (the unique number of the subject, person) in a Python list.
            names (-optional): The corresponding parent folder names in a Python list.
    """
    names = []
    current_identifier = 0
    images, identifiers = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            if ret_folder_names:
                names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in sorted(os.listdir(subject_path)):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if image_size is not None:  # resize to given size (if given)
                        im = cv2.resize(im, image_size)
                    images.append(np.asarray(im, dtype=np.uint8))
                    identifiers.append(current_identifier)
                    print(filename)
                except IOError as err:
                    print("I/O error({0}): {1}".format(err.errno, err.strerror))
                except Exception:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            current_identifier = current_identifier + 1

    images = np.asarray(images, np.uint8)
    identifiers = np.asarray(identifiers)
    if ret_folder_names:
        return images, identifiers, names
    return images, identifiers


########################################################################################################################


"""from: https://github.com/PacktPublishing/Learning-OpenCV-4-Computer-Vision-with-Python-Third-Edition/blob
/master/chapter05/generate_training_faces.py"""


def generate_training_faces(subject_name):
    output_folder = f'../subjects_faces/{subject_name}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(0)
    count = 0
    while (cv2.waitKey(1) == -1):
        success, frame = camera.read()
        if success:
            # frame = cv2.resize(frame, TRAINING_IMAGE_SIZE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.3, 5, minSize=(120, 120))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                face_filename = '%s/%d.pgm' % (output_folder, count)
                cv2.imwrite(face_filename, face_img)
                count += 1
            cv2.imshow('Capturing Faces...', frame)


########################################################################################################################

def tri_centroid(tri: np.array) -> np.array:
    (x1, y1), (x2, y2), (x3, y3) = tri
    return np.array([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3])


def find_relative_3D_space(Y: np.array) -> np.array:
    rotation_mat = np.array([[0, 1],
                             [-1, 0]])
    rotated_vec = np.dot(rotation_mat, Y[:-1])
    X = np.append(rotated_vec, 0)  # make it a 3D vector
    Z = np.array([0, 0, 1])
    return X, Y, Z

def round_up(num):
    threshold = 13
    rounded = num
    if num <= 20:
        if threshold <= num <= 20:
            rounded = 20
        else:
            rounded = 0
    return rounded
