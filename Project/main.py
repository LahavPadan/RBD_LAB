import getpass
import os

import cv2
import numpy as np

from tello_control import TelloCV
from image_filters import init_control_gui, apply_hsv_filter, apply_edge_filter
from face_recognition import FaceRec
import utils
import pointcloud


# INPUT_FORM: VIDEO/WEB-CAM/TELLO
INPUT_FORM = 'WEB-CAM'
RUN_ORB_SLAM = True
OUTPUT_FORM = 'WINDOW'


# face_rec = FaceRec()
def processFrame(frame):
    output_frames = [
        frame,
        # face_rec.recognize(frame),
        # apply_edge_filter(frame),
        # apply_hsv_filter(frame)
    ]
    return frame, output_frames


class App(utils.CppCommunication):
    def __init__(self):
        self.controls = {
            's': lambda: self._tello.ack.set(),
        }
        self.commands_to_cpp = {
            "END": 1,
            "SAVEMAP": 2
        }
        self.queries_from_cpp = [
            "listPose",
            "isWall",
            "isMapSaved",
            "isSlamInitialized"
        ]
        subproc_path = f"{os.getenv('HOME')}/ORB_SLAM2/Examples/Monocular/mono_tum" if RUN_ORB_SLAM else None
        super(App, self).__init__(subproc_path=subproc_path, commands_to_cpp=self.commands_to_cpp,
                                  queries_from_cpp=self.queries_from_cpp, controls=self.controls)

        self._tello = None
        self._cap = None
        self._input_video_path = input_video_path
        self.init_cap()
        self.factory = utils.generators_factory(self.__framesGenerator(), size=10)

        if INPUT_FORM == 'TELLO':
            self._tello = TelloCV(self)

        # DELETE THIS
        if INPUT_FORM == 'WEB-CAM':
            self._tello = TelloCV(self)

    def on_press(self, keyname):
        """Override CppCommunication on_press function"""
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            print('+' + keyname)
            if keyname == 'Key.esc':
                self.end()
                return False
            elif keyname in self.controls:
                self.controls[keyname]()
        except AttributeError:
            print('special key {0} pressed'.format(keyname))

    def init_cap(self):
        if INPUT_FORM == 'VIDEO':
            if self._input_video_path is None:
                print("ERROR: in input_video_path")
                exit(1)
            self._cap = cv2.VideoCapture(input_video_path)

        if INPUT_FORM == 'WEB-CAM':
            self._cap = cv2.VideoCapture(0)
            self._cap.open(0)

        if INPUT_FORM == 'TELLO':
            self._cap = self._tello.drone.get_video_capture()

    # https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    def __framesGenerator(self):
        is_success, frame = self._cap.read()
        while frame is not None:
            # orb_slam requires size of WIDTH 640, HEIGHT 480
            frame = cv2.resize(frame, (640, 480))
            yield frame
            if self.running:
                is_success, frame = self._cap.read()
            else:
                break

    def run(self):
        if INPUT_FORM not in ['WEB-CAM', 'VIDEO', 'TELLO']:
            print("ERROR: INPUT FORM not defined properly")
            self.end()

        print(f"[{INPUT_FORM}] [DISPLAYING] existing {INPUT_FORM.lower()}. Press 'Esc' to stop EXECUTION")

        if not RUN_ORB_SLAM:
            cv2.namedWindow(f'Processed Feed of {INPUT_FORM.lower()}')  # create window to display processed feed
        frames = self.factory()
        for frame in frames:
            output_frames = processFrame(frame)
            if OUTPUT_FORM == 'WINDOW':
                if RUN_ORB_SLAM:
                    # https://stackoverflow.com/questions/21689365/python-3-typeerror-must-be-str-not-bytes-with-sys-stdout-write
                    try:
                        self.subproc.stdin.buffer.write(output_frames[0].tobytes())
                    except BrokenPipeError:
                        pass
                else:
                    # https://stackoverflow.com/questions/21906382/embedding-windows-in-gui
                    img_stack = np.hstack(output_frames[1])
                    cv2.imshow(f'Processed Feed of {INPUT_FORM.lower()}', img_stack)
                    if cv2.waitKey(20) == 27:  # quit if 'Esc' (escape) is pressed
                        break

    def end(self):
        super(App, self).end()
        print("Im in App's end")
        cv2.destroyAllWindows()
        if INPUT_FORM == 'TELLO':
            self._tello.end()
        elif INPUT_FORM == 'WEB-CAM' or INPUT_FORM == 'VIDEO':
            # self._cap.release()
            print("IM here")
        self.output_listener.join()
        self.command_sender.join()


if __name__ == "__main__":
    # utils.generate_training_faces(subject_name=getpass.getuser())
    # init_control_gui()

    if INPUT_FORM == 'VIDEO':
        input_video_path = '../data/Casino Royale Opening original.mp4'
    else:
        input_video_path = None

    App().run()

