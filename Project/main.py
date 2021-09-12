import getpass
import os
import time
from threading import Event
import cv2
import numpy as np

from tello_control import TelloCV
from image_filters import init_control_gui, apply_hsv_filter, apply_edge_filter
from face_recognition import FaceRec
import utils
import pointcloud

import color_tracker

# INPUT_FORM: VIDEO/WEB-CAM/TELLO
INPUT_FORM = 'TELLO'
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
            's': lambda: self._tello.set_ack(),
        }
        self.commands_to_cpp = {
            "END": 1,
            "SAVEMAP": 2
        }
        self.queries_from_cpp = [
            "listPose",
            "isWall",
            "isMapSaved",
            "isSlamInitialized",
            "isTrackingLost"
        ]
        subproc_path = f"{os.getenv('HOME')}/ORB_SLAM2/Examples/Monocular/mono_tum" if RUN_ORB_SLAM else None
        super(App, self).__init__(subproc_path=subproc_path, commands_to_cpp=self.commands_to_cpp,
                                  queries_from_cpp=self.queries_from_cpp, controls=self.controls)

        self._tello = None
        self._cap = None
        self._input_video_path = input_video_path

        # self.stop_frames = Event()
        self.factory = utils.generators_factory(self.__framesGenerator(), size=10)

        if INPUT_FORM == 'TELLO':
            self._tello = TelloCV(self)

        self.init_cap()
        # ***DELETE THIS***
        if INPUT_FORM == 'WEB-CAM':
            self._tello = TelloCV(self)
    def init_cap(self):
        if INPUT_FORM == 'VIDEO':
            if self._input_video_path is None:
                print("ERROR: in input_video_path")
                exit(1)
            self._cap = cv2.VideoCapture(input_video_path)

        if INPUT_FORM == 'WEB-CAM':
            self._cap = cv2.VideoCapture(0)
            self._cap.open(0)
        """
        if INPUT_FORM == 'TELLO':
            self._cap = self._tello.drone.get_video_capture()
        """

    def __framesGenerator(self):
        if INPUT_FORM == 'TELLO':
            """
            for packet in self._tello.container.demux((self._tello.vid_stream,)):
                for frame in packet.decode():
                    # convert frame to opencv image
                    frame = cv2.cvtColor(np.array(
                        frame.to_image()), cv2.COLOR_RGB2BGR)
                    yield frame
            """
            frame_obj = self._tello.drone.get_frame_read()
            frame = frame_obj.frame
            frame = cv2.resize(frame, (640, 480))
            while frame is not None:
                yield frame
                if self.running:
                    frame = frame_obj.frame
                    frame = cv2.resize(frame, (640, 480))
                else:
                    break
                time.sleep(1 / 30)

        else:
            is_success, frame = self._cap.read()
            while frame is not None:
                # ORB_SLAM2 requires size of WIDTH 640, HEIGHT 480
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
                        # print("Passing frames to ORB_SLAM2...")
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
        self.running = False

        if self.subproc_path is not None:
            self.request_from_cpp("END")
            return_code = self.subproc.wait()
            print("C++ script finished with exit code: ", return_code)

        print("[APP][END] Ending App...")
        cv2.destroyAllWindows()
        if INPUT_FORM == 'TELLO':
            self._tello.end()

        # ***DELETE THIS***
        if INPUT_FORM == 'WEB-CAM':
            self._tello.end()

        elif INPUT_FORM == 'WEB-CAM' or INPUT_FORM == 'VIDEO':
            # self._cap.release()
            print("IM here")

        #self.stop_frames.set()
        self.can_stopListening.set()
        self.output_listener.join()
        print("output_listener joined.")
        self.command_sender.join()
        print("command_sender joined.")
        # no need to join key_listener
        # as listener is joined once we return False from on_press
        # see on_press for source link

if __name__ == "__main__":
    # utils.generate_training_faces(subject_name=getpass.getuser())
    # init_control_gui()

    if INPUT_FORM == 'VIDEO':
        input_video_path = '../data/Casino Royale Opening original.mp4'
    else:
        input_video_path = None

    App().run()
