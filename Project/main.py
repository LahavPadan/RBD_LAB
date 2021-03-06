import os
import time
import cv2
import numpy as np
from Project.tello_control import TelloCV
import Project.utils as utils

# INPUT_FORM: VIDEO/WEB-CAM/TELLO
INPUT_FORM = 'WEB-CAM'
RUN_ORB_SLAM = False
OUTPUT_FORM = 'WINDOW'


class App(utils.CppCommunication):
    """
    attributes:
            supported by subclassing CppCommunication:
                controls:dict - keys with specified function callback when they are pressed
                commands_to_cpp:dict - commands supported by the cpp code
                queries_from_cpp:list - data which can be fetched from stdout of cpp code
        factory:function_pointer - invoking factory returns an iterator of frames
    """
    def __init__(self):
        self.controls = {
            's': lambda: self._tello.ack_accepted.set()
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
            """instance a TelloCV object"""
            self._tello = TelloCV(self)
        else:
            """instance the capture suiting to INPUT_FORM"""
            self.init_cap()


    def init_cap(self):
        """
        initialize self._cap
        """
        if INPUT_FORM == 'VIDEO':
            if self._input_video_path is None:
                print("ERROR: in input_video_path")
                exit(1)
            self._cap = cv2.VideoCapture(input_video_path)

        if INPUT_FORM == 'WEB-CAM':
            self._cap = cv2.VideoCapture(0)
            self._cap.open(0)

    def __framesGenerator(self):
        """
        to be used internally by self.factory to fetch the next frame
        """
        if INPUT_FORM == 'TELLO':
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

    @staticmethod
    def processFrame(frame):
        """
        output from additional processing can be added
        all output_frames are displayed (simultaneously) if RUN_ORB_SLAM is False
        """
        output_frames = [
            frame,
        ]
        return frame, output_frames

    def run(self):
        if INPUT_FORM not in ['WEB-CAM', 'VIDEO', 'TELLO']:
            print("ERROR: INPUT FORM not defined properly")
            self.end()

        print(f"[{INPUT_FORM}] [DISPLAYING] existing {INPUT_FORM.lower()}. Press 'Esc' to stop EXECUTION")

        if not RUN_ORB_SLAM:
            cv2.namedWindow(f'Processed Feed of {INPUT_FORM.lower()}')  # create window to display processed feed
        frames = self.factory()
        for frame in frames:
            output_frames = self.processFrame(frame)
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
        """
        end method overridden in CppCommunication. Releases resources.
        Invoked upon handling ESC key stroke callback
        """
        self.running = False

        if self.subproc_path is not None:
            self.request_from_cpp("END")
            return_code = self.subproc.wait()
            print("C++ script finished with exit code: ", return_code)

        print("[APP][END] Ending App...")
        cv2.destroyAllWindows()
        if INPUT_FORM == 'TELLO':
            self._tello.end()

        self.can_stopListening.set()
        self.output_listener.join()
        print("output_listener joined.")
        self.command_sender.join()
        print("command_sender joined.")
        # no need to join key_listener
        # as listener is joined once we return False from on_press
        # see on_press for source link


if __name__ == "__main__":

    if INPUT_FORM == 'VIDEO':
        input_video_path = '../data/Casino Royale Opening original.mp4'
    else:
        input_video_path = None

    App().run()
