import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#wCam, hCam = 640, 480
#cap = cv2.VideoCapture(0)
#cap.set(3, hCam)
#cap.set(4, wCam)

class VolumeHandControl(htm.handDetector):
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5, frames=[], tello=None):
        htm.handDetector.__init__(self, mode, maxHands, detectionCon, trackCon, frames, tello)
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        self.volRange = self.volume.GetVolumeRange()

        self.minVol = self.volRange[0]
        self.maxVol = self.volRange[1]

    def run(self):
        pTime = cTime = 0
        for img in self.frames:
            img = img.copy()
            img = self.findHands(img)
            lmList = self.findPositions(img, draw=False)
            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2
                cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                length = math.hypot(x2 - x1, y2 - y1)
                if length < 50:#  close two fingers
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                vol = np.interp(length, [50, 230], [self.minVol, self.maxVol])
                print(length, vol)
                self.volume.SetMasterVolumeLevel(vol, None)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, "FPS: {0}".format(str(int(fps))), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)
            cv2.imshow('Image', img)
            cv2.waitKey(1)