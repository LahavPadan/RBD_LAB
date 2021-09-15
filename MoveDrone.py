import cv2
import mediapipe as mp
import time
import math
from threading import Lock
import HandTrackingModule as htm

class MoveDrone(htm.handDetector):
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5, frames=[], tello=None):
        htm.handDetector.__init__(self, mode, maxHands, detectionCon, trackCon, frames, tello)

    def run(self):
        pTime = cTime = 0
        i = 0
        interval = 200
        #cap = cv2.VideoCapture(0)
        #detector = handDetector()
        pos1 = pos2 = [None, None]
        dx = dy = None
        for img in self.frames:
            img = img.copy()
            img = self.findHands(img)
            lmList = self.findPosition(img, draw=False)
            avgx = 0
            for j in lmList:
                avgx += j[1]
            avgx /= 21
            avgy = 0
            for j in lmList:
                avgy += j[2]
            avgy /= 21
            #print(avgx, avgy)
            cv2.circle(img, (int(avgx), int(avgy)), 15, (0, 255, 255), cv2.FILLED)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)) + ", " + str(int(avgx)) + " " + str(int(avgy)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)
            if len(lmList) > 0 and (i == 0 or i == interval//2):
                if i == 0:
                    pos1 = [avgx, avgy]
                elif i == interval//2:
                    pos2 = [avgx, avgy]
                    if pos2[0] and pos1[0]:
                        dx = pos2[0] - pos1[0]
                    if pos2[1] and pos1[1]:
                        dy = pos2[1] - pos1[1]
                    if dx and dy:
                        if abs(dx) > abs(dy):
                            if dx > 0:
                                print("Go left")
                                self.tello.move_left(10)
                            else:
                                print("Go right")
                                self.tello.move_right(10)
                        else:
                            if dy > 0:
                                print("Go down")
                                self.tello.move_down(10)
                            else:
                                print("Go up")
                                self.tello.move_up(10)
                print(i, pos1, pos2, dx, dy)
            i += 1
            if i == interval:
                i = 0
            cv2.imshow('Image', img)
            cv2.waitKey(1)

