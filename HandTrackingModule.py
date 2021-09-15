import cv2
import mediapipe as mp
import time
from threading import Lock


class handDetector(object):
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5, frames=[], tello=None):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.frames = frames
        self.tello = tello
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.lock = Lock()

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True, num=None):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw and num == None:
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            if draw and num != None:
                cv2.circle(img, (lmList[num][1], lmList[num][2]), 15, (0, 0, 255), cv2.FILLED)
        return lmList




    def run(self):
        pTime = cTime = 0
        #cap = cv2.VideoCapture(0)
        #detector = handDetector()
        for img in self.frames:
            img = img.copy()
            img = self.findHands(img)
            lmList = self.findPosition(img, num=8)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 3)

            cv2.imshow('Image', img)
            cv2.waitKey(1)


#if __name__ == "__main__":
#    main()