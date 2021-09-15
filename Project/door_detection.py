import cv2
import numpy as np
from threading import Lock

class DoorDetector(object):
    def __init__(self, frames):
        self.frames = frames
        self.whT = 320
        self.confThreshold = 0.5
        self.nmsThreshold = 0.3
        classesFile = 'door.names'
        self.classNames = []
        with open(classesFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')
        print(f'class names {self.classNames}')

        self.modelConfiguration = '/home/daniel/RBD_LAB/Project/yolo-door.cfg'
        self.modelWeights = '/home/daniel/RBD_LAB/Project/yolo-door.weights'

        self.net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self._last_bbox = None
        self.last_bbox_lock = Lock()


    def findObjects(self, outputs, img):
        def object_area(box):
            x, y, w, h = box[0], box[1], box[2], box[3]
            return w * h

        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int(det[0] * wT - w / 2), int(det[1] * hT - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))
        print(len(bbox))
        indices = cv2.dnn.NMSBoxes(bbox, confs, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f'{self.classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

        if len(bbox):
            with self.last_bbox_lock:
                self._last_bbox = max(bbox, key=object_area)

    @property
    def last_bbox(self):
        with self.last_bbox_lock:
            bbox = self._last_bbox.copy()
        return bbox

    def run(self):
        for img in self.frames:
            print("Door detection in loop")
            img = img.copy()
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.whT, self.whT), [0, 0, 0], 1, crop=False)
            self.net.setInput(blob)

            layerNames = self.net.getLayerNames()
            outputNames = [layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

            outputs = self.net.forward(outputNames)
            self.findObjects(outputs, img)

            cv2.imshow('Image', img)
            cv2.waitKey(1)
