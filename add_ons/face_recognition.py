"""OpenCV drawing API reference: https://docs.opencv.org/3.4/dc/da5/tutorial_py_drawing_functions.html"""

import cv2
import utils

class FaceRec:
    def __init__(self, detect_rect_color=(255, 255, 0)):
        training_images, training_identifiers, self._names = utils.read_images("../subjects_faces",
                                                                               image_size=utils.TRAINING_IMAGE_SIZE,
                                                                               ret_folder_names=True)
        self._model = cv2.face.EigenFaceRecognizer_create()
        self._model.train(training_images, training_identifiers)
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self._detect_rect_color = detect_rect_color

    def recognize(self, frame):
        frame = cv2.resize(frame, utils.TRAINING_IMAGE_SIZE)  # rescale to the size of a standard training image
        faces = self._face_cascade.detectMultiScale(frame, 1.3, 5)
        if faces is None:
            print("No face was recognized.")
        for (x, y, w, h) in faces:
            # cv2.rectangle(:image, :top-left coordinates, :bottom-right coordinates, :color,
            # :thickness(of line), :lineType)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), 2)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            """Crop into the region of the detection rectangle - result is stored into frame_face).
            Then, rescale gray_frame to the size of a standard training image"""
            frame_face = gray_frame[x:x + w, y:y + h]  # crop into detection rectangle
            if frame_face.size == 0:
                # Maybe the face is at the image edge.
                # Skip it.
                continue

            # rescale to the size of a standard training image
            frame_face = cv2.resize(frame_face, utils.TRAINING_IMAGE_SIZE)
            identifier, confidence = self._model.predict(frame_face)
            """Notice: text is put on top of the not processed frame"""
            text = f'{self._names[identifier]}; confidence: {confidence}'
            # cv2.putText(:img, :text, :bottom-left, :font-type, :font-scale, :color, :thickness, :line-type)
            cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        return frame


