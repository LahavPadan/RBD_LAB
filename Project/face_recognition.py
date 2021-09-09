"""OpenCV drawing API reference: https://docs.opencv.org/3.4/dc/da5/tutorial_py_drawing_functions.html"""

import cv2
import utils


class Tracker:
    """
    Based on the tutorial:
    https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

    A basic color tracker, it will look for colors in a range and
    return the center and radius of ball that matches this color
    """

    def __init__(self, width, height, color_lower, color_upper):
        self.color_lower = color_lower
        self.color_upper = color_upper
        self.midx = int(width / 2)
        self.midy = int(height / 2)
        self.xoffset = 0
        self.yoffset = 0


    def draw_arrows(self, frame):
        """Show the direction vector output in the cv2 window"""
        #cv2.putText(frame,"Color:", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
        cv2.arrowedLine(frame, (self.midx, self.midy),
                        (self.midx + self.xoffset, self.midy - self.yoffset),
                        (0, 0, 255), 5)
        return frame


    def track(self, frame):
        """Simple HSV color space tracking"""
        # resize the frame, blur it, and convert it to the HSV
        # color space
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                """
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                """
                self.xoffset = int(center[0] - self.midx)
                self.yoffset = int(self.midy - center[1])

                cv2.imshow("Frame", self.draw_arrows(frame))
                key = cv2.waitKey(1) & 0xFF

                return {'center': center, 'radius': radius}
            """
            else:
                self.xoffset = 0
                self.yoffset = 0
            """
        else:
            """
            self.xoffset = 0
            self.yoffset = 0
            """
            return None



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


