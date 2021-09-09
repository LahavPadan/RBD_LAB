"""taken from https://github.com/learncodebygaming/opencv_tutorials/tree/master/007_canny_edge"""

import cv2
import numpy as np

TRACKBAR_WINDOW = "Filter Trackbars"


# custom data structure to hold the state of an HSV filter
class HsvFilter:

    def __init__(self, hMin=None, sMin=None, vMin=None, hMax=None, sMax=None, vMax=None,
                 sAdd=None, sSub=None, vAdd=None, vSub=None):
        self.hMin = hMin
        self.sMin = sMin
        self.vMin = vMin
        self.hMax = hMax
        self.sMax = sMax
        self.vMax = vMax
        self.sAdd = sAdd
        self.sSub = sSub
        self.vAdd = vAdd
        self.vSub = vSub


# custom data structure to hold the state of a Canny edge filter
class EdgeFilter:

    def __init__(self, kernelSize=None, erodeIter=None, dilateIter=None, canny1=None,
                 canny2=None):
        self.kernelSize = kernelSize
        self.erodeIter = erodeIter
        self.dilateIter = dilateIter
        self.canny1 = canny1
        self.canny2 = canny2


# create gui window with controls for adjusting arguments in real-time
def init_control_gui():
    cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TRACKBAR_WINDOW, 350, 700)

    # required callback. we'll be using getTrackbarPos() to do lookups
    # instead of using the callback.
    def nothing(position):
        pass

    # create trackbars for bracketing.
    # OpenCV scale for HSV is H: 0-179, S: 0-255, V: 0-255
    cv2.createTrackbar('HMin', TRACKBAR_WINDOW, 0, 179, nothing)
    cv2.createTrackbar('SMin', TRACKBAR_WINDOW, 0, 255, nothing)
    cv2.createTrackbar('VMin', TRACKBAR_WINDOW, 0, 255, nothing)
    cv2.createTrackbar('HMax', TRACKBAR_WINDOW, 0, 179, nothing)
    cv2.createTrackbar('SMax', TRACKBAR_WINDOW, 0, 255, nothing)
    cv2.createTrackbar('VMax', TRACKBAR_WINDOW, 0, 255, nothing)
    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', TRACKBAR_WINDOW, 179)
    cv2.setTrackbarPos('SMax', TRACKBAR_WINDOW, 255)
    cv2.setTrackbarPos('VMax', TRACKBAR_WINDOW, 255)

    # trackbars for increasing/decreasing saturation and value
    cv2.createTrackbar('SAdd', TRACKBAR_WINDOW, 0, 255, nothing)
    cv2.createTrackbar('SSub', TRACKBAR_WINDOW, 0, 255, nothing)
    cv2.createTrackbar('VAdd', TRACKBAR_WINDOW, 0, 255, nothing)
    cv2.createTrackbar('VSub', TRACKBAR_WINDOW, 0, 255, nothing)

    # trackbars for edge creation
    cv2.createTrackbar('KernelSize', TRACKBAR_WINDOW, 1, 30, nothing)
    cv2.createTrackbar('ErodeIter', TRACKBAR_WINDOW, 1, 5, nothing)
    cv2.createTrackbar('DilateIter', TRACKBAR_WINDOW, 1, 5, nothing)
    cv2.createTrackbar('Canny1', TRACKBAR_WINDOW, 0, 200, nothing)
    cv2.createTrackbar('Canny2', TRACKBAR_WINDOW, 0, 500, nothing)
    # Set default value for Canny trackbars
    cv2.setTrackbarPos('KernelSize', TRACKBAR_WINDOW, 5)
    cv2.setTrackbarPos('Canny1', TRACKBAR_WINDOW, 100)
    cv2.setTrackbarPos('Canny2', TRACKBAR_WINDOW, 200)


# returns an HSV filter object based on the control GUI values
def get_hsv_filter_from_controls():
    # Get current positions of all trackbars
    hsv_filter = HsvFilter()
    hsv_filter.hMin = cv2.getTrackbarPos('HMin', TRACKBAR_WINDOW)
    hsv_filter.sMin = cv2.getTrackbarPos('SMin', TRACKBAR_WINDOW)
    hsv_filter.vMin = cv2.getTrackbarPos('VMin', TRACKBAR_WINDOW)
    hsv_filter.hMax = cv2.getTrackbarPos('HMax', TRACKBAR_WINDOW)
    hsv_filter.sMax = cv2.getTrackbarPos('SMax', TRACKBAR_WINDOW)
    hsv_filter.vMax = cv2.getTrackbarPos('VMax', TRACKBAR_WINDOW)
    hsv_filter.sAdd = cv2.getTrackbarPos('SAdd', TRACKBAR_WINDOW)
    hsv_filter.sSub = cv2.getTrackbarPos('SSub', TRACKBAR_WINDOW)
    hsv_filter.vAdd = cv2.getTrackbarPos('VAdd', TRACKBAR_WINDOW)
    hsv_filter.vSub = cv2.getTrackbarPos('VSub', TRACKBAR_WINDOW)
    return hsv_filter


# returns a Canny edge filter object based on the control GUI values
def get_edge_filter_from_controls():
    # Get current positions of all trackbars
    edge_filter = EdgeFilter()
    edge_filter.kernelSize = cv2.getTrackbarPos('KernelSize', TRACKBAR_WINDOW)
    edge_filter.erodeIter = cv2.getTrackbarPos('ErodeIter', TRACKBAR_WINDOW)
    edge_filter.dilateIter = cv2.getTrackbarPos('DilateIter', TRACKBAR_WINDOW)
    edge_filter.canny1 = cv2.getTrackbarPos('Canny1', TRACKBAR_WINDOW)
    edge_filter.canny2 = cv2.getTrackbarPos('Canny2', TRACKBAR_WINDOW)
    return edge_filter


# given an image and an HSV filter, apply the filter and return the resulting image.
# if a filter is not supplied, the control GUI trackbars will be used
def apply_hsv_filter(original_image, hsv_filter=None):
    # convert image to HSV
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # if we haven't been given a defined filter, use the filter values from the GUI
    if not hsv_filter:
        hsv_filter = get_hsv_filter_from_controls()

    # add/subtract saturation and value
    h, s, v = cv2.split(hsv)
    s = shift_channel(s, hsv_filter.sAdd)
    s = shift_channel(s, -hsv_filter.sSub)
    v = shift_channel(v, hsv_filter.vAdd)
    v = shift_channel(v, -hsv_filter.vSub)
    hsv = cv2.merge([h, s, v])

    # Set minimum and maximum HSV values to display
    lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
    upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])
    # Apply the thresholds
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(hsv, hsv, mask=mask)

    # convert back to BGR for imshow() to display it properly
    img = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    return img


# given an image and a Canny edge filter, apply the filter and retusrn the resulting image.
# if a filter is not supplied, the control GUI trackbars will be used
def apply_edge_filter(original_image, edge_filter=None):
    # if we haven't been given a defined filter, use the filter values from the GUI
    if not edge_filter:
        edge_filter = get_edge_filter_from_controls()

    kernel = np.ones((edge_filter.kernelSize, edge_filter.kernelSize), np.uint8)
    eroded_image = cv2.erode(original_image, kernel, iterations=edge_filter.erodeIter)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=edge_filter.dilateIter)

    # canny edge detection
    result = cv2.Canny(dilated_image, edge_filter.canny1, edge_filter.canny2)

    # convert single channel image back to BGR
    img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return img


# apply adjustments to an HSV channel
# https://stackoverflow.com/questions/49697363/shifting-hsv-pixel-values-in-python-using-numpy
def shift_channel(c, amount):
    if amount > 0:
        lim = 255 - amount
        c[c >= lim] = 255
        c[c < lim] += amount
    elif amount < 0:
        amount = -amount
        lim = amount
        c[c <= lim] = 0
        c[c > lim] -= amount
    return c
