import cv2
import numpy as np

import odm as o

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# Load Object Detector
d = o.HomogeneousDetector()

# Load Image
img = cv2.imread("phone_aruco_marker.jpg")

# Get Aruco marker
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# Draw polygon around the marker
int_corners = np.int0(corners)
cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

# Aruco Perimeter
aruco_perimeter = cv2.arcLength(corners[0], True)

# Pixel to cm ratio
pixel_cm_ratio = aruco_perimeter / 20

#detect objects
c = d.detect_object(img)

# Draw objects boundaries
for cnt in c:
    # Get rect
    rect = cv2.minAreaRect(cnt)
    # print(rect)
    (x, y), (w, h), angle = rect

    # Get Width and Height of the Objects by applying the Ratio pixel to cm
    object_w = w / pixel_cm_ratio
    object_h = h / pixel_cm_ratio

    # Display rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
    cv2.putText(img, "Width {} cm".format(round(object_w, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(img, "Height {} cm".format(round(object_h, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)