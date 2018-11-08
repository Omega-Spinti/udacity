import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2


# prepare object points
nx = 8
ny = 6

img = mpimg.imread("images/calibration_test.png")
plt.imshow(img)

objpoints = []  # 3D points i nreal world space
imgpoints = []  # 2D points in image plane

# Prepare object points, like (0,0,0), (1,0,0), .... (7,5,0)
objp = np.zeros((6*8, 3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)  # x, y coordinates

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)


# If corners are found, add object points, image points
if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    img = cv2.drawChessboardCorners(img , (nx, ny), corners, ret)
    plt.imshow(img)
    plt.show()