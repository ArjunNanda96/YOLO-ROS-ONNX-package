import cv2
import numpy as np
import os
import glob


CONE_CALIB_IMAGE = 'distance/cone_x40cm.png'
CONE_TEST_IMAGE = 'distance/cone_unknown.png'
CONE_CALIB_DISTANCE = 40.0
CONE_CALIB_BASE_P = [0.0, 0.0]
CONE_TEST_PTS_P = [0.0, 0.0]
CAMERA_HEIGHT = np.nan


def save_calib_coords(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        CONE_CALIB_BASE_P[0] = x
        CONE_CALIB_BASE_P[1] = y
        print(f"Base Location: (x, y) = ({x}, {y})")


def save_test_coords(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        CONE_TEST_PTS_P[0] = x
        CONE_TEST_PTS_P[1] = y
        print(f"Base Location: (x, y) = ({x}, {y})")


def compute_camera_height(A, yp, d):
    return abs(yp - A[1, 2]) * d / A[1, 1]


def estimate_pixel_pose(A, p, H):
    xc = A[1, 1] * (p[2] + H) / (p[1] - A[1, 2])
    yc = xc * (p[0] - A[0, 2]) / A[0, 0]
    return xc, yc


if __name__ == '__main__':
    # Define the dimensions of checkerboard
    CHECKERBOARD = (6, 8)

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                          * CHECKERBOARD[1],
                          3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored
    # in a given directory. Since no path is
    # specified, it will take current directory
    # jpg files alone
    images = glob.glob('/Users/arjunnanda/F1ESE619/lab-8-vision-lab-team_07/calibration/*.png')

    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            grayColor, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        print(ret)
        if ret == True:
            threedpoints.append(objectp3d)

            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)

            twodpoints.append(corners2)

            # Draw and display the corners
            image = cv2.drawChessboardCorners(image,
                                              CHECKERBOARD,
                                              corners2, ret)

        cv2.imshow('img', image)

    cv2.destroyAllWindows()

    h, w = image.shape[:2]

    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    # Import camera height calibration image
    img = cv2.imread(CONE_CALIB_IMAGE, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', save_calib_coords)
    print('Select point on base of cone, then press Q.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compute camera height and verify
    CAMERA_HEIGHT = compute_camera_height(matrix, CONE_CALIB_BASE_P[1], CONE_CALIB_DISTANCE)
    p_calib = np.array([CONE_CALIB_BASE_P[0], CONE_CALIB_BASE_P[1], 0])
    x_calib, y_calib = estimate_pixel_pose(matrix, p_calib, CAMERA_HEIGHT)

    # Import test cone image
    img = cv2.imread(CONE_TEST_IMAGE, 1)
    cv2.imshow('image_test', img)
    cv2.setMouseCallback('image_test', save_test_coords)
    print('Select point on base of cone, then press Q.')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compute cone distance
    p = np.array([CONE_TEST_PTS_P[0], CONE_TEST_PTS_P[1], 0.0])
    x_cone, y_cone = estimate_pixel_pose(matrix, p, CAMERA_HEIGHT)

    # Displaying required output
    print("Intrinsic Camera Matrix:")
    print(matrix)

    # print("\nDistortion Coefficient:")
    # print(distortion)
    #
    # print("\nRotation Vectors:")
    # print(r_vecs)
    #
    # print("\nTranslation Vectors:")
    # print(t_vecs)

    print(f"Camera Height: H = {CAMERA_HEIGHT} cm")
    print(f"Verification: X_calib = {x_calib}")
    print(f"Unknown Cone Pose: (X, Y)_cone = ({x_cone}, {y_cone}) cm")

