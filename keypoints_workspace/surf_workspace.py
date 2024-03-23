'''
Speeded Up Robust Features (SURF)

- It is faster than SIFT.
- It is robust to rotation, scaling, and noise.
- It is used in combination with other feature descriptors such as SIFT, FAST, and BRIEF.

It is a 4-step process:
  1. Initialize the SURF detector
  2. Detect key points:
    SURF uses a Hessian matrix to detect key points.
    The key points are detected at extrema in the
    determinant of the Hessian matrix.
  3. Draw the key points
  4. Write/Show the image with the key points

Note: SURF is not available in OpenCV 4.4.0 due to patent issues. It will be used Akaze instead.

Differences between SURF and AKAZE:
- SURF is faster than SIFT but slower than AKAZE.
- AKAZE is free to use in commercial applications.
- AKAZE is more robust than SURF.
'''
from sys import path
path.append('..')
from helpers.main import *

akaze = CV.AKAZE_create()

keypoints = akaze.detect(GRAY_IMAGE, None)

img = CV.drawKeypoints(GRAY_IMAGE, keypoints, MAIN_IMAGE)

save_image(img, 'surf_keypoints.jpeg')
show_image(img, 'SURF Keypoints')
