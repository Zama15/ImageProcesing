'''
Features from Accelerated Segment Test (FAST)

- It is faster than other corner detection algorithms.
- It is used in combination with other feature descriptors such as SIFT, SURF, and ORB.

It is a 4-step process:
  1. Initialize the FAST detector
  2. Detect key points:
    FAST detects corners in images by comparing the
    intensity of the pixel with the intensity of the
    surrounding pixels.
    If the intensity of the pixel is greater or less
    than the intensity of the surrounding pixels by a
    certain threshold, then the pixel is a corner.
  3. Draw the key points
  4. Write/Show the image with the key points
'''
from sys import path
path.append('..')
from helpers.main import *

fast = CV.FastFeatureDetector_create()

keypoints = fast.detect(GRAY_IMAGE, None)

img = CV.drawKeypoints(GRAY_IMAGE, keypoints, MAIN_IMAGE)

save_image(img, 'fast_keypoints.jpeg')
show_image(img, 'FAST Keypoints')
