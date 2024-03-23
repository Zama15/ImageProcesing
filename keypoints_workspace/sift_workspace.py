'''
Scale-Invariant Feature Transform (SIFT)

- SIFT is a method to detect and describe local features in images.

It is a 4-step process:
  1. Initialize the SIFT detector
  2. Detect key points:
    SIFT searches for extrema in the difference of
    Gaussian function. These are points where the
    image brightness changes rapidly in all directions.
  3. Draw the key points
  4. Write/Show the image with the key points
'''
from sys import path
path.append('..')
from helpers.main import *

sift = CV.SIFT_create()

keypoints = sift.detect(GRAY_IMAGE, None)

img = CV.drawKeypoints(GRAY_IMAGE, keypoints, MAIN_IMAGE)

save_image(img, 'sift_keypoints.jpeg')
show_image(img, 'SIFT Keypoints')
