'''
Oriented FAST and Rotated BRIEF (ORB)

- It is faster than SIFT and SURF.
- It is a combination of FAST and BRIEF.
- It is rotationa and scale invariant.
- It is robust to noise, occlusion, viewpoint changes, and illumination changes.

It is a 4-step process:
  1. Initialize the ORB detector
  2. Detect key points
  3. Compute the descriptors
  4. Draw the key points
  5. Write/Show the image with the key points
'''
from sys import path
path.append('..')
from helpers.main import *

orb = CV.ORB_create()

keypoints = orb.detect(GRAY_IMAGE, None)

keypoints, descriptors = orb.compute(GRAY_IMAGE, keypoints)

img = CV.drawKeypoints(GRAY_IMAGE, keypoints, MAIN_IMAGE)

save_image(img, 'orb_keypoints.jpeg')
show_image(img, 'ORB Keypoints')
