'''
Binary Robust Independent Elementary Features (BRIEF)

- It is faster than SIFT and SURF.

It is a 4-step process:
  1. Initialize the FAST detector
  2. Initialize the BRIEF descriptor
  3. Detect key points with FAST
  4. Compute the descriptors with BRIEF
  5. Draw the key points
  6. Write/Show the image with the key points
'''
from sys import path
path.append('..')
from helpers.main import *

fast = CV.FastFeatureDetector_create()
brief = CV.xfeatures2d.BriefDescriptorExtractor_create()

keypoints = fast.detect(GRAY_IMAGE, None)
keypoints, descriptors = brief.compute(GRAY_IMAGE, keypoints)

img = CV.drawKeypoints(GRAY_IMAGE, keypoints, MAIN_IMAGE)

save_image(img, 'brief_keypoints.jpeg')
show_image(img, 'BRIEF Keypoints')
