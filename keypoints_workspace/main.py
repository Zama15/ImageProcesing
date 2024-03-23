'''
In conclusion, we have seen how to detect keypoints using different algorithms.
The main difference between these algorithms is the speed and robustness. But the
most important thing is to know when to use each algorithm because each one meet the
objective of detecting keypoints.
'''
from sys import path
path.append('..')
from helpers.main import *

sift = CV.SIFT_create()
akaze = CV.AKAZE_create()
fast = CV.FastFeatureDetector_create()
brief = CV.xfeatures2d.BriefDescriptorExtractor_create()
orb = CV.ORB_create()

sift_keypoints = sift.detect(GRAY_IMAGE, None)
akaze_keypoints = akaze.detect(GRAY_IMAGE, None)
fast_keypoints = fast.detect(GRAY_IMAGE, None)
brief_keypoints, brief_descriptors = brief.compute(GRAY_IMAGE, fast_keypoints)
orb_keypoints, orb_descriptors = orb.compute(GRAY_IMAGE, fast_keypoints)

sift_img = CV.drawKeypoints(GRAY_IMAGE, sift_keypoints, MAIN_IMAGE)
akaze_img = CV.drawKeypoints(GRAY_IMAGE, akaze_keypoints, MAIN_IMAGE)
fast_img = CV.drawKeypoints(GRAY_IMAGE, fast_keypoints, MAIN_IMAGE)
brief_img = CV.drawKeypoints(GRAY_IMAGE, brief_keypoints, MAIN_IMAGE)
orb_img = CV.drawKeypoints(GRAY_IMAGE, orb_keypoints, MAIN_IMAGE)

images_array = [sift_img, akaze_img, fast_img, brief_img, orb_img]
titles_array = ['SIFT Keypoints', 'AKAZE Keypoints', 'FAST Keypoints', 'BRIEF Keypoints', 'ORB Keypoints']

show_multiple_images(images_array, titles_array)
