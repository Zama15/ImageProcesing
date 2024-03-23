'''
This file contains helper functions for the keypoints workspace.
Such as global variables, error handling, and other utility functions.

Global variables:
- ROOT_DIR: The root directory of the project. i.e. ImageProcessing
- CV: The OpenCV library
- MAIN_IMAGE: The main image used in the workspace
- GRAY_IMAGE: The main image converted to grayscale

Utility functions:
- save_image: Save an image to the imgs directory
- show_image: Display an image in a window

'''

import cv2
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV = cv2
MAIN_IMAGE = CV.imread(os.path.join(ROOT_DIR, 'imgs/image1.jpeg'))
GRAY_IMAGE = CV.cvtColor(MAIN_IMAGE, CV.COLOR_BGR2GRAY)

def save_image(image, file_name):
  try:
    CV.imwrite(os.path.join(CURRENT_DIR, 'imgs', file_name), image)
  except Exception as e:
    print('Error saving image:', e)

def show_image(image, title='Image'):
  try:
    CV.imshow(title, image)
    CV.waitKey(0)
    CV.destroyAllWindows()
  except Exception as e:
    print('Error showing image:', e)

def show_multiple_images(images, titles):
  try:
    for i in range(len(images)):
      CV.imshow(titles[i], images[i])
    CV.waitKey(0)
    CV.destroyAllWindows()
  except Exception as e:
    print('Error showing images:', e)

print('Helper functions loaded successfully')
print('ROOT_DIR:', ROOT_DIR)
print('CURRENT_DIR:', CURRENT_DIR)

