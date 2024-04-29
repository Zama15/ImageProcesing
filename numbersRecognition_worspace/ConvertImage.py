from os import path as Path, remove as Remove
from PIL import Image
from numpy import reshape as Reshape
import cv2

def EPS2PNG(path, name):
  epsPath = Path.join(path, name + ".eps")
  img = Image.open(epsPath)

  pngPath = Path.join(path, name + ".png")
  if not(img.save(pngPath, "png")):
    Remove(epsPath)
    invertColors(pngPath)
  else:
    print("Error: Image not saved")

def invertColors(image):
  img = cv2.imread(image)
  img = cv2.bitwise_not(img)
  cv2.imwrite(image, img)
    
def ToVector(image):
  img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (28, 28))
  vector = Reshape(img, (784, 1))
  vector = vector / 255.0
  return vector


if __name__ == "__main__":
  path = Path.join(Path.dirname(Path.abspath(__file__)), 'imgs')
  name = 'numberToPredict.png'

  print(ToVector(Path.join(path, name)))
