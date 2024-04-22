import os
import gzip
import numpy as np
import cv2
import pickle5 as pickle
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))

def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

def load_data():
  mnist = gzip.open(os.path.join(DIR, 'mnist.pkl.gz'), 'rb')
  training_data, classification_data, test_data = pickle.load(mnist, encoding='latin1')
  mnist.close()
  return (training_data, classification_data, test_data)

def wrap_data():
  tr_d, va_d, te_d = load_data()
  training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
  training_results = [vectorized_result(y) for y in tr_d[1]]
  training_data = zip(training_inputs, training_results)
  validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
  validation_data = zip(validation_inputs, va_d[1])
  test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
  test_data = zip(test_inputs, te_d[1])
  return (training_data, validation_data, test_data)


def display_image(data):
  for i in range(10):
    img = data[0][i]
    img = img.reshape(28, 28)
    label = data[1][i]
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()

    
if __name__ == '__main__':
  training_data, validation_data, test_data = wrap_data()
  tr_d, va_d, te_d = load_data()
  display_image(te_d)
