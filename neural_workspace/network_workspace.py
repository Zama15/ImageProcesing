# File Extracted from: Pegueros Pérez Mariana Ixchel

import numpy as np

bias1 = 2
bias2 = 3
bias3 = 0.5
inputs = [ 1, 2, 3, 2.5, 1 ]
weights1 = [ 0.2, 0.8, -0.5, 1, bias1 ]
weights2 = [ 0.5, -0.91, 0.26, -0.5, bias2 ]
weights3 = [ -0.26, -0.27, 0.17, 0.87, bias3 ]

'''
  Numpy method dot(inputs, weights)

  @arg inputs: list of inputs, the last element is the 1 for the bias
  @arg weights: list of weights, the last element is the bias

  @return: sum of the product of each element of the lists
'''

outputs = [
  #neuron1:
  np.dot(inputs, weights1),
  #neuron2:
  np.dot(inputs, weights2),
  #neuron2:
  np.dot(inputs, weights3)
]

print(outputs)
