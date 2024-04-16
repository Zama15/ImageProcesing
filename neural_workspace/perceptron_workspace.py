import numpy as np

class Perceptron:
  def __init__(self, n_inputs, learning_rate=1):
    self.weights = np.random.rand(n_inputs)
    self.bias = 0 # np.random.rand()
    self.learning_rate = learning_rate

  def predict(self, inputs):
    # return np.dot(inputs, self.weights) + self.bias 
    suma = np.dot(inputs, self.weights) + self.bias
    if suma > 0:
      return 1
    else:
      return 0

  def train(self, targets, n_epocas, inputs):
    for epocas in range(n_epocas):
      error = 0
      for i in range(len(inputs)):
        output = self.predict(inputs[i]) # la prediccion
        delta = targets[i] - output # restar la prediccion con el target(el deseado) para obtener el error
        self.weights += delta * inputs[i] * self.learning_rate # retropropagacion, ajustar los pesos en caso de que el error sea diferente de 0
        self.bias += delta * self.learning_rate # ajustar el bias en caso de que el error sea diferente de 0
        error += delta # sumar el error para saber si es 0

      if error == 0:
        print(f'Epoca: {epocas}')
        break


if __name__ == '__main__':
  inputs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]])
  targets = np.array([0, 0, 0, 0, 0, 0, 1])
  
  perceptron = Perceptron(3)
  print(perceptron.weights)
  print(perceptron.bias)
  perceptron.train(targets, 100, inputs)
  print(perceptron.weights)
  print(perceptron.bias)
  for i in range(len(inputs)):
    output = perceptron.predict(inputs[i])
    print(f'Input: {inputs[i]} Output: {output}')
