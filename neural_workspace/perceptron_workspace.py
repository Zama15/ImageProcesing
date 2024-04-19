import numpy as np

class Perceptron:
  def __init__(self, n_inputs, learning_rate=0.1):
    self.weights = np.random.rand(n_inputs)
    self.bias = 0
    self.learning_rate = learning_rate

  # Make the addition of the inputs with the weights and the bias(like neural_workspace/network_workspace.py)
  # If the result is greater than 0, then the neuron is activated, otherwise it is not activated
  def predict(self, inputs):
    suma = np.dot(inputs, self.weights) + self.bias
    if suma > 0:
      return 1 
    else:
      return 0

  # epochs: number of times the training will be done
  # targets: the real values that the perceptron should predict
  # The training detect the error and adjust the weights and the bias
  # if the error is 0, then the training it is done
  def train(self, targets, n_epocas, inputs):
    for epocas in range(n_epocas):
      error = 0
      for i in range(len(inputs)):
        output = self.predict(inputs[i]) # la prediccion
        delta = targets[i] - output # restar la prediccion con el target(el deseado) para obtener el error
        self.weights += delta * inputs[i] * self.learning_rate # retropropagacion, ajustar los pesos en caso de que el error sea diferente de 0
        self.bias += delta * self.learning_rate # ajustar el bias en caso de que el error sea diferente de 0
        error += abs(delta) # sumar el error para saber si es 0

      if error == 0:
        print(f'Epoca: {epocas}')
        break


if __name__ == '__main__':
  print('============ AND GATE 2 ENTRIES ============')
  and2Entries = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
  targets2Entries = np.array([0, 0, 0, 1])

  perceptron = Perceptron(2)

  print('After training')
  perceptron.train(targets2Entries, 1000, and2Entries)
  for i in range(len(and2Entries)):
    output = perceptron.predict(and2Entries[i])
    print(f'Input: {and2Entries[i]} Output: {output}')
  
  print('============ AND GATE 3 ENTRIES ============')
  and3Entries = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
  targets3Entries = np.array([0, 0, 0, 0, 0, 0, 0, 1])

  perceptron = Perceptron(3)

  print('After training')
  perceptron.train(targets3Entries, 1000, and3Entries)
  for i in range(len(and3Entries)):
    output = perceptron.predict(and3Entries[i])
    print(f'Input: {and3Entries[i]} Output: {output}')
