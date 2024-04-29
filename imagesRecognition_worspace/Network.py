from os import path as Path
from pickle5 import load as Load, dump as Dump
from random import shuffle as Shuffle
from ConvertImage import ToVector
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network(object):
  def __init__(self, sizes):
      '''Constructor
      The constructor of the Network class, it is used to initialize the model

      Parameters:
      sizes: list
        The list will contain the number of neurons in each layer of the network
        The first element of the list should be the number of neurons in the input layer
        The last element of the list should be the number of neurons in the output layer
        The other elements of the list should be the number of neurons in the hidden layers
      '''
      self.num_layers = len(sizes)
      self.sizes = sizes
      self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
      self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

  def feedforward(self, a):
    '''Feedforward
    The Predict method of the model, it is used to predict the output of the model

    Parameters:
    a: np.array
        The input to predict the output
    '''
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a

  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    '''Stochastic Gradient Descent
    The training method of the model

    Parameters:
    training_data: list
        The training data to train the model, it should be a list of tuples(x, y):
        x: training input
        y: expected output
    epochs: int
        The number of epochs to train the model
    mini_batch_size: int
        The size of the mini-batches to use when sampling the training data, 
        the mini-batches are used to update the model weights and biases
    eta: float
        The learning rate, it is used to control the step size when updating the model weights and biases
    test_data: list
        It is used to see the model performance on the test data after each epoch
    '''
    if test_data:
        n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        Shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
        if test_data:
            test_result = self.evaluate(test_data)
            print(f"Epoch {j}: {test_result} / {n_test}")
        else:
            print(f"{j} complete")

  def update_mini_batch(self, mini_batch, eta):
      nabla_b = [np.zeros(b.shape) for b in self.biases]
      nabla_w = [np.zeros(w.shape) for w in self.weights]
      for x, y in mini_batch:
          delta_nabla_b, delta_nabla_w = self.backprop(x, y)
          nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
          nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
      self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
      self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

  def backprop(self, x, y):
      nabla_b = [np.zeros(b.shape) for b in self.biases]
      nabla_w = [np.zeros(w.shape) for w in self.weights]
      # Feedforward
      activation = x
      activations = [x]
      zs = []
      for b, w in zip(self.biases, self.weights):
          z = np.dot(w, activation) + b
          zs.append(z)
          activation = sigmoid(z)
          activations.append(activation)
      # Backward pass
      delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
      nabla_b[-1] = delta
      nabla_w[-1] = np.dot(delta, activations[-2].transpose())
      for l in range(2, self.num_layers):
          z = zs[-l]
          sp = sigmoid_prime(z)
          delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
          nabla_b[-l] = delta
          nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
      return (nabla_b, nabla_w)

  def evaluate(self, test_data):
      test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
      return sum(int(x == y) for (x, y) in test_results) / len(test_data)

  def cost_derivative(self, output_activations, y):
      return (output_activations - y)
  
  def save(self, filename):
    model_data = {
      'biases': self.biases,
      'weights': self.weights
    }

    with open(filename, 'wb') as file:
      Dump(model_data, file)

  def load(self, filename):
    data = Load(open(filename, "rb"))
    self.biases = data['biases']
    self.weights = data['weights']

if __name__ == '__main__':
  dir = Path.dirname(Path.abspath(__file__))
  modelPath = Path.join(dir, 'srcs', 'model.pkl')
  imageToPredict = Path.join(dir, 'imgs', 'numberToPredict.png')

  net = Network([784, 30, 10])
  net.load(modelPath)

  vector = ToVector(imageToPredict)
  print(np.argmax(net.feedforward(vector)))
