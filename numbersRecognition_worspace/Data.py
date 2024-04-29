from gzip import open as Open
from Network import Network
from numpy import zeros as Zeros, reshape as Reshape
from pickle5 import load as Load
from ConvertImage import ToVector
from os import path as Path

DIR = Path.dirname(Path.abspath(__file__))
SRC = Path.join(DIR, 'srcs')

def vectorized_result(j):
  e = Zeros((10, 1))
  e[j] = 1.0
  return e

def load_data():
  mnist = Open(Path.join(SRC, 'mnist.pkl.gz'), 'rb')
  training_data, classification_data, test_data = Load(mnist, encoding='latin1')
  mnist.close()
  return (training_data, classification_data, test_data)

def wrap_data():
  tr_d, va_d, te_d = load_data()
  training_inputs = [Reshape(x, (784, 1)) for x in tr_d[0]]
  training_results = [vectorized_result(y) for y in tr_d[1]]
  training_data = zip(training_inputs, training_results)
  validation_inputs = [Reshape(x, (784, 1)) for x in va_d[0]]
  validation_data = zip(validation_inputs, va_d[1])
  test_inputs = [Reshape(x, (784, 1)) for x in te_d[0]]
  test_data = zip(test_inputs, te_d[1])
  return (training_data, validation_data, test_data)

    
if __name__ == '__main__':
  training_data, validation_data, test_data = wrap_data()
  # 784, 50, 10
  sizes = [784, 50, 10]
  # 50
  epochs = 40
  # 20
  mini_batch_size = 20
  # 4.0
  learning_rate = 4.0
  # total = 0.9525
  net = Network(sizes)
  print('Model:')
  print(f'Layers: {len(sizes)}')
  print(f'Input layer: {sizes[0]}')
  print(f'Hidden layer: {sizes[1:-1]}')
  print(f'Output layer: {sizes[-1]}')
  print(f'Epochs: {epochs}')
  print(f'Mini batch size: {mini_batch_size}')
  print(f'Learning rate: {learning_rate}')
  net.SGD(list(training_data), epochs, mini_batch_size, learning_rate, test_data=list(test_data))

  modelPath = Path.join(SRC, 'model.pkl')
  net.save(modelPath)

