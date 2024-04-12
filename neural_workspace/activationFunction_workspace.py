import numpy as np
import matplotlib.pyplot as plt

'''
  Sigmoid function

  @arg x: number

  @return: number

  The sigmoid function is a mathematical
  function used to map the output of a
  neuron to a value between 0 and 1.

  Notes: The output is between 0 and 1,
  it is not related to the probability of
  the neuron being activated.
'''
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

'''
  Step function

  @arg x: number

  @return: number

  The step function is a mathematical
  function used to map the output of a
  neuron to a value between 0 and 1.

  Notes: The output is between 0 and 1,
  it like round out the arg if it is
  greater than 0.
'''
def step(x):
  return np.heaviside(x, 1)

'''
  Lineal function

  @arg x: number

  @return: number

  The lineal function is a mathematical
  function used to map the output of a
  neuron to a value between 0 and 1.

  Notes: The output is the same as the
  input.
'''
def lineal(x):
  return x

'''
  ReLU(Recitified Linear Unit) function

  @arg x: number

  @return: number

  The ReLU function is a mathematical
  function used to map the output of a
  neuron to a value between 0 and 1.

  Notes: The output is the same as the
  input if it is greater than 0, otherwise
  it is 0.

'''
def relu(x):
  return np.maximum(0, x)

'''
  Leaky ReLU(Recitified Linear Unit) function

  @arg x: number

  @return: number

  The Leaky ReLU function is a mathematical
  function used to map the output of a
  neuron to a value between 0 and 1.
'''
def leaky_relu(x, alpha=0.1):
  return np.maximum(alpha * x, x)

'''
  Tanh function

  @arg x: number

  @return: number

  Note: This is like the sigmoid
  function, but it is centered at 0 allowing
  negative values.
'''
def tanh(x):
  return np.tanh(x)

'''
  Softmax function

  @arg x: number

  @return: number

  Notes: 
'''
def softmax(x):
  ex = np.exp(x - np.max(x))
  return ex / np.sum(ex, axis=0)

x = np.linspace(-2, 2, 100)
sigmoid_y = [sigmoid(i) for i in x]
step_y = [step(i) for i in x]
lineal_y = [lineal(i) for i in x]
relu_y = [relu(i) for i in x]
leaky_relu_y = [leaky_relu(i) for i in x]
tanh_y = [tanh(i) for i in x]
softmax_y = softmax(x)

plt.plot(x, sigmoid_y, label="Sigmoid", color="red")
plt.plot(x, step_y, label="Step", color="green")
plt.plot(x, lineal_y, label="Lineal", color="blue")
plt.plot(x, relu_y, label="ReLU", color="orange")
plt.plot(x, leaky_relu_y, label="Leaky ReLU", color="purple")
plt.plot(x, tanh_y, label="Tanh", color="black")
plt.plot(x, softmax_y, label="Softmax", color="brown")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sigmoid Function")
plt.grid()

plt.show()

