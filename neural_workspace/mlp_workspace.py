import numpy as np

class MLP:
  def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    # initialize weights matrix and biases
    self.W_input_hidden = np.random.rand(self.input_size, self.hidden_size)
    self.b_input_hidden = np.zeros((1, self.hidden_size))
    self.W_hidden_output = np.random.rand(self.hidden_size, self.output_size)
    self.b_hidden_output = np.zeros((1, self.output_size))

    # auxiliar functions

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def d_sigmoid(self, x):
    return x * (1 - x)

  def forward(self, input_data):
    # Forward propagation
    hidden_layer_input = np.dot(input_data, self.W_input_hidden) + self.b_input_hidden
    hidden_layer_output = self.sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, self.W_hidden_output) + self.b_hidden_output
    output = self.sigmoid(output_layer_input)

    return hidden_layer_output, output

  def backward(self, input_data, target, hidden_output, output, lr=0.2):
    # Backward propagation
    output_error = target - output
    output_grad = output_error * self.d_sigmoid(output)

    hidden_error = np.dot(output_grad, self.W_hidden_output.T)
    hidden_grad = hidden_error * self.d_sigmoid(hidden_output)

    # Update weights and biases using gradient descent
    self.W_hidden_output = self.W_hidden_output + np.dot(hidden_output.T, output_grad)*lr
    self.b_hidden_output = self.b_hidden_output + np.sum(output_grad)*lr

    self.W_input_hidden = self.W_input_hidden + np.dot(input_data.T, hidden_grad)*lr
    self.b_input_hidden = self.b_input_hidden + np.sum(hidden_grad, axis=0, keepdims=True)*lr

  def train(self, input_data, target, epochs=10000):
    for _ in range(epochs):
      hidden_output, output = self.forward(input_data)
      self.backward(input_data, target, hidden_output, output)

if __name__ == '__main__':
  # =================== XOR gate ===================
  # XOR truth table values
  xor_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  xor_target = np.array([[0], [1], [1], [0]])

  # MLP parameters
  input_size = 2
  hidden_size = 10
  output_size = 1

  # instatiate the MLP
  model = MLP(input_size, hidden_size, output_size)
  model.train(xor_input, xor_target)

  # test for each input
  test_xor_00 = np.array([[0, 0]])
  test_xor_01 = np.array([[0, 1]])
  test_xor_10 = np.array([[1, 0]])
  test_xor_11 = np.array([[1, 1]])
  _, prediction_00 = model.forward(test_xor_00)
  _, prediction_01 = model.forward(test_xor_01)
  _, prediction_10 = model.forward(test_xor_10)
  _, prediction_11 = model.forward(test_xor_11)

  prediction_00 = 1 if prediction_00 > 0.5 else 0
  prediction_01 = 1 if prediction_01 > 0.5 else 0
  prediction_10 = 1 if prediction_10 > 0.5 else 0
  prediction_11 = 1 if prediction_11 > 0.5 else 0

  print("===================== XOR gate =====================")
  print("Predicted 00 output:", prediction_00)
  print("Predicted 01 output:", prediction_01)
  print("Predicted 10 output:", prediction_10)
  print("Predicted 11 output:", prediction_11)

  # =================== AND gate ===================
  # AND truth table values
  and_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  and_target = np.array([[0], [0], [0], [1]])

  # MLP parameters
  input_size = 2
  hidden_size = 1
  output_size = 1

  # instatiate the MLP
  model = MLP(input_size, hidden_size, output_size)
  model.train(and_input, and_target)

  # test for each input
  test_and_00 = np.array([[0, 0]])
  test_and_01 = np.array([[0, 1]])
  test_and_10 = np.array([[1, 0]])
  test_and_11 = np.array([[1, 1]])
  _, prediction_00 = model.forward(test_and_00)
  _, prediction_01 = model.forward(test_and_01)
  _, prediction_10 = model.forward(test_and_10)
  _, prediction_11 = model.forward(test_and_11)

  prediction_00 = 1 if prediction_00 > 0.5 else 0
  prediction_01 = 1 if prediction_01 > 0.5 else 0
  prediction_10 = 1 if prediction_10 > 0.5 else 0
  prediction_11 = 1 if prediction_11 > 0.5 else 0

  print("===================== AND gate =====================")
  print("Predicted 00 output:", prediction_00)
  print("Predicted 01 output:", prediction_01)
  print("Predicted 10 output:", prediction_10)
  print("Predicted 11 output:", prediction_11)

  # =================== OR gate ===================
  # OR truth table values
  or_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  or_target = np.array([[0], [1], [1], [1]])

  # MLP parameters
  input_size = 2
  hidden_size = 1
  output_size = 1

  # instatiate the MLP
  model = MLP(input_size, hidden_size, output_size)
  model.train(or_input, or_target)

  # test for each input
  test_or_00 = np.array([[0, 0]])
  test_or_01 = np.array([[0, 1]])
  test_or_10 = np.array([[1, 0]])
  test_or_11 = np.array([[1, 1]])
  _, prediction_00 = model.forward(test_or_00)
  _, prediction_01 = model.forward(test_or_01)
  _, prediction_10 = model.forward(test_or_10)
  _, prediction_11 = model.forward(test_or_11)

  prediction_00 = 1 if prediction_00 > 0.5 else 0
  prediction_01 = 1 if prediction_01 > 0.5 else 0
  prediction_10 = 1 if prediction_10 > 0.5 else 0
  prediction_11 = 1 if prediction_11 > 0.5 else 0

  print("===================== OR gate =====================")
  print("Predicted 00 output:", prediction_00)
  print("Predicted 01 output:", prediction_01)
  print("Predicted 10 output:", prediction_10)
  print("Predicted 11 output:", prediction_11)

  # =================== NOT gate ===================
  # NOR truth table values
  nor_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  nor_target = np.array([[1], [0], [0], [0]])

  # MLP parameters
  input_size = 2
  hidden_size = 1
  output_size = 1

  # instatiate the MLP
  model = MLP(input_size, hidden_size, output_size)
  model.train(nor_input, nor_target)

  # test for each input
  test_nor_00 = np.array([[0, 0]])
  test_nor_01 = np.array([[0, 1]])
  test_nor_10 = np.array([[1, 0]])
  test_nor_11 = np.array([[1, 1]])
  _, prediction_00 = model.forward(test_nor_00)
  _, prediction_01 = model.forward(test_nor_01)
  _, prediction_10 = model.forward(test_nor_10)
  _, prediction_11 = model.forward(test_nor_11)

  prediction_00 = 1 if prediction_00 > 0.5 else 0
  prediction_01 = 1 if prediction_01 > 0.5 else 0
  prediction_10 = 1 if prediction_10 > 0.5 else 0
  prediction_11 = 1 if prediction_11 > 0.5 else 0

  print("===================== NOR gate =====================")
  print("Predicted 00 output:", prediction_00)
  print("Predicted 01 output:", prediction_01)
  print("Predicted 10 output:", prediction_10)
  print("Predicted 11 output:", prediction_11)
