import numpy as np

'''
  Mean Absolute Error

  @arg y_true: list of real values
  @arg y_pred: list of predicted values

  @return: number
'''
def mae(y_true, y_pred):
  return sum([abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]) / len(y_true)

'''
  Mean Squared Error

  @arg y_true: list of real values
  @arg y_pred: list of predicted values

  @return: number
'''
def mse(y_true, y_pred):
  return sum([abs(y_true[i] - y_pred[i])**2 for i in range(len(y_true))]) / len(y_true)

'''
  Root Mean Squared Error

  @arg y_true: list of real values
  @arg y_pred: list of predicted values

  @return: number
'''
def rmse(y_true, y_pred):
  return np.sqrt(mse(y_true, y_pred))

def binary_crossentropy(y_true, y_pred):
  return -sum([y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i]) for i in range(len(y_true))]) / len(y_true)

def categorical_crossentropy(y_true, y_pred):
  return -sum([y_true[i] * np.log(y_pred[i]) for i in range(len(y_true))]) / len(y_true)
