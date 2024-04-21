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

'''
  Binary Crossentropy

  @arg y_true: list of real values
  @arg y_pred: list of predicted values

  @return: number
'''
def binary_crossentropy(y_true, y_pred):
  total_loss = 0
  for i in range(len(y_true)):
    true_value = y_true[i]
    pred_value = y_pred[i]
    loss = true_value * np.log(pred_value) + (1 - true_value) * np.log(1 - pred_value)
    total_loss += loss
  average_loss = -total_loss / len(y_true)
  return average_loss

'''
  Categorical Crossentropy

  @arg sp_pred: value for the positive class
  @arg sn_pred: list of predicted values

  @return: number
'''
def categorical_crossentropy(sp_pred, sn_pred):
  sp_exp = np.exp(sp_pred)
  sn_exp = 0
  for i in range(len(sn_pred)):
    sn_exp += np.exp(sn_pred[i])
  return -np.log(sp_exp / sn_exp)
