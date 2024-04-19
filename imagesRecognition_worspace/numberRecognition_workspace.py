import os
import gzip
import pickle5 as pickle

DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
  mnist = gzip.open(os.path.join(DIR, 'mnist.pkl.gz'), 'rb')
  training_data, classification_data, test_data = pickle.load(mnist, encoding='latin1')
  mnist.close()
  return (training_data, classification_data, test_data)

print(load_data())
