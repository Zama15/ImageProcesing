from Network import Network
from os import path as Path
from Paint import Paint

if __name__ == '__main__':
  dir = Path.dirname(Path.abspath(__file__))
  modelPath = Path.join(dir, 'srcs', 'model.pkl')
  imageToPredict = Path.join(dir, 'imgs', 'numberToPredict.png')

  net = Network([784, 30, 10])
  net.load(modelPath)

  app = Paint(net.feedforward)
  app.run()
