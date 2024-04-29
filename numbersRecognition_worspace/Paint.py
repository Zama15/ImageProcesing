from ConvertImage import EPS2PNG
from ConvertImage import ToVector
from numpy import argmax as Argmax
from os import path
from tkinter import Tk, Frame, Canvas, CENTER, Button, NW, Label, SOLID, DOTBOX

########### Constants ###########

MAIN_COLOR = "black"
SECONDARY_COLOR = "white"
WINDOW_HEIGHT = 460
WINDOW_WIDTH = 540
CANVAS_HEIGHT = 280
CANVAS_WIDTH = 280
IMG_PATH = path.join(path.dirname(path.abspath(__file__)), 'imgs')
IMG_NAME = 'numberToPredict'

########### Class ###########

class Paint:
  def __init__(self, predictFunc):
    self.root = Tk()
    self._setup_window()
    self._create_frames()
    self._create_canvas()
    self._create_tools()
    self._bind_events()
    self.__prevPoint = [0, 0]
    self.__currentPoint = [0, 0]
    self.__penColor = MAIN_COLOR
    self.stroke = 9
    self.predict = 'None'
    self.__predicting = False
    self._cnn_tools(predictFunc)
    self.vectorizedImage = []

  def _setup_window(self):
    self.root.title("Number Recognition")
    self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    self.root.resizable(False, False)

  def _create_frames(self):
    self.frame1 = Frame(self.root, height=150, width=540)
    self.frame1.grid(row=0, column=0)
    self.holder = Frame(self.frame1, height=120, width=510, bg=SECONDARY_COLOR, padx=6, pady=10)
    self.holder.grid(row=0, column=0, sticky=NW)
    self.holder.place(relx=0.5, rely=0.5, anchor=CENTER)
    self.holder.columnconfigure(0, minsize=120)
    self.holder.columnconfigure(1, minsize=120)
    self.holder.columnconfigure(2, minsize=120)
    self.holder.columnconfigure(3, minsize=120)
    self.holder.rowconfigure(0, minsize=30)

    self.frame2 = Frame(self.root, height=310, width=540)
    self.frame2.grid(row=1, column=0)

  def _create_canvas(self):
    self.canvas = Canvas(self.frame2, height=CANVAS_HEIGHT, width=CANVAS_WIDTH, bg=SECONDARY_COLOR)
    self.canvas.grid(row=0, column=0)
    self.canvas.place(relx=0.5, rely=0.5, anchor=CENTER)
    self.canvas.config(cursor="pencil")

  def _create_tools(self):
    label123 = Label(self.holder, text="TOOLS", borderwidth=1, relief=SOLID, width=15)
    label123.grid(row=0, column=0)
    pencilButton = Button(self.holder, text="Pencil", height=1, width=12, command=self.__pencil)
    pencilButton.grid(row=1, column=0)
    __eraserButton = Button(self.holder, text="Eraser", height=1, width=12, command=self.__eraser)
    __eraserButton.grid(row=2, column=0)

    label7 = Label(self.holder, text="OTHER", borderwidth=1, relief=SOLID, width=15)
    label7.grid(row=0, column=2)
    clearButton = Button(self.holder, text="CLEAR", height=1, width=12, command=self.__clearScreen)
    clearButton.grid(row=1, column=2)
    exitButton = Button(self.holder, text="Exit", height=1, width=12, command=self.root.destroy)
    exitButton.grid(row=2, column=2)

    label8910 = Label(self.holder, text="STROKE SIZE", borderwidth=1, relief=SOLID, width=15)
    label8910.grid(row=0, column=3)
    sizeiButton = Button(self.holder, text="Increase", height=1, width=12, command=self.__strokeI)
    sizeiButton.grid(row=1, column=3)
    sizedButton = Button(self.holder, text="Decrease", height=1, width=12, command=self.__strokeD)
    sizedButton.grid(row=2, column=3)
    defaultButton = Button(self.holder, text="Default", height=1, width=12, command=self.__strokeDf)
    defaultButton.grid(row=3, column=3)

  def _cnn_tools(self, predictFunc):
    label456 = Label(self.holder, text="CNN TOOLS", borderwidth=1, relief=SOLID, width=15)
    label456.grid(row=0, column=1)
    predictButton = Button(self.holder, text="Predict", height=1, width=12, command=lambda: self.__predict(predictFunc),
                           background=MAIN_COLOR, activebackground="#393837",
                           foreground=SECONDARY_COLOR, activeforeground=SECONDARY_COLOR)
    predictButton.grid(row=1, column=1)
    predictResult = Label(self.holder, text=f"Prediction: {self.predict}", relief=SOLID, width=15, height=1)
    predictResult.grid(row=2, column=1)

  def __predict(self, predictFunc):
    self.__saveImg()
    self.vectorizedImage = ToVector(path.join(IMG_PATH, IMG_NAME + '.png'))
    self.predict = Argmax(predictFunc(self.vectorizedImage))
    predictResult = Label(self.holder, text=f"Prediction: {self.predict}", relief=SOLID, width=15, height=1)
    predictResult.grid(row=2, column=1)


  def _bind_events(self):
    self.canvas.bind("<B1-Motion>", self.__paint)
    self.canvas.bind("<ButtonRelease-1>", self.__paint)
    self.canvas.bind("<Button-1>", self.__paint)

  def __strokeI(self):
    if self.stroke != 10:
        self.stroke += 1

  def __strokeD(self):
    if self.stroke != 1:
        self.stroke -= 1

  def __strokeDf(self):
    self.stroke = 1

  def __pencil(self):
    self.__penColor = MAIN_COLOR
    self.canvas["cursor"] = "pencil"

  def __eraser(self):
    self.__penColor = SECONDARY_COLOR
    self.canvas["cursor"] = DOTBOX

  def __paint(self, event):
    x = event.x
    y = event.y

    self.__currentPoint = [x, y]

    if self.__prevPoint != [0, 0]:
      self.canvas.create_polygon(
        self.__prevPoint[0],
        self.__prevPoint[1],
        self.__currentPoint[0],
        self.__currentPoint[1],
        fill=self.__penColor,
        outline=self.__penColor,
        width=self.stroke,
      )

    self.__prevPoint = self.__currentPoint

    if event.type == "5":
      self.__prevPoint = [0, 0]

  def __clearScreen(self):
    self.canvas.delete("all")

  def __saveImg(self):
    self.canvas.postscript(file=path.join(IMG_PATH, IMG_NAME + '.eps'), colormode='color')
    EPS2PNG(IMG_PATH, IMG_NAME)

  def run(self):
    self.root.mainloop()


if __name__ == "__main__":
  def predict():
    print("Predicting...")
      
  app = Paint(predict)
  app.run()
