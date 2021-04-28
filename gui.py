from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit
from PySide2.QtGui import QImage, QPainter, QMouseEvent, QPen, QPaintEvent
from PySide2.QtCore import Qt, QPoint

from Model import Model1, Model2
from PIL import Image

import torch
import numpy as np


NET = Model1()
NET.load_state_dict(torch.load('../model.pth'))
NET.eval()


def prepare_image(path: str):
    """
    Converting image to MNIST dataset format
    """

    im = Image.open(path).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    new_image = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        new_image.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        new_image.paste(img, (wleft, 4))  # paste resized image on white canvas

    pixels = list(new_image.getdata())  # get pixel values
    pixels_normalized = [(255 - x) * 1.0 / 255.0 for x in pixels]

    # Need adequate shape
    adequate_shape = np.reshape(pixels_normalized, (1, 28, 28))
    output = torch.FloatTensor(adequate_shape).unsqueeze(0)
    return output


class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.setWindowTitle('Handwritten digit recognition')

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.init_text()

        self.drawing = False
        self.brush_size = 8
        self.brush_color = Qt.black
        self.last_point = QPoint()

        self.init_btn_clear()
        self.init_btn_recognize()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event: QPaintEvent):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def init_btn_clear(self):
        btn = QPushButton('Clear', self)
        btn.resize(80, 25)
        btn.move(50, 340)
        btn.show()
        btn.clicked.connect(self.clear)

    def clear(self):
        self.image.fill(Qt.white)
        self.text.setText('')
        self.update()

    def init_btn_recognize(self):
        btn = QPushButton('Recognize', self)
        btn.resize(80, 25)
        btn.move(150, 340)
        btn.show()
        btn.clicked.connect(self.recognize)

    def init_text(self):
        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QTextEdit.NoWrap)
        self.text.insertPlainText('')
        font = self.text.font()
        font.setFamily('Rockwell')
        font.setPointSize(25)
        self.text.setFont(font)
        self.text.resize(50, 50)
        self.text.move(266, 324)

    def recognize(self):
        # Convert to image
        image = self.image.convertToFormat(QImage.Format_ARGB32)
        width = image.width()
        height = image.height()
        ptr = image.constBits()
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        im = Image.fromarray(arr[..., :3])
        im.save('dataset/img.png')

        # Evaluate net and show result
        input_img = prepare_image('dataset/img.png')
        prediction = torch.argmax(NET(input_img)).item()
        self.text.setText(' '+str(prediction))


