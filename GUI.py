from random import random
from kivy.app import App
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics import Color, Line, Rectangle
from Function import test_image
import ctypes


def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        color = (0, 0, 0, 1)
        with self.canvas:
            Color(*color, mode='hsv')
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class MyPaintApp(App):
    def build(self):
        Config.write()
        Window.size = (300, 300)
        self.title = "My Painter"

        parent = Widget()
        self.painter = MyPaintWidget(size=Window.size)
        with self.painter.canvas:
            Color(255, 255, 255, 1)
            Rectangle(pos=self.painter.pos, size=self.painter.size)
        clearbtn = Button(text='Clear', size_hint=(None, None), height=80, width=100)
        test1 = Button(text='Test Model 1', size_hint=(None, None), height=80, width=100, pos=(100, 0))
        test2 = Button(text='Test Model 2', size_hint=(None, None), height=80, width=100, pos=(200, 0))

        clearbtn.bind(on_release=self.clear_canvas)
        test1.bind(on_release=self.test_model1)
        test2.bind(on_release=self.test_model2)

        parent.add_widget(self.painter)
        parent.add_widget(clearbtn, index=0)
        parent.add_widget(test1, index=1)
        parent.add_widget(test2, index=2)
        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        with self.painter.canvas:
            Color(255, 255, 255, 1)
            Rectangle(pos=self.painter.pos, size=self.painter.size)

    def test_model1(self, obj):
        self.painter.export_to_png('test/test.png')
        Mbox('Handwriting Recognition', str(test_image('test/test.png', 1)), 1)

    def test_model2(self, obj):
        self.painter.export_to_png('test/test.png')
        Mbox('Handwriting Recognition', str(test_image('test/test.png', 2)), 1)

if __name__ == '__main__':
    MyPaintApp().run()