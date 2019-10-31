from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse, Line
from mnist_classifier import NNClassifier, KNNClassifier
from kivy.core.window import Window

model_idx = 0
models = {0 : KNNClassifier(), 
1 : NNClassifier('cnn_model.pt'), 
2 : NNClassifier('cnn_model_aug.pt')}
model_names = {0 : 'KNN', 
1 : 'Conv Net', 
2 : 'Conv Net Aug'}


from kivy.config import Config
Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '400')
# Config.write()

Window.size = (400, 400)

class MyPaintWidget(Widget):
    def __init__(self, **kwargs):
        self.brush_width = 26
        super(MyPaintWidget, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(
            self._keyboard_closed, self, 'text')
        if self._keyboard.widget:
            # If it exists, this widget is a VKeyboard object which you can use
            # to change the keyboard layout.
            pass
    #     self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _keyboard_closed(self):
        print('My keyboard have been closed!')
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def on_touch_down(self, touch):
        with self.canvas:
            Color((1, 1, 1))
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=self.brush_width)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class MiniPaintApp(App):

    def build(self):
        self.classifier = KNNClassifier()

        parent = Widget()
        self.painter = MyPaintWidget(size=(800, 800))
        self.clear_btn = Button(text='Clear', size=(50, 50))
        self.predict_btn = Button(text='Predict', pos = (60, 0), size=(50, 50))
        self.model_btn = Button(text='Model', pos = (120, 0), size=(50, 50))

        self.painter._keyboard.bind(on_key_down=self._on_keyboard_down)

        self.class_label = Label(text='Class: ? | Proba: ?', 
        pos = (200, 0), 
        color=(1, 0, 0, 1),
        size=(300, 100),
        halign='left',
        valign='top')
        self.class_label.bind(texture_size=self.class_label.setter('size'))

        self.model_label = Label(text='KNN', 
        pos = (0, 60), 
        color=(0, 1, 0, 1),
        size=(300, 100),
        halign='left')
        self.model_label.bind(texture_size=self.model_label.setter('size'))

        self.clear_btn.bind(on_release=self.clear)
        self.predict_btn.bind(on_release=self.predict)
        self.model_btn.bind(on_release=self.change_model)

        parent.add_widget(self.painter)
        parent.add_widget(self.clear_btn)
        parent.add_widget(self.predict_btn)
        parent.add_widget(self.model_btn)
        parent.add_widget(self.class_label)
        parent.add_widget(self.model_label)
        return parent

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        # print(self.children)
        if keycode[1] == 'w':
            # увеличиваем размер кисти
            if self.painter.brush_width < 100:
                self.painter.brush_width += 1
                print('Размер кисти увеличился до %d' % (self.painter.brush_width))
            else:
                print('Достигнут наибольший размер кисти')
        elif keycode[1] == 's':
            # уменьшаем размер кисти
            if self.painter.brush_width > 1:
                self.painter.brush_width -= 1
                print('Размер кисти уменьшился до %d' % (self.painter.brush_width))
            else:
                print('Достигнут наименьший размер кисти')


        elif keycode[1] == 'q':
            # увеличиваем размер кнопок и лэйблов
            if self.clear_btn.size[0] < 200:
                self.clear_btn.size[0] += 5; self.clear_btn.size[1] += 5
                self.predict_btn.size[0] += 5; self.predict_btn.size[1] += 5
                self.model_btn.size[0] += 5; self.model_btn.size[1] += 5


                self.predict_btn.pos[0] += 5
                self.model_btn.pos[0] += 10
                self.class_label.pos[0] += 15

                self.model_label.pos[1] += 5
            else:
                print('Достигнут наибольший размер кнопок')
        elif keycode[1] == 'a':
            # уменьшаем размер кнопок и лэйблов
            if self.clear_btn.size[0] > 30:
                self.clear_btn.size[0] -= 5; self.clear_btn.size[1] -= 5
                self.predict_btn.size[0] -= 5; self.predict_btn.size[1] -= 5
                self.model_btn.size[0] -= 5; self.model_btn.size[1] -= 5


                self.predict_btn.pos[0] -= 5
                self.model_btn.pos[0] -= 10
                self.class_label.pos[0] -= 15

                self.model_label.pos[1] -= 5
            else:
                print('Достигнут наименьший размер кнопок')


        elif keycode[1] == 'c':
            # очистить
            self.clear(None)

        elif keycode[1] == 'p':
            # очистить
            self.predict(None)

        elif keycode[1] == 'm':
            # очистить
            self.change_model(None)

        return True

    def clear(self,event):
        self.class_label.text = 'Class: ? | Proba: ?'
        self.painter.canvas.clear()

    def predict(self, event):
        self.painter.export_to_png('tmp.png')
        self.class_label.text = "Class: %1d | Proba: %5.3f" % (self.classifier.predict())

    def change_model(self, event):
        global model_idx, models, model_names
        
        model_idx = (model_idx + 1) % 3
        self.classifier = models[model_idx]
        self.model_label.text = model_names[model_idx]

if __name__ == '__main__':
    MiniPaintApp().run()