import pygame as pg
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pygame_menu


class Interface:

    def search(self):
        """
        Функция ищет все файлы с названием "Model"
        и записывает все подходящие файлы в переменую self.models : [str]
        :return: None
        """
        pass

    def __init__(self, size_window=448, model="Model_CNN_ex_128_10", fl_ci=True):
        pg.init()
        self.search()
        """
        :param size_window: размер окна
        :param model: название модели(путь к ней)
        :param fl_ci: True - это рисовать кругами,False - это рисовать квадратами
        (это не будет влиять на результат)
        """

        self.fl_ci = fl_ci
        self.model = keras.models.load_model(model)
        self.size_window = size_window
        self.sc = pg.display.set_mode((size_window, size_window))
        pg.display.set_caption("Neural Network Interface")
        # self.menu =

    def __call__(self):
        # self.menu.mainloop(self.sc)
        clock = pg.time.Clock()
        fl_draw = False
        sw = self.size_window
        x, y = 0, 0
        while True:
            for event in pg.event.get():

                match event.type:
                    case pg.QUIT:
                        pg.quit()
                        return
                    case pg.MOUSEMOTION:
                        x, y = event.pos
                    case pg.MOUSEBUTTONDOWN:
                        fl_draw = True
                    case pg.MOUSEBUTTONUP:
                        fl_draw = False
                    case pg.KEYDOWN:
                        if event.key == pg.K_TAB:
                            x3 = pg.surfarray.pixels3d(self.sc)
                            result = []
                            temp = sw // 28
                            for i in range(0, sw, temp):
                                for j in range(0, sw, temp):
                                    summ = 0

                                    for i1 in range(i, i + temp):
                                        for j2 in range(j, j + temp):
                                            summ += x3[i1][j2][0] / 255
                                    summ = x3[i][j][0] / 255
                                    result.append(summ)

                            result = np.ndarray((28, 28), buffer=np.array(result[::-1]),
                                                dtype=float)
                            result = np.rot90(result, k=1, axes=(0, 1))
                            result = np.fliplr(result)
                            result = tf.reshape(tf.cast(result, tf.float32), [1, 28, 28, 1])
                            res = self.model(result)

                            for i in range(len(res[0])):
                                if max(res[0]) == res[0][i]:
                                    print(f"Result: {i}\nConfidence:{res.numpy()[0][i]}")
                                    break

                        elif event.key == pg.K_CAPSLOCK:
                            update_sc = pg.draw.circle(self.sc, (0, 0, 0), (sw // 2, sw // 2), sw)
                            pg.display.update(update_sc)

            if fl_draw and self.fl_ci:
                circle1 = pg.draw.circle(self.sc, (255, 255, 255), (x, y), sw * 0.036)
                pg.display.update(circle1)
            elif fl_draw and not self.fl_ci:
                rect = pg.draw.rect(self.sc,
                                    (255, 255, 255),
                                    (x // 28 * 28, y // 28 * 28, sw / (sw // 28) - 1,
                                     sw / (sw // 28) - 1))
                pg.display.update(rect)

        clock.tick(120)


if __name__ == '__main__':
    inter = Interface()
    inter()
