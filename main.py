import pygame as pg
import numpy as np
from tensorflow import keras
from os import listdir
import sys
from time import time


class Interface:

    def search(self):
        """
        Функция ищет все файлы с названием "Model"
        и записывает все подходящие файлы в переменую self.models : [str]
        :return: None
        """
        self.models = [i for i in listdir() if "Model" in i]

    def __init__(self, model="Model_CNN_ex_128_10"):
        """
        :param model: название модели(путь к ней)
        """

        self.sc = pg.display.set_mode((448 + 60, 448))
        pg.display.set_caption("Neural Network Interface")
        self.search()

        self.size_window = 448

        self.model = keras.models.load_model(model)
        self.pool = keras.models.load_model("Pool")

    def __call__(self):
        pg.init()
        clock = pg.time.Clock()
        fl_draw = False
        fl_menu = False
        sw = self.size_window
        x, y = 0, 0

        f = pg.font.SysFont('arial', 50)
        text_menu = f.render("M", True, (255, 255, 255))

        self.update(sw, text_menu)

        while True:
            for event in pg.event.get():

                match event.type:
                    case pg.QUIT:
                        sys.exit()
                    case pg.MOUSEMOTION:
                        x, y = event.pos
                    case pg.MOUSEBUTTONDOWN:
                        fl_draw = True
                    case pg.MOUSEBUTTONUP:
                        fl_draw = False
                    case pg.KEYDOWN if fl_menu is False:
                        if event.key == pg.K_TAB:
                            x3 = pg.surfarray.pixels3d(self.sc)
                            print(len(x3[0]))
                            x3 = np.array(x3)[0:448, :, 0].T / 255

                            x3.shape = (1, 448, 448, 1)

                            result = np.array(self.pool(x3))
                            copy_res = result.squeeze()

                            draw_see_model = []

                            res = self.model(result)
                            n = np.argmax(res)
                            print(f"Result: {n}\nConfidence:{res[0][n]}")

                            for j in range(28):
                                for i in range(28):
                                    col = result[0][i][j][0] * 255
                                    rect = pg.draw.rect(self.sc,
                                                        (col, col, col),
                                                        (j * 16, i * 16,
                                                         j * 16 + 16, i * 16 + 16))
                                    draw_see_model.append(rect)

                            self.update(sw, text_menu, draw_see_model)

                            res = self.model(result)
                            n = np.argmax(res)
                            print(f"Result: {n}\nConfidence:{res[0][n]}")

                        elif event.key == pg.K_CAPSLOCK:
                            update_sc = pg.draw.rect(self.sc, (0, 0, 0),
                                                     (0, 0, sw, sw),
                                                     sw)
                            pg.display.update(update_sc)
                        elif event.key == pg.K_0:
                            print(
                                *[f"{i + 1}) {k}" for i, k in enumerate(self.models)],
                                sep="\n"
                            )
                            while True:
                                try:
                                    n = int(input("Введите номер модели")) - 1
                                    self.model = keras.models.load_model(self.models[n])
                                    break
                                except:
                                    print("Ты оладушек ")

            if fl_draw and x < sw - 15:
                circle1 = pg.draw.circle(self.sc, (255, 255, 255), (x, y), sw * 0.036)
                pg.display.update(circle1)

            elif fl_draw and 455 <= x <= 510 and 0 <= y <= 40:
                col_menu = (255, 255, 255) if fl_menu is False else (0, 0, 0)

                update_menu = pg.draw.rect(self.sc, col_menu,
                                           (0, 0, sw, sw),
                                           sw)
                pg.display.update(update_menu)
                fl_menu = not fl_menu
                fl_draw = False

        clock.tick(120)

    def update(self, sw, text=None, draw_see_model=None):

        menu = pg.draw.rect(self.sc,
                            (100, 100, 100),
                            (sw, 0, sw + 60, sw),
                            sw)

        self.sc.blit(text, (sw + 9, -10))

        # if draw_see_model:
        #     [pg.display.update(rect) for rect in draw_see_model]
        # else:
        pg.display.update()

    def draw_result(self):
        """Отрисовывает результат нейронной сети"""


if __name__ == '__main__':
    inter = Interface(model="Model_ex_64_32")
    inter()
