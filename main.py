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

    def __init__(self, model_name="Model_CNN_ex_128_10"):
        """
        :param model_name: название модели(путь к ней)
        """

        self.model_name = model_name
        self.sc = pg.display.set_mode((448 + 60, 448))
        pg.display.set_caption("Neural Network Interface")
        self.search()

        self.size_window = 448
        self.cord_res = [(452, i * 40 + 20) for i in range(1, 11)]

        self.model = keras.models.load_model(model_name)
        self.pool = keras.models.load_model("Pool")

    def __call__(self):
        pg.init()
        clock = pg.time.Clock()
        fl_draw = False
        fl_menu = False
        sw = self.size_window
        x, y = 0, 0
        f = pg.font.SysFont('arial', 50)
        f2 = pg.font.SysFont('arial', 30)
        text_menu = f.render("M", True, (255, 255, 255))
        step_model = sw // len(self.models)

        result_models = np.array(
            [[0 for i in range(10)]]
        )

        act_model = (
            f2.render(self.model_name, True, (50, 50, 50)),
            self.model_name,
            (0, 0))

        text_models = [
            (
                f2.render(mod, True, (100, 100, 100)),
                mod,
                (448 // 6, step_model * num)
            )
            for num, mod in enumerate(self.models, start=1)
        ]

        borders = [
            (
                mod,
                (448 // 6, step_model * num),
                (448 // 6 + len(mod) * 18, step_model * num + 38)
            )
            for num, mod in enumerate(self.models, start=1)
        ]

        self.update(sw, text_menu)

        while True:
            for event in pg.event.get():

                match event.type:

                    case pg.QUIT:
                        sys.exit()

                    case pg.MOUSEMOTION:
                        x, y = event.pos
                        # print(x, y)

                    case pg.MOUSEBUTTONDOWN:
                        fl_draw = True

                    case pg.MOUSEBUTTONUP:
                        fl_draw = False

                    case pg.KEYDOWN if event.key == pg.K_TAB and fl_menu is False:
                        x3 = pg.surfarray.pixels3d(self.sc)
                        print(len(x3[0]))
                        x3 = np.array(x3)[0:448, :, 0].T / 255

                        x3.shape = (1, 448, 448, 1)

                        result = np.array(self.pool(x3))
                        copy_res = result.squeeze()

                        # draw_see_model = []

                        result_models = self.model(result)
                        n = np.argmax(result_models)
                        print(f"Result: {n}\nConfidence:{result_models[0][n]}")

                        for j in range(28):
                            for i in range(28):
                                col = result[0][i][j][0] * 255
                                rect = pg.draw.rect(self.sc,
                                                    (col, col, col),
                                                    (j * 16, i * 16,
                                                     j * 16 + 16, i * 16 + 16))

                        self.update(sw, text_menu,res_model=result_models)
                        # self.draw_result_models(f2,result_models)

                    case pg.KEYDOWN if event.key == pg.K_CAPSLOCK and fl_menu is False:
                        update_sc = pg.draw.rect(self.sc, (0, 0, 0),
                                                 (0, 0, sw, sw),
                                                 sw)
                        pg.display.update(update_sc)

                    case pg.KEYDOWN if event.key == pg.K_0 and fl_menu is False:
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

            if fl_draw and x < sw - 15 and not fl_menu:
                circle1 = pg.draw.circle(self.sc, (255, 255, 255), (x, y), sw * 0.036)
                pg.display.update(circle1)

            elif fl_draw and 455 <= x <= 510 and 0 <= y <= 40:
                col_menu = (255, 255, 255) if fl_menu is False else (0, 0, 0)

                pg.draw.rect(self.sc, col_menu,
                             (0, 0, sw, sw),
                             sw)
                if fl_menu is False:
                    self.update(sw, menu=False, text_models=text_models, activate_model=act_model)

                pg.display.update()
                fl_menu = not fl_menu
                fl_draw = False

            if fl_menu is True:
                """
                Логика если мы находимя в меню и выбираем что-то из меню
                """
                for name, (x1, y1), (x2, y2) in borders:
                    if x1 <= x <= x2 and y1 <= y <= y2 and fl_draw:
                        self.model_name = name
                        act_model = (
                            f2.render(self.model_name, True, (50, 50, 50)),
                            self.model_name,
                            (0, 0))
                        self.update(sw, activate_model=act_model, text=text_menu)
                        self.model = keras.models.load_model(self.model_name)
            # self.update(sw, text_menu, res_model=result_models)

        clock.tick(120)

    def update(self, sw, text=None, text_models=None, menu=True, activate_model=None, res_model = None):
        """Обновляет главное меню(создает его)
        так же отвечает за прорисовку как видит рисунок модель"""

        if menu:
            pg.draw.rect(self.sc,
                         (100, 100, 100),
                         (sw, 0, sw + 60, sw),
                         sw)
        if text:
            self.sc.blit(text, (sw + 9, -10))

        if res_model is not None:
            f = pg.font.SysFont('arial', 20)
            self.draw_result_models(f,res_model)

        if text_models:
            """
            x,y = x,y 
            x2,y2 = x + len()* 18,y + 38"""
            for mod, name, (x, y) in text_models:
                self.sc.blit(mod, (x, y))

        if activate_model:
            """
            x1,y1 = 0,0
            x2,y2 = len() * 18, 38
            """
            mod, name, (x, y) = activate_model
            pg.draw.rect(self.sc,
                         (255, 255, 255),
                         (x, y, 448, y + 38)
                         )

            self.sc.blit(mod, (x, y))

        pg.display.update()

    def draw_result_models(self, f, res: np.array):
        n = np.argmax(res)
        for i in range(10):
            x, y = self.cord_res[i]
            if n == i:
                temp = f.render(f"{i}-{round(float(res[0][i]), 2)}", True, (255, 0, 0))
            else:
                temp = f.render(f"{i}-{round(float(res[0][i]),2)}", True, (255, 255, 255))
            self.sc.blit(temp, (x, y))


if __name__ == '__main__':
    inter = Interface(model_name="Model_ex_64_32")
    inter()
