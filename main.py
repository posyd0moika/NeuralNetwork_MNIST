import pygame as pg
import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from os import listdir
import pygame_menu


class Interface:

    def search(self):
        """
        Функция ищет все файлы с названием "Model"
        и записывает все подходящие файлы в переменую self.models : [str]
        :return: None
        """
        self.models = [i for i in listdir() if "Model" in i]

    def __init__(self, size_window=448, model="Model_CNN_ex_128_10", fl_ci=True):
        self.sc = pg.display.set_mode((size_window, size_window))
        self.search()
        print(self.models)
        """
        :param size_window: размер окна
        :param model: название модели(путь к ней)
        :param fl_ci: True - это рисовать кругами,False - это рисовать квадратами
        (это не будет влиять на результат)
        """

        self.fl_ci = fl_ci
        self.model = keras.models.load_model(model)
        self.size_window = size_window
        pg.display.set_caption("Neural Network Interface")
        self.pool = keras.models.load_model("Pool")

        # self.menu = pygame_menu.Menu('Welcome', 400, 300,
        #                         theme=pygame_menu.themes.THEME_BLUE)

        # self.menu.add.text_input('Name :', default='John Doe')
        # self.menu.add.selector('Difficulty :', [(k,n) for n,k in enumerate(self.models)], onchange=lambda *ar:print(1))
        # self.menu.add.button('Play', self)
        # self.menu.add.button('Quit', pygame_menu.events.EXIT)

    def __call__(self):
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
                            x3 = np.array(x3)[:, :, 0].T.copy() / 255
                            x3.shape = (1, 448, 448, 1)

                            result = np.array(self.pool(x3))
                            copy_res = result.squeeze()
                            plt.imshow(result.squeeze(), cmap=plt.cm.binary)
                            plt.show()
                            for i in range(28):
                                for j in range(28):
                                    col = result[0][i][j][0] * 255
                                    rect = pg.draw.rect(self.sc,
                                                        (col, col, col),
                                                        (j * 16, i * 16,
                                                         j * 16 + 16, i * 16 + 16))
                                    pg.display.update(rect)

                            res = self.model(result)
                            n = np.argmax(res)
                            print(f"Result: {n}\nConfidence:{res[0][n]}")
                        elif event.key == pg.K_CAPSLOCK:
                            update_sc = pg.draw.circle(self.sc, (0, 0, 0), (sw // 2, sw // 2), sw)
                            pg.display.update(update_sc)
                        elif event.key == pg.K_0:
                            print(*[f"{i + 1}) {k}" for i, k in enumerate(self.models)],
                                  sep="\n")
                            while True:
                                try:
                                    n = int(input("Введите номер модели")) - 1
                                    self.model = keras.models.load_model(self.models[n])
                                    break
                                except:
                                    print("Ты оладушек")
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
    inter = Interface(fl_ci=True, model="Model_ex_64_32")
    inter()
