import pygame as pg
from numpy import argmax,array
from modelClass import Model


class Menu:

    def __init__(
            self,
            sc: pg.surfarray,
            sw: int,
    ):
        pg.init()
        self.sc = sc
        self.size = sw

        self.f1 = pg.font.SysFont('arial', 50)
        self.f2 = pg.font.SysFont('arial', 30)
        self.f3 = pg.font.SysFont('arial', 20)

        self.text_menu = self.f1.render("M", True, (255, 255, 255))
        self.result = array(
            [[0 for i in range(10)]]
        )

        self.cord_res = [(452, i * 40 + 20) for i in range(1, 11)]

    def menu_borden(self, model_name: str, models: Model, step):

        self.act_model = (
            self.f2.render(model_name, True, (50, 50, 50)),
            model_name,
            (0, 0)
        )

        self.text_models = [
            (
                self.f2.render(mod, True, (100, 100, 100)),
                mod,
                (448 // 6, step * num )
            )
            for num, mod in enumerate(models, start=1)
        ]
        self.borders = [
            (
                mod,
                (448 // 6, step * num ),
                (448 // 6 + len(mod) * 18, step * num + 38)
            )
            for num, mod in enumerate(models, start=1)
        ]

    def update(self,
               text=True,
               text_models=False,
               menu=True,
               activate_model=False,
               res_model=True
               ):
        """Обновляет главное меню(создает его)
        так же отвечает за прорисовку как видит рисунок модель"""
        sw = self.size
        if menu:
            pg.draw.rect(self.sc,
                         (100, 100, 100),
                         (sw, 0, sw + 60, sw),
                         sw)
        if text:
            self.sc.blit(self.text_menu, (sw + 9, -10))

        if res_model is not None:
            self.draw_result_models()

        if text_models:
            """
            x,y = x,y 
            x2,y2 = x + len()* 18,y + 38"""
            for mod, name, (x, y) in self.text_models:
                self.sc.blit(mod, (x, y))

        if activate_model:
            """
            x1,y1 = 0,0
            x2,y2 = len() * 18, 38
            """
            mod, name, (x, y) = self.act_model
            pg.draw.rect(self.sc,
                         (255, 255, 255),
                         (x, y, 448, y + 38)
                         )

            self.sc.blit(mod, (x, y))

        pg.display.update()

    def draw_result_models(self):
        res = self.result
        n = argmax(res)
        for i in range(10):
            x, y = self.cord_res[i]
            if n == i:
                temp = self.f3.render(f"{i}-{round(float(res[0][i]), 2)}", True, (255, 0, 0))
            else:
                temp = self.f3.render(f"{i}-{round(float(res[0][i]), 2)}", True, (255, 255, 255))
            self.sc.blit(temp, (x, y))

    def choise_model(self, x, y, draw, model):
        """
        Логика если мы находимя в меню и выбираем что-то из меню
        """
        for name, (x1, y1), (x2, y2) in self.borders:
            if x1 <= x <= x2 and y1 <= y <= y2 and draw:
                self.act_model = (
                    self.f2.render(name, True, (50, 50, 50)),
                    name,
                    (0, 0))
                self.update(activate_model=True)
                model.update_model_path(name)
                draw = False
