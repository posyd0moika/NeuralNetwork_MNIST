from Menu import *
import sys


class Interface:

    def __init__(self, model_name="Model_CNN_ex_128_10"):
        """
        :param model_name: название модели(путь к ней)
        """

        self.sc = pg.display.set_mode((448 + 60, 448))
        pg.display.set_caption("Neural Network Interface")
        self.size_window = 448

        self.model = Model(model_name)
        self.menu = Menu(self.sc, self.size_window)

    def main_loop(self):
        clock = pg.time.Clock()
        fl_draw = False
        fl_menu = False
        sw = self.size_window
        x, y = 0, 0

        step = sw // len(self.model)
        self.menu.menu_borden(
            str(self.model),
            self.model.models,
            step
        )
        self.menu.update()

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
                        x3 = np.array(x3)[0:448, :, 0].T / 255
                        result, img = self.model(x3)

                        for j in range(28):
                            for i in range(28):
                                col = img[0][i][j][0] * 255
                                pg.draw.rect(self.sc,
                                             (col, col, col),
                                             (j * 16, i * 16,
                                              j * 16 + 16, i * 16 + 16
                                              )
                                             )

                        self.menu.result = result
                        self.menu.update()

                    case pg.KEYDOWN if event.key == pg.K_CAPSLOCK and fl_menu is False:
                        update_sc = pg.draw.rect(self.sc, (0, 0, 0),
                                                 (0, 0, sw, sw),
                                                 sw)
                        pg.display.update(update_sc)

            if fl_draw and x < sw - 15 and not fl_menu:
                circle1 = pg.draw.circle(self.sc, (255, 255, 255), (x, y), sw * 0.036)
                pg.display.update(circle1)

            elif fl_draw and 455 <= x <= 510 and 0 <= y <= 40:
                col_menu = (255, 255, 255) if fl_menu is False else (0, 0, 0)

                pg.draw.rect(self.sc, col_menu,
                             (0, 0, sw, sw),
                             sw)
                if fl_menu is False:
                    self.menu.update(menu=False, text_models=True, activate_model=True)
                else:
                    self.menu.update(menu=False)

                fl_menu = not fl_menu
                fl_draw = False

            if fl_menu is True:
                self.menu.choise_model(x, y, fl_draw, self.model)

        clock.tick(120)


if __name__ == '__main__':
    inter = Interface()
    inter.main_loop()
