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
        self.save_result = False

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

                    case pg.KEYDOWN if event.key == pg.K_1 and fl_menu is False and self.save_result is not False:
                        self.menu.result = self.save_result
                        for j in range(28):
                            for i in range(28):
                                col = self.save_img[0][i][j][0] * 255
                                pg.draw.rect(self.sc,
                                             (col, col, col),
                                             (
                                                 j * 16, i * 16,
                                                 j * 16 + 16, i * 16 + 16
                                             )
                                             )


                    case pg.KEYDOWN if event.key == pg.K_TAB and fl_menu is False:
                        self.update_result(paint=False, fl_save=True)

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
                if fl_menu is False:
                    self.update_result(paint=False, fl_save=True)

                pg.draw.rect(self.sc, col_menu,
                             (0, 0, sw, sw),
                             sw)
                if fl_menu is False:
                    self.menu.update(menu=False, text_models=True, activate_model=True)
                else:
                    self.menu.result = self.save_result
                    for j in range(28):
                        for i in range(28):
                            col = self.save_img[0][i][j][0] * 255
                            pg.draw.rect(self.sc,
                                         (col, col, col),
                                         (
                                             j * 16, i * 16,
                                             j * 16 + 16, i * 16 + 16
                                         )
                                         )
                    self.menu.update(menu=False)

                fl_menu = not fl_menu
                fl_draw = False

            if fl_menu is True:
                self.menu.choise_model(x, y, fl_draw, self.model)
            else:
                self.update_result()

        clock.tick(120)

    def update_result(self, paint: bool = True,fl_save: bool=False):
        img = pg.surfarray.pixels3d(self.sc)
        img = np.array(img)[0:448, :, 0].T / 255
        result, img = self.model(img)

        if fl_save:
            self.save_img = img
            self.save_result = result

        if paint is True:
            for j in range(28):
                for i in range(28):
                    col = img[0][i][j][0] * 255
                    pg.draw.rect(self.sc,
                                 (col, col, col),
                                 (
                                     j * 16, i * 16,
                                     j * 16 + 16, i * 16 + 16
                                 )
                                 )

        self.menu.result = result
        self.menu.update()




if __name__ == '__main__':
    inter = Interface()
    inter.main_loop()
