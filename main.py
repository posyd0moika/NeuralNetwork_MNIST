import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

fl_ci = True


def interface():
    size_window = 476
    sc = pg.display.set_mode((size_window, size_window))
    pg.display.set_caption("Neural Network Interface")
    clock = pg.time.Clock()
    flruning = True
    flkey = False
    fldraw = False
    x, y = 0, 0
    result = []
    model = keras.models.load_model("Model_CNN_ex_128_10")

    while flruning:
        for event in pg.event.get():

            match event.type:
                case pg.QUIT:
                    pg.quit()
                    flruning = False
                case pg.MOUSEMOTION:
                    # print(event.pos)
                    x, y = event.pos
                case pg.MOUSEBUTTONDOWN:
                    fldraw = True
                case pg.MOUSEBUTTONUP:
                    fldraw = False
                case pg.KEYDOWN:
                    if event.key == pg.K_TAB:
                        x3 = pg.surfarray.pixels3d(sc)
                        result = []
                        temp = size_window // 28
                        for i in range(0, size_window, temp):
                            for j in range(0, size_window, temp):
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
                        res = model(result)

                        for i in range(len(res[0])):
                            if max(res[0]) == res[0][i]:
                                print(f"Result: {i}\nConfidence:{res.numpy()[0][i]}")
                                break

                    elif event.key == pg.K_CAPSLOCK:
                        update_sc = pg.draw.circle(sc, (0, 0, 0), (size_window // 2, size_window // 2), size_window)
                        pg.display.update(update_sc)

        if fldraw:
            if fl_ci:
                circle1 = pg.draw.circle(sc, (255, 255, 255), (x, y), size_window * 0.036)
                pg.display.update(circle1)
            else:
                rect = pg.draw.rect(sc,
                                    (255, 255, 255),
                                    (x // 28 * 28, y // 28 * 28, size_window / (size_window // 28),
                                     size_window / (size_window // 28)))
                pg.display.update(rect)

        clock.tick(120)


if __name__ == '__main__':
    interface()
