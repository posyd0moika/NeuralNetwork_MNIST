import numpy as np
from tensorflow import keras


def extension_base(x, y):
    if len(x) != len(y):
        raise ValueError("Длинна параметров разная")
    new_xbase = np.array([np.zeros(shape=(28, 28), dtype=float) for i in range(len(x) * 77)])
    new_ybase = np.array([0 for i in range(len(y) * 77)])

    g = 0
    for k in range(len(x)):
        for i in range(7):
            for j in range(11):
                new_xbase[g][i:21 + i, j:17 + j] = x[k][4:25, 5:22]
                new_ybase[g] = y[k]
                g += 1

    return new_xbase, new_ybase


def gen_extension_base(x, y, size_drop=100):
    """
    :param x: тренировочная выборка - не раздеенная на 255
    :param y: результат тренировочной выборки не разбита на вектор shape = (10)
    :param size_drop: сколько партий на получение данных (len(x) / 100) * 77 такое количество элементов
    """

    x = x / 255.0

    while True:
        t = 0
        for i in range(len(x) // size_drop, len(x), len(x) // size_drop):
            new_x, new_y = extension_base(x[t:i], y[t:i])
            new_y = keras.utils.to_categorical(new_y, 10)
            yield new_x, new_y
            t = i
