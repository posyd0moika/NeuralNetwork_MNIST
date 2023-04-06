from tensorflow.keras.models import load_model
import numpy as np
from os import listdir


class Model:
    def __init__(self, path=None):

        self.update_model_path(path)
        self.pool = load_model("Pool")

    def update_model_path(self, new_path: str):
        self.search()
        if new_path in self.models:
            self.model = load_model(new_path)
            self.path = new_path
        elif new_path is None:
            self.model = load_model(self.models[0])
            self.path = self.models[0]

        else:
            raise ValueError(
                f"Не найден файл {new_path} в ближайшем окружении\n" +
                f"File {new_path} not found in immediate environment"
            )

    def __call__(self, imag: np.array, fl_print: bool = False) -> np.array:
        """

        :param imag: np.array ,shape = (448,448)
        :param fl_print:
            if fl_print:
                n = np.argmax(self.result_models)
                print(f"Result: {n}\nConfidence:{self.result_models[0][n]}")

        :return:result models : np.array, shape = (1,10)
        """
        imag.shape = (1, 448, 448, 1)

        pool_imag = np.array(self.pool(imag))

        self.result_models = self.model(pool_imag)
        if fl_print:
            n = np.argmax(self.result_models)
            print(f"Result: {n}\nConfidence:{self.result_models[0][n]}")

        return self.result_models, pool_imag

    def search(self, fl_return: bool = False):
        """
        Функция ищет все файлы с названием "Model"
        и записывает все подходящие файлы в переменую self.models : [str]
        :param fl_return
            if fl_return is True:
                :return: List[str] models
            else:
                :return None
        """
        self.models = sorted([i for i in listdir() if "Model" in i])

        return self.models if fl_return is True else None

    def __len__(self):
        return len(self.models)

    def __str__(self):
        return self.path
