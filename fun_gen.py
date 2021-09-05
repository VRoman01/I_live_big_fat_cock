#!../venv/bin/python3
import numpy as np
from matplotlib import pyplot as plt


def lin_fun_gen(n_dots: int, x_scale: int, k: int, y0: int):
    rng = np.random.RandomState(1)
    x = x_scale * rng.rand(n_dots)
    y = k * x + y0 + rng.randn(n_dots)
    return x, y


def drew(x, y):
    fig = plt.figure()
    axe = fig.add_subplot()
    axe.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    x, y = lin_fun_gen(50, 10, 2, 5)
    drew(x, y)
