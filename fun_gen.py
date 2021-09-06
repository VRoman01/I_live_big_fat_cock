#!../venv/bin/python3
import numpy as np
from matplotlib import pyplot as plt


def lin_fun_gen(n_dots: int, x_scale: tuple, k: int, y0: int, std: int):
    rng = np.random.RandomState(1)
    x = x_scale[0] +(x_scale[1]-x_scale[0]) * rng.rand(n_dots)
    y = k * x + y0 + std*rng.randn(n_dots)
    return x, y


def approx(x, y):
    n = x.size
    sum_x = x.sum()
    sum_y = y.sum()
    sum_x2 = np.square(x).sum()
    sum_xy = (x * y).sum()
    k = (n*sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - k * sum_x) / n
    return k, b


def draw(x, y, name):
    fig = plt.figure()
    axe = fig.add_subplot()
    axe.scatter(x,y)
    k, b = approx(x, y)
    y_approx = k*x + b
    axe.plot(x, y_approx)
    plt.savefig('image/{}.png'.format(name))


if __name__ == '__main__':
    x, y = lin_fun_gen(n_dots=50, x_scale=(-10, 10), k=2, y0=5, std=3)
    draw(x, y, 'first')