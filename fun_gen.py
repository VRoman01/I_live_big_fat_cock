#!../venv/bin/python3
import numpy as np
from matplotlib import pyplot as plt


def approx(x, y):
    n = x.size
    sum_x = x.sum()
    sum_y = y.sum()
    sum_x2 = np.square(x).sum()
    sum_xy = (x * y).sum()
    k = (n*sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - k * sum_x) / n
    return k, b


def lin_fun_gen(n_dots: int, x_scale: tuple, k: int, b: int, std: int):
    rng = np.random.RandomState(1)
    x = x_scale[0] +(x_scale[1]-x_scale[0]) * rng.rand(n_dots)
    y = k * x + b + std*rng.randn(n_dots)
    return x, y


def draw(x, y, name, inf):
    fig = plt.figure()
    axe = fig.add_subplot()
    axe.scatter(x, y, c='g', label=str(inf))
    k, b = approx(x, y)
    y_approx = k*x + b
    axe.plot(x, y_approx, c='r', label='k={:2.1f} b={:2.1f}'.format(k, b))
    axe.legend()
    # plt.savefig('image/{}'.format(name))
    plt.show()


def power_lin_fun_gen(n_dots: int, x_scale: tuple, w, std: int):
    rng = np.random.RandomState(1)
    # Training data
    x = x_scale[0] + (x_scale[1] - x_scale[0]) * rng.rand(n_dots)
    X_train = np.array([np.ones(n_dots), x]).T
    Y_train = np.matmul(X_train,np.array(w)) + std*rng.randn(n_dots)
    # Test data
    x = np.arange(x_scale[0], x_scale[1], (x_scale[1]-x_scale[0])/n_dots)
    X_test = np.array([np.ones(n_dots), x]).T
    Y_test = np.matmul(X_test, np.array(w))
    return X_train, Y_train, X_test, Y_test


def analytical_solution(X, Y):
    try:
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return w
    except np.linalg.LinAlgError as e:
        print(e)


def power_draw(X, Y, name, inf):
    fig = plt.figure()
    axe = fig.add_subplot()
    axe.scatter(X.T[1], Y, c='g', label=str(inf))
    k, b = approx(X.T[1], Y)
    y_approx = k * X.T[1] + b
    axe.plot(X.T[1], y_approx, c='r', label='k={:2.1f} b={:2.1f}'.format(k, b))
    axe.legend()
    # plt.savefig('image/{}'.format(name))
    plt.show()


if __name__ == '__main__':
    inf = {'n_dots': 50,
           'x_scale': (-10, 10),
           'w': [5, 3],
           'std': 3,
           }
    X_train, Y_train, X_test, Y_test = power_lin_fun_gen(**inf)
    # power_draw(X_train, Y_train, name='second', inf=inf)
    w = analytical_solution(X_train, Y_train)
    print(w)


    # inf = {'n_dots': 50, 'x_scale': (-10, 10), 'k': 2, 'b': 5, 'std': 3}
    # x, y = lin_fun_gen(**inf)
    # draw(x, y, 'first', inf)
