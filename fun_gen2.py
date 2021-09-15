#!../venv/bin/python
import numpy as np
from matplotlib import pyplot as plt
from fun_gen import gradient_descent


def power2_lin_fun_gen(x_scale, n_sectors, n_sectors_dots, w, std_y, std_x):
    rng = np.random.RandomState(1)
    x = np.array([scale[0] + (scale[1] - scale[0]) * rng.rand(n_sectors) for scale in x_scale])
    x = np.repeat(x, n_sectors_dots, axis=1) + std_x*np.random.randn(len(x_scale), n_sectors*n_sectors_dots)
    X = np.array([np.ones(n_sectors*n_sectors_dots), *x]).T
    Y = np.matmul(X, np.array(w)) + std_y * np.random.randn(n_sectors*n_sectors_dots)
    return X, Y


def analytical_solution(X, Y):
    try:
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return w
    except np.linalg.LinAlgError as e:
        print(e)


if __name__ == '__main__':
    inf = {'n_sectors': 100,
           'n_sectors_dots': 10,
           'x_scale': ((-10, 10), (-5, 5)),
           'std_y': 0.3,
           'std_x': 0.3,
           'w': [5, 3, 2],
           }
    X, Y = power2_lin_fun_gen(**inf)

    w = np.round(analytical_solution(X, Y), 3)
    inf['w_p'] = list(w)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X.T[1], X.T[2], Y, marker='o', label=str(inf))
    ax.legend()
    plt.show()

