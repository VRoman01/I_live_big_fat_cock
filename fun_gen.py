#!../venv/bin/python3
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


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

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    return X_train, Y_train, X_test, Y_test


def analytical_solution(X, Y):
    try:
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return w
    except np.linalg.LinAlgError as e:
        print(e)


def gradient_descent(X, Y):
    w_list = []
    X_list = np.array_split(X, 10)
    Y_list = np.array_split(Y, 10)

    rng = np.random.RandomState(1)
    for X, Y in zip(X_list, Y_list):
        w = rng.rand(2)

        lr = 1e-3
        n_epochs = 1000

        for epoch in range(n_epochs):
            yhat = X@w

            error = (Y - yhat)

            w0_grad = -2 * error.mean()
            w1_grad = -2 * (X.T[1] * error).mean()

            w[0] = w[0] - lr * w0_grad
            w[1] = w[1] - lr * w1_grad

        w_list.append(w)

    return w_list


def power_draw(X, Y, w, name, inf):
    fig = plt.figure()
    axe = fig.add_subplot()
    axe.scatter(X.T[1], Y, c='g', label=str(inf))
    y_approx = w[1] * X.T[1] + w[0]
    axe.plot(X.T[1], y_approx, c='r', label='w0={:2.1f} w1={:2.1f}'.format(w[0], w[1]))
    axe.legend()
    # plt.savefig('image/{}'.format(name))
    plt.show()


def R2(Y, Y_predict):
    r_value = 1 - np.sum((Y-Y_predict)**2)/np.sum((Y-np.mean(Y))**2)
    return r_value


if __name__ == '__main__':
    inf = {'n_dots': 1000,
           'x_scale': (-10, 10),
           'w': [5, 3],
           'std': 0.5,
           }
    X_train, Y_train, X_test, Y_test = power_lin_fun_gen(**inf)

    w_gradient_list = gradient_descent(X_train,Y_train)
    for w_g in w_gradient_list:
        print(R2(Y_test, X_test@w_g))

    w_analytical = analytical_solution(X_train,Y_train)
    print('R2_a = ',R2(Y_test, X_test@w_analytical))


