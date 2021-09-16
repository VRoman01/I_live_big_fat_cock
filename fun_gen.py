#!../venv/bin/python3
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


def power_lin_fun_gen(n_dots: int, x_scale: tuple, w, std: int):
    rng = np.random.RandomState(1)
    # Training data
    x = x_scale[0] + (x_scale[1] - x_scale[0]) * rng.rand(n_dots)
    X_train = np.array([np.ones(n_dots), x]).T
    Y_train = np.matmul(X_train,np.array(w)) + std*rng.randn(n_dots)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

    return X_train, Y_train


def analytical_solution(X, Y):
    try:
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return w
    except np.linalg.LinAlgError as e:
        print(e)


def gradient_descent(X, Y):
    n_epochs = 10
    lr = 1e-2
    w_list = []
    rng = np.random.RandomState(1)
    w = rng.rand(2)
    w_list.append(w)

    for epoch in range(n_epochs):
        X, Y = shuffle(X, Y, random_state=0)

        X_list = np.array_split(X, 10)
        Y_list = np.array_split(Y, 10)
        w = w_list[-1]

        v0_p, v1_p = 0, 0

        for x, y in zip(X_list, Y_list):
            yhat = x @ w

            error = (y - yhat)

            w0_grad = -2 * error.mean()
            w1_grad = -2 * (x.T[1] * error).mean()
            v0 = 0.9 * v0_p + lr * w0_grad
            v1 = 0.9 * v1_p + lr * w1_grad

            w[0] = w[0] - v0
            w[1] = w[1] - v1

            v0_p = v0
            v1_p = v1

        w_list.append(w)

    return w_list


def R2(Y, Y_predict):
    r_value = 1 - np.sum((Y-Y_predict)**2)/np.sum((Y-np.mean(Y))**2)
    return r_value


if __name__ == '__main__':
    inf = {'n_dots': 1000,
           'x_scale': (-10, 10),
           'w': [5, 3],
           'std': 0.5,
           }
    X, Y = power_lin_fun_gen(**inf)

    X_train, Y_train = X[:700], Y[:700]
    X_valid, Y_valid = X[700:900], Y[700:900]
    X_test, Y_test = X[900:], Y[900:]

    print(gradient_descent(X_train, Y_train))





