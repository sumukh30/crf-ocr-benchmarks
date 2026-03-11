import numpy as np
from scipy.optimize import check_grad

from crf_impl import objective_and_grad, pack_params


def create_small_dataset():

    np.random.seed(0)

    m = 4
    D = 128
    K = 26

    # slightly larger feature scale
    X = 0.2 * np.random.randn(m, D)
    y = np.random.randint(0, K, size=m)

    return [{"X": X, "y": y}]


def main():

    np.random.seed(0)

    word_list = create_small_dataset()

    D = 128
    K = 26

    # increase parameter magnitude
    W = 0.2 * np.random.randn(D, K)
    T = 0.2 * np.random.randn(K, K)

    x = pack_params(W, T)

    C = 1.0

    def func(x):
        f, _ = objective_and_grad(x, word_list, C)
        return f

    def grad(x):
        _, g = objective_and_grad(x, word_list, C)
        return g

    error = check_grad(func, grad, x)

    print("Gradient check error =", error)


if __name__ == "__main__":
    main()