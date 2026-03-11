import numpy as np

from data_io import load_crf_words
from crf_impl import objective_and_grad, unpack_params


def load_solution(path):
    """
    Load the learned parameter vector (solution.txt)
    """
    return np.loadtxt(path, dtype=float)


def main():

    # load training data
    train_words = load_crf_words("../data/train.txt")

    # load optimal parameters
    x_opt = load_solution("../result/solution.txt")

    # hyperparameter
    C = 1000.0

    # compute objective value
    obj, _ = objective_and_grad(x_opt, train_words, C)

    print("Optimal objective value =", obj)


if __name__ == "__main__":
    main()