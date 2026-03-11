import numpy as np

from data_io import load_crf_words
from extra_io import load_model_params
from crf_impl import logp_and_grad_word


def main():

    # load training dataset
    train_words = load_crf_words("../data/train.txt")

    # load provided model parameters
    W, T = load_model_params("../data/model.txt")

    total_logp = 0.0

    for w in train_words:
        logp, _, _ = logp_and_grad_word(w, W, T)
        total_logp += logp

    avg_logp = total_logp / len(train_words)

    print("Average log p(y|X) =", avg_logp)


if __name__ == "__main__":
    main()