#This file generates the answers for questions 1c, 2a and 2b

import os
import numpy as np
import scipy.optimize as opt

from data_io import load_crf_words
from crf_impl import (
    node_scores, viterbi, logp_and_grad_word,
    objective_and_grad, unpack_params, pack_params, decode_words
)
from extra_io import load_model_params, load_decode_input


#np.seterr(all="raise")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_column_vector(path, v):
    v = np.asarray(v, dtype=float).reshape(-1)
    with open(path, "w") as f:
        for x in v:
            f.write(f"{x:.18e}\n")


def write_int_lines(path, ints_1based):
    with open(path, "w") as f:
        for a in ints_1based:
            f.write(f"{int(a)}\n")


def main():
    result_dir = "../result"
    ensure_dir(result_dir)

    # ---------- (1) decode_output.txt ----------
    # The PDF decode_input is one word with 100 letters plus parameters.      
    X100, Wd, Td = load_decode_input("../data/decode_input.txt", m=100)
    # def stats(name, A):
    #     A = np.asarray(A)
    #     print(name, "shape=", A.shape, "dtype=", A.dtype,
    #         "finite=", np.isfinite(A).all(),
    #         "min=", np.nanmin(A), "max=", np.nanmax(A),
    #         "absmax=", np.nanmax(np.abs(A)))

    # stats("X100", X100)
    # stats("Wd", Wd)
    # print("Any nonfinite in X100:", np.any(~np.isfinite(X100)))
    # print("Any nonfinite in Wd:",  np.any(~np.isfinite(Wd)))

    U100 = node_scores(X100, Wd)
    yhat, best_obj = viterbi(U100, Td)
    write_int_lines(os.path.join(result_dir, "decode_output.txt"), yhat + 1)
    print("decode_output.txt written; best objective =", best_obj)

    # ---------- Load train/test ----------
    train_words = load_crf_words("../data/train.txt")
    test_words = load_crf_words("../data/test.txt")

    # ---------- (2) gradient.txt at model.txt ----------
    # Required: average gradient of log p(y|X) over the training set, in model vector order.      
    Wm, Tm = load_model_params("../data/model.txt")
    sum_gW = np.zeros_like(Wm)
    sum_gT = np.zeros_like(Tm)
    sum_logp = 0.0

    for w in train_words:
        lp, gW, gT = logp_and_grad_word(w, Wm, Tm)
        sum_logp += lp
        sum_gW += gW
        sum_gT += gT

    avg_gW = sum_gW / len(train_words)
    avg_gT = sum_gT / len(train_words)

    grad_vec = pack_params(avg_gW, avg_gT)
    write_column_vector(os.path.join(result_dir, "gradient.txt"), grad_vec)
    print("gradient.txt written; avg logp =", (sum_logp / len(train_words)))

    # ---------- (3) solution.txt via LBFGS (fmin_tnc), C=1000 ----------
    C = 1000.0  # specified in training section    1  
    x0 = np.zeros(128 * 26 + 26 * 26, dtype=float)

    def func(x):
        return objective_and_grad(x, train_words, C)

    x_opt, nfeval, rc = opt.fmin_tnc(func=func, x0=x0, maxfun=200, ftol=1e-3, disp=5)
    write_column_vector(os.path.join(result_dir, "solution.txt"), x_opt)
    print("solution.txt written; nfeval =", nfeval, "rc =", rc)

    # ---------- (4) prediction.txt ----------
    # Required: predicted label for each test letter in the same order as in test.txt.      
    Wopt, Topt = unpack_params(x_opt)
    yhat_words = decode_words(Wopt, Topt, test_words)
    yhat_flat_1based = np.concatenate(  yh + 1 for yh in yhat_words  )
    write_int_lines(os.path.join(result_dir, "prediction.txt"), yhat_flat_1based)
    print("prediction.txt written; num test letters =", yhat_flat_1based.size)


if __name__ == "__main__":
    main()
