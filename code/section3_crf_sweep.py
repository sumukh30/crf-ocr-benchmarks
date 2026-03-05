#!/usr/bin/env python3
import sys, csv
from pathlib import Path
import numpy as np
import scipy.optimize as opt

ROOT = Path(__file__).resolve().parents[1]   # Lab_1/
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT))

# If your loader file is named data_io.py, use data_io (not dataio).
from data_io import load_crf_words
from crf_impl import objective_and_grad, unpack_params, decode_words  # same as run_all.py style [file:168]

DATA = ROOT / "data"
OUTD = ROOT / "result" / "section3"
OUTD.mkdir(parents=True, exist_ok=True)

TRAIN_CRF = DATA / "train.txt"
TEST_CRF  = DATA / "test.txt"

def get_y(word):
    # Supports dict-based words and object-based words.
    if isinstance(word, dict):
        for k in ("y", "Y", "label", "labels"):
            if k in word:
                return word[k]
        raise KeyError(f"Word dict has no label key. Keys={list(word.keys())[:20]}")
    if hasattr(word, "y"):
        return word.y
    if hasattr(word, "Y"):
        return word.Y
    raise AttributeError("Word has no y/Y attribute")

def normalize_labels(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in ("U", "S", "O"):  # letters like 'a'
        return np.array([ord(str(a)) - ord("a") + 1 for a in arr], dtype=int)
    arr = arr.astype(int)
    # Some implementations store labels as 0..25; convert to 1..26 if needed.
    if arr.size and arr.min() == 0 and arr.max() <= 25:
        arr = arr + 1
    return arr

def letter_acc(a, b):
    return float(np.mean(a == b))

def word_acc(testwords, yhatwords):
    ok = 0
    for w, yh in zip(testwords, yhatwords):
        yt = normalize_labels(np.asarray(get_y(w)).reshape(-1))
        yp = normalize_labels(np.asarray(yh).reshape(-1))
        ok += int(np.all(yt == yp))
    return ok / len(testwords)

def main():
    cvals = [1, 10, 100, 1000]

    trainwords = load_crf_words(str(TRAIN_CRF))
    testwords  = load_crf_words(str(TEST_CRF))

    y_true_flat = normalize_labels(
        np.concatenate([np.asarray(get_y(w)).reshape(-1) for w in testwords])
    )

    x0 = np.zeros(128 * 26 + 26 * 26, dtype=float)  # same parameter size used in provided code pattern [file:168]

    rows = []
    for C in cvals:
        def func(x):
            return objective_and_grad(x, trainwords, float(C))  # objective+grad interface [file:168]

        xopt, nfeval, rc = opt.fmin_tnc(func=func, x0=x0, maxfun=200, ftol=1e-3, disp=0)  # as in run_all.py [file:168]
        Wopt, Topt = unpack_params(xopt)
        yhatwords = decode_words(Wopt, Topt, testwords)

        y_pred_flat = normalize_labels(
            np.concatenate([np.asarray(yh).reshape(-1) for yh in yhatwords])
        )

        la = letter_acc(y_true_flat, y_pred_flat)
        wa = word_acc(testwords, yhatwords)
        rows.append({"model": "CRF", "c": C, "letter_acc": la, "word_acc": wa})

        print(f"[CRF] C={C:>4}  letter_acc={la:.4f}  word_acc={wa:.4f}  (nfeval={nfeval}, rc={rc})")

    out_csv = OUTD / "crf.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","c","letter_acc","word_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
