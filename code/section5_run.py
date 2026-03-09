# code/section5_run.py
from __future__ import annotations

import os
import time
import numpy as np
import scipy.optimize as opt

from data_io import load_crf_words
from crf_impl import objective_and_grad, unpack_params, decode_words

from liblinear.liblinearutil import train as ll_train
from liblinear.liblinearutil import predict as ll_predict

from transform_utils import (
    parse_transform_file,
    apply_transforms_to_words,
    decode_accuracy_letterwise,
    decode_accuracy_wordwise,
)

np.seterr(all="raise")


def ensuredir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_data_file(data_dir: str, candidates: list[str]) -> str:
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these exist under {data_dir}: {candidates}")


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def _get_X(word):
    if hasattr(word, "X"):
        return word.X
    if isinstance(word, dict) and "X" in word:
        return word["X"]
    raise TypeError("Word must have .X or ['X'].")


def _get_y(word):
    if hasattr(word, "y"):
        return word.y
    if isinstance(word, dict) and "y" in word:
        return word["y"]
    raise TypeError("Word must have .y or ['y'].")


def _ensure_letters_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D X, got {X.shape}")
    if X.shape[1] == 128:
        return X
    if X.shape[0] == 128:
        return X.T
    raise ValueError(f"X must be (m,128) or (128,m); got {X.shape}")


def words_to_letters(words):
    Xs = []
    ys = []
    lens = []
    for w in words:
        X = _ensure_letters_rows(_get_X(w))
        y = np.asarray(_get_y(w)).reshape(-1)
        if y.size > 0 and y.min() >= 1 and y.max() <= 26:
            y = y - 1  # normalize to 0..25
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatch letters between X and y")
        Xs.append(X.astype(np.float32))
        ys.append(y.astype(int))
        lens.append(int(y.shape[0]))
    Xall = np.vstack(Xs) if Xs else np.zeros((0, 128), dtype=np.float32)
    yall = np.concatenate(ys) if ys else np.zeros((0,), dtype=int)
    return Xall, yall, lens


def dense_to_liblinear_x(X: np.ndarray):
    # liblinearutil expects list of dicts with 1-based feature indices
    X = np.asarray(X)
    out = []
    for i in range(X.shape[0]):
        row = X[i, :]
        nz = np.flatnonzero(row)
        d = {int(j) + 1: float(row[j]) for j in nz}
        out.append(d)
    return out


def wordwise_accuracy_from_flat(y_pred: np.ndarray, y_true: np.ndarray, lens: list[int]) -> float:
    pos = 0
    correct = 0
    for L in lens:
        if L == 0:
            correct += 1
            continue
        if np.all(y_pred[pos:pos + L] == y_true[pos:pos + L]):
            correct += 1
        pos += L
    return correct / max(1, len(lens))


def main():
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "..", "data")
    result_dir = os.path.join(base, "..", "result", "section5")
    # os.makedirs(result_dir, exist_ok=True)
    ensuredir(result_dir)

    train_path = resolve_data_file(data_dir, ["datatrain.txt", "train.txt"])
    test_path = resolve_data_file(data_dir, ["datatest.txt", "test.txt"])
    transform_path = resolve_data_file(data_dir, ["datatransform.txt", "transform.txt"])

    print("Loading words...")
    train_words = load_crf_words(train_path)
    test_words = load_crf_words(test_path)
    transforms = parse_transform_file(transform_path)

    # Required x values for Section 5. [0,500,1000,1500,2000]
    x_values = [0, 500, 1000, 1500, 2000]

    # Use the "best C" you found in Section 3 for each model.
    C_crf = 1000.0
    C_svm = 1000.0

    # ----- Precompute test letters once (test set is never distorted) -----
    Xte, yte0, test_lens = words_to_letters(test_words)
    yte = (yte0 + 1).astype(int)  # liblinear labels: 1..26
    xte_ll = dense_to_liblinear_x(Xte)

    rows = []
    x0 = np.zeros(128 * 26 + 26 * 26, dtype=float)

    for x in x_values:
        print(f"\n=== Distorting first {x} transforms ===")
        open(os.path.join(result_dir, f"STARTED_x_{x}.txt"), "w").write("started\n")
        distorted_train = apply_transforms_to_words(
            train_words, transforms, num_lines=x, shape=(8, 16), clip01=True
        )

        # =========================
        # CRF
        # =========================
        t0 = time.time()

        def func(v):
            return objective_and_grad(v, distorted_train, C_crf)

        xopt, nfeval, rc = opt.fmin_tnc(
            func=func, x0=x0, bounds=None, maxfun=200, ftol=1e-3, disp=5
        )
        Wopt, Topt = unpack_params(xopt)
        yhat_words = decode_words(Wopt, Topt, test_words)
        crf_letter_acc = decode_accuracy_letterwise(yhat_words, test_words)
        crf_word_acc = decode_accuracy_wordwise(yhat_words, test_words)

        t1 = time.time()
        crf_seconds = t1 - t0

        # =========================
        # SVM-MC (LibLinear)
        # =========================
        t2 = time.time()

        Xtr, ytr0, _ = words_to_letters(distorted_train)
        ytr = (ytr0 + 1).astype(int)  # 1..26
        xtr_ll = dense_to_liblinear_x(Xtr)

        # Solver choice:
        # -s 4 is Crammer-Singer multiclass SVM (true multiclass).
        # -B 1 adds a bias feature (recommended if you didn't add it yourself).
        param = f"-s 4 -c {C_svm} -B 1"
        model = ll_train(ytr.tolist(), xtr_ll, param)
        ypred, _, _ = ll_predict(yte.tolist(), xte_ll, model, "-q")
        ypred = np.asarray(ypred, dtype=int)

        svm_letter_acc = float(np.mean(ypred == yte))
        svm_word_acc = wordwise_accuracy_from_flat(ypred, yte, test_lens)

        t3 = time.time()
        svm_seconds = t3 - t2

        print(
            f"Done x={x}: "
            f"CRF letter={crf_letter_acc:.6f} word={crf_word_acc:.6f} ({crf_seconds:.1f}s), "
            f"SVM letter={svm_letter_acc:.6f} word={svm_word_acc:.6f} ({svm_seconds:.1f}s)"
        )

        rows.append(
            dict(
                num_transforms=x,
                C_crf=C_crf,
                crf_letter_accuracy=crf_letter_acc,
                crf_word_accuracy=crf_word_acc,
                crf_nfeval=int(nfeval),
                crf_rc=int(rc),
                crf_seconds=round(crf_seconds, 3),
                C_svm=C_svm,
                svm_letter_accuracy=svm_letter_acc,
                svm_word_accuracy=svm_word_acc,
                svm_seconds=round(svm_seconds, 3),
            )
        )

        # Optional warm-start for CRF across x values (speeds up):
        x0 = xopt
        
    import csv
    import matplotlib.pyplot as plt

    outdir = result_dir
    os.makedirs(outdir, exist_ok=True)

    # Save CSV without pandas
    csv_path = os.path.join(outdir, "section5_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Extract series
    rows_sorted = sorted(rows, key=lambda d: d["num_transforms"])
    x = [r["num_transforms"] for r in rows_sorted]
    crf_letter = [r["crf_letter_accuracy"] for r in rows_sorted]
    svm_letter = [r["svm_letter_accuracy"] for r in rows_sorted]
    crf_word = [r["crf_word_accuracy"] for r in rows_sorted]
    svm_word = [r["svm_word_accuracy"] for r in rows_sorted]

    # Plot 5a
    plt.figure(figsize=(7,4))
    plt.plot(x, crf_letter, marker="o", label="CRF")
    plt.plot(x, svm_letter, marker="o", label="SVM-MC")
    plt.xlabel("Number of transforms (first x lines)")
    plt.ylabel("Test letter-wise accuracy")
    plt.title("Section 5a")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "section5a_letter_accuracy.png"), dpi=200)
    plt.close()

    # Plot 5b
    plt.figure(figsize=(7,4))
    plt.plot(x, crf_word, marker="o", label="CRF")
    plt.plot(x, svm_word, marker="o", label="SVM-MC")
    plt.xlabel("Number of transforms (first x lines)")
    plt.ylabel("Test word-wise accuracy")
    plt.title("Section 5b")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "section5b_word_accuracy.png"), dpi=200)
    plt.close()


    out_csv = os.path.join(result_dir, "section5_crf_svmmc_accuracy.csv")
    write_csv(out_csv, rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
