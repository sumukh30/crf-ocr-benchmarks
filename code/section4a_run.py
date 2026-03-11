#This file generates the answers for question 4a


import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

np.seterr(over="raise", invalid="raise", divide="ignore", under="ignore")
from section4_common import (
    try_import_loader,
    unpack_params,
    objective_and_grad_exact,
    wordwise_error,
)

D = 128
K = 26
P = D * K + K * K


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def plot_curves(path, curves, xlabel, ylabel, title):
    plt.figure()
    for name, xs, ys in curves:
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=name)
    plt.grid(True, ls=":")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def sanity_check_words(words, name="train"):
    # The dataset encodes each letter as 128 pixels of 0/1, and labels are 26-way. [file:1][file:218]
    w0 = words[0]
    X = np.asarray(w0["X"], dtype=np.float64)
    y = np.asarray(w0["y"], dtype=int)

    if X.ndim != 2 or X.shape[1] != D:
        raise ValueError(f"{name}[0]['X'] has shape {X.shape}; expected (m,{D}).")

    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"{name}[0]['y'] shape {y.shape} incompatible with X {X.shape}.")

    xmn, xmx = float(X.min()), float(X.max())
    if xmn < -1e-12 or xmx > 1 + 1e-12:
        raise ValueError(
            f"{name}[0]['X'] not binary-like; min={xmn}, max={xmx}. "
            "Your loader may be including non-pixel fields."
        )

    if y.min() < 0 or y.max() >= K:
        raise ValueError(f"{name}[0]['y'] out of range [0,{K-1}].")


def clip_grad_norm(g, max_norm):
    gn = np.linalg.norm(g)
    if not np.isfinite(gn):
        raise FloatingPointError("Gradient norm is non-finite.")
    if gn > max_norm:
        g = g * (max_norm / (gn + 1e-12))
    return g


def run_sgd(train, test, C, *, B=20, lr=1e-4, steps=2000, eval_every=50, seed=0, momentum=0.9, max_norm=1.0):
    rng = np.random.default_rng(seed)
    n = len(train)

    x = np.zeros(P, dtype=np.float64)
    v = np.zeros_like(x)

    log = []

    for k in range(1, steps + 1):
        idx = rng.integers(0, n, size=B)
        batch = [train[i] for i in idx]

        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f"x became non-finite at iter {k}")
        if np.max(np.abs(x)) > 1e6:
            raise FloatingPointError(f"x exploded (max|x|={np.max(np.abs(x))}) at iter {k}")

        # Mini-batch stochastic objective/grad as required in 4a (sample B words). [file:1]
        _, g = objective_and_grad_exact(x, batch, C)

        if not np.all(np.isfinite(g)):
            raise FloatingPointError(f"g became non-finite at iter {k}")

        g = clip_grad_norm(g, max_norm=max_norm)

        # Correct minimization update (SGD + momentum). [file:1]
        if momentum > 0:
            v = momentum * v + g
            x = x - lr * v
        else:
            x = x - lr * g

        if (k == 1) or (k % eval_every == 0):
            eff_pass = (k * B) / n  # effective passes kB/n [file:1]
            f_full, _ = objective_and_grad_exact(x, train, C)  # objective Eq. 6 on full training set [file:1]
            W, T = unpack_params(x)
            te = wordwise_error(test, W, T)  # word-wise test error (required plot) [file:1]
            log.append((float(eff_pass), float(f_full), float(te)))

            tag = "MOM" if momentum > 0 else "SGD"
            print(f"[{tag}] pass={eff_pass:.3f}  obj={float(f_full):.3f}  test_word_err={te:.4f}")

    return x, log


def run_lbfgs(train, test, C, *, maxfun=200):
    x0 = np.zeros(P, dtype=np.float64)

    eval_count = {"n": 0}
    hist = []

    # fmin_tnc’s callback doesn’t give eval count; the lab suggests a counter hack. [file:1]
    def func(x):
        eval_count["n"] += 1
        return objective_and_grad_exact(x, train, C)

    def callback(xk):
        f, _ = objective_and_grad_exact(xk, train, C)  # recompute (lab warns buffering may be off). [file:1]
        W, T = unpack_params(xk)
        te = wordwise_error(test, W, T)
        eff_pass = eval_count["n"]  # LBFGS effective passes = #objective evals. [file:1]
        hist.append((float(eff_pass), float(f), float(te)))
        print(f"[LBFGS] evals={eval_count['n']:>4}  obj={float(f):.3f}  test_word_err={te:.4f}")

    xopt, nfeval, rc = opt.fmin_tnc(func=func, x0=x0, maxfun=maxfun, ftol=1e-3, disp=0, callback=callback)
    return xopt, hist, nfeval, rc


def main():
    ROOT = Path(__file__).resolve().parents[1]
    loadcrfwords = try_import_loader()

    # The lab uses C=1000 in training (earlier section); 4a says fix C to best from previous section. [file:1][file:168]
    C = 1000

    train = loadcrfwords(str(ROOT / "data" / "train.txt"))
    test  = loadcrfwords(str(ROOT / "data" / "test.txt"))

    sanity_check_words(train, "train")
    sanity_check_words(test, "test")

    outd = ROOT / "result" / "section4"
    outd.mkdir(parents=True, exist_ok=True)

    # Hyperparameters chosen to be stable; tune if you want faster decay. [file:1]
    B = 50
    steps = 10000
    eval_every = 200

    # SGD
    x_sgd, log_sgd = run_sgd(train, test, C, B=B, lr = 5e-4, steps=steps, eval_every=eval_every,
                            seed=0, momentum=0.0, max_norm=5.0)

    # Momentum
    x_mom, log_mom = run_sgd(train, test, C, B=B, lr = 2e-4, steps=steps, eval_every=eval_every,
                            seed=0, momentum=0.9, max_norm=5.0)

    # LBFGS
    x_lb, log_lb, nfeval, rc = run_lbfgs(train, test, C, maxfun=200)
    print(f"LBFGS done: nfeval={nfeval}, rc={rc}")

    # Save CSVs
    write_csv(outd / "4a_sgd.csv",   ["effective_pass", "train_objective", "test_word_error"], log_sgd)
    write_csv(outd / "4a_mom.csv",   ["effective_pass", "train_objective", "test_word_error"], log_mom)
    write_csv(outd / "4a_lbfgs.csv", ["effective_pass", "train_objective", "test_word_error"], log_lb)

    # Plot (two required figures, three curves each). [file:1]
    sgd_x = [r[0] for r in log_sgd]; sgd_f = [r[1] for r in log_sgd]; sgd_e = [r[2] for r in log_sgd]
    mom_x = [r[0] for r in log_mom]; mom_f = [r[1] for r in log_mom]; mom_e = [r[2] for r in log_mom]
    lb_x  = [r[0] for r in log_lb];  lb_f  = [r[1] for r in log_lb];  lb_e  = [r[2] for r in log_lb]

    plot_curves(outd / "4a_train_objective_vs_pass.png",
                [("SGD", sgd_x, sgd_f), ("Momentum", mom_x, mom_f), ("LBFGS", lb_x, lb_f)],
                "Effective number of passes", "Training objective (Eq. 6)",
                f"4a: objective vs effective passes (C={C})")

    plot_curves(outd / "4a_test_word_error_vs_pass.png",
                [("SGD", sgd_x, sgd_e), ("Momentum", mom_x, mom_e), ("LBFGS", lb_x, lb_e)],
                "Effective number of passes", "Test word-wise error",
                f"4a: test word-wise error vs effective passes (C={C})")


if __name__ == "__main__":
    main()
