import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from section4_common import (
    try_import_loader, pack_params, unpack_params,
    objective_and_grad_exact, wordwise_error
)
from section4_mcmc import logp_grad_word_mcmc

def objective_and_grad_mcmc_full(x, words, C, S, rb=False, seed=0):
    W, T = unpack_params(x)
    rng = np.random.default_rng(seed)

    n = len(words)
    sum_gW = np.zeros_like(W)
    sum_gT = np.zeros_like(T)

    for w in words:
        _, gW, gT = logp_grad_word_mcmc(w, W, T, S=S, rng=rng, rb=rb)
        sum_gW += gW
        sum_gT += gT

    avg_gW = sum_gW / n
    avg_gT = sum_gT / n

    # Use exact objective value (Eq. 6) for logging/plotting, as required
    f_exact, _ = objective_and_grad_exact(x, words, C)

    gW_total = -(C * avg_gW) + W
    gT_total = -(C * avg_gT) + T
    g = pack_params(gW_total, gT_total)
    return float(f_exact), g

def objective_and_grad_mcmc_batch(x, batch, C, S, rb=False, seed=0):
    W, T = unpack_params(x)
    rng = np.random.default_rng(seed)

    B = len(batch)
    sum_gW = np.zeros_like(W)
    sum_gT = np.zeros_like(T)
    for w in batch:
        _, gW, gT = logp_grad_word_mcmc(w, W, T, S=S, rng=rng, rb=rb)
        sum_gW += gW
        sum_gT += gT
    avg_gW = sum_gW / B
    avg_gT = sum_gT / B

    # objective estimate not needed for SGD step; return exact f for convenience
    f_exact = 0.0 #objective_and_grad_exact(x, batch, C)
    gW_total = -(C * avg_gW) + W
    gT_total = -(C * avg_gT) + T
    return float(f_exact), pack_params(gW_total, gT_total)

def run_sgd_mcmc(train, test, C, S, B=20, lr=0.05, steps=400, eval_every=10, seed=0, momentum=0.0):
    rng = np.random.default_rng(seed)
    n = len(train)
    x = np.zeros(128 * 26 + 26 * 26, dtype=float)
    v = np.zeros_like(x)
    log = []

    for k in range(1, steps + 1):
        idx = rng.integers(0, n, size=B)
        batch = [train[i] for i in idx]
        _, g = objective_and_grad_mcmc_batch(x, batch, C, S=S, rb=False, seed=seed + k)

        if momentum > 0:
            v = momentum * v + lr * g
            x = x - v
        else:
            x = x - lr * g

        if k == 1 or (k % eval_every) == 0:
            eff = (k * B) / n
            f_full, _ = objective_and_grad_exact(x, train, C)
            W, T = unpack_params(x)
            te = wordwise_error(test, W, T)
            log.append((eff, f_full, te))
            tag = "MOM" if momentum > 0 else "SGD"
            print(f"[4b-{tag}] S={S} pass={eff:.3f} obj={f_full:.3f} test_word_err={te:.4f}")

    return x, log

def run_lbfgs_mcmc(train, test, C, S, maxfun=120):
    x0 = np.zeros(128 * 26 + 26 * 26, dtype=float)
    eval_count = {"n": 0}
    hist = []

    def func(x):
        eval_count["n"] += 1
        return objective_and_grad_mcmc_full(x, train, C, S=S, rb=False, seed=eval_count["n"])

    def callback(xk):
        f, _ = objective_and_grad_exact(xk, train, C)
        W, T = unpack_params(xk)
        te = wordwise_error(test, W, T)
        hist.append((eval_count["n"], float(f), te))
        print(f"[4b-LBFGS] S={S} evals={eval_count['n']:>4} obj={float(f):.3f} test_word_err={te:.4f}")

    xopt, nfeval, rc = opt.fmin_tnc(func=func, x0=x0, maxfun=maxfun, ftol=1e-3, disp=0, callback=callback)
    return xopt, hist

def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def plot_curves(path, curves, xlabel, ylabel, title):
    plt.figure()
    for name, xs, ys in curves:
        plt.plot(xs, ys, marker="o", label=name)
    plt.grid(True, ls=":")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=10)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    loadcrfwords = try_import_loader()

    C = 1000
    S = args.S
    train = loadcrfwords(str(ROOT / "data" / "train.txt"))
    test  = loadcrfwords(str(ROOT / "data" / "test.txt"))
    outd = ROOT / "result" / "section4"

    B, lr, mu = 20, 0.01, 0.9
    steps, eval_every = 400, 10

    _, log_sgd = run_sgd_mcmc(train, test, C, S=S, B=B, lr=lr, steps=steps, eval_every=eval_every, seed=0, momentum=0.0)
    _, log_mom = run_sgd_mcmc(train, test, C, S=S, B=B, lr=lr, steps=steps, eval_every=eval_every, seed=0, momentum=mu)
    _, log_lb  = run_lbfgs_mcmc(train, test, C, S=S, maxfun=120)

    write_csv(outd / f"4b_S{S}_sgd.csv",   ["effective_pass","train_objective","test_word_error"], log_sgd)
    write_csv(outd / f"4b_S{S}_mom.csv",   ["effective_pass","train_objective","test_word_error"], log_mom)
    write_csv(outd / f"4b_S{S}_lbfgs.csv", ["obj_evals","train_objective","test_word_error"],     log_lb)

    sgd_x=[r[0] for r in log_sgd]; sgd_f=[r[1] for r in log_sgd]; sgd_e=[r[2] for r in log_sgd]
    mom_x=[r[0] for r in log_mom]; mom_f=[r[1] for r in log_mom]; mom_e=[r[2] for r in log_mom]
    lb_x =[r[0] for r in log_lb];  lb_f =[r[1] for r in log_lb];  lb_e =[r[2] for r in log_lb]

    plot_curves(outd / f"4b_S{S}_train_objective_vs_pass.png",
                [("SGD",sgd_x,sgd_f),("Momentum",mom_x,mom_f),("LBFGS",lb_x,lb_f)],
                "Effective number of passes", "Training objective (Eq. 6)", f"4b: objective vs passes (C={C}, S={S})")

    plot_curves(outd / f"4b_S{S}_test_word_error_vs_pass.png",
                [("SGD",sgd_x,sgd_e),("Momentum",mom_x,mom_e),("LBFGS",lb_x,lb_e)],
                "Effective number of passes", "Test word-wise error", f"4b: test error vs passes (C={C}, S={S})")

if __name__ == "__main__":
    main()
