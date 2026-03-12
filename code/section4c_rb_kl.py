#This file generates the answers for question 4c


import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from section4_common import (
    try_import_loader, unpack_params, pack_params,
    get_X, node_scores, forward_backward
)
from section4_mcmc import mcmc_marginals

# ---------------------------------------------------------
# KL Divergence computation
#
# Computes KL(p || q) between two probability distributions
# Small epsilon added for numerical stability
# ---------------------------------------------------------
def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

# ---------------------------------------------------------
# Load model parameters from file
# Returns flattened parameter vector
# ---------------------------------------------------------
def load_model_vec(path):
    v = np.loadtxt(path, dtype=float).reshape(-1)
    return v

def main():
    ROOT = Path(__file__).resolve().parents[1]
    loadcrfwords = try_import_loader()

    model_path = ROOT / "data" / "model.txt"
    train_path = ROOT / "data" / "train.txt"

    x = load_model_vec(model_path)
    W, T = unpack_params(x)

    train = loadcrfwords(str(train_path))
    w0 = train[0]

    # ---------------------------------------------------------
    # Compute TRUE marginals using Forward-Backward
    # ---------------------------------------------------------
    U = node_scores(get_X(w0), W)
    _, p_node_true, p_edge_true = forward_backward(U, T)

    max_samples = 200
    xs = list(range(1, max_samples + 1))

    node_kl_hard = []
    edge_kl_hard = []
    node_kl_rb = []
    edge_kl_rb = []

    rng = np.random.default_rng(0)

    # ---------------------------------------------------------
    # Evaluate KL divergence for increasing sample sizes
    # ---------------------------------------------------------
    for S in xs:
        rng_h = np.random.default_rng(0)
        rng_r = np.random.default_rng(0)
        p_node_h, p_edge_h = mcmc_marginals(w0, W, T, S=S, rng=rng_h, rb=False)
        p_node_r, p_edge_r = mcmc_marginals(w0, W, T, S=S, rng=rng_r, rb=True)

        node_kl_hard.append(sum(kl(p_node_true[i], p_node_h[i]) for i in range(p_node_true.shape[0])))
        node_kl_rb.append(sum(kl(p_node_true[i], p_node_r[i]) for i in range(p_node_true.shape[0])))

        edge_kl_hard.append(sum(kl(p_edge_true[i].reshape(-1), p_edge_h[i].reshape(-1)) for i in range(p_edge_true.shape[0])))
        edge_kl_rb.append(sum(kl(p_edge_true[i].reshape(-1), p_edge_r[i].reshape(-1)) for i in range(p_edge_true.shape[0])))

        if S in (1, 5, 10, 50, 100, 200):
            print(f"S={S:>3} nodeKL(hard)={node_kl_hard[-1]:.3f} nodeKL(RB)={node_kl_rb[-1]:.3f}")

    outd = ROOT / "result" / "section4" #saving results
    outd.mkdir(parents=True, exist_ok=True)

    with open(outd / "4c_kl.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S","nodeKL_hard","edgeKL_hard","nodeKL_RB","edgeKL_RB"])
        for i, S in enumerate(xs):
            w.writerow([S, node_kl_hard[i], edge_kl_hard[i], node_kl_rb[i], edge_kl_rb[i]])

    plt.figure()
    plt.plot(xs, node_kl_hard, label="Node KL (hard)")
    plt.plot(xs, edge_kl_hard, label="Edge KL (hard)")
    plt.plot(xs, node_kl_rb,   label="Node KL (RB)")
    plt.plot(xs, edge_kl_rb,   label="Edge KL (RB)")
    plt.yscale("log")
    plt.grid(True, ls=":")
    plt.xlabel("#samples (S)")
    plt.ylabel("KL divergence sum (log scale)")
    plt.title("4c: KL vs #samples (with/without Rao–Blackwellization)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outd / "4c_kl_rb.png", dpi=200)
    plt.close()

    print(f"Wrote {outd / '4c_kl_rb.png'} and {outd / '4c_kl.csv'}")

if __name__ == "__main__":
    main()
