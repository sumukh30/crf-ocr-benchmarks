import numpy as np
from section4_common import ALPH, get_X, node_scores, normalize_labels

def softmax_log(logits):
    m = np.max(logits)
    z = np.exp(logits - m)
    return z / np.sum(z)

def init_from_T0(U, rng):
    # T=0 => independent p(y_s|X) ∝ exp(U[s,:])
    m = U.shape[0]
    y = np.zeros(m, dtype=int)
    for s in range(m):
        p = softmax_log(U[s])
        y[s] = int(rng.choice(ALPH, p=p))
    return y

def cond_dist(s, y, U, T):
    # p(y_s=c | neighbors, X) ∝ exp(U[s,c] + T[y_{s-1},c] + T[c,y_{s+1}])
    logits = U[s].copy()
    if s - 1 >= 0:
        logits += T[y[s-1], :]
    if s + 1 < len(y):
        logits += T[:, y[s+1]]
    return softmax_log(logits)

def mcmc_marginals(word, W, T, S=10, rng=None, rb=False):
    if rng is None:
        rng = np.random.default_rng(0)

    X = get_X(word)
    U = node_scores(X, W)
    m = U.shape[0]

    node_counts = np.zeros((m, ALPH), dtype=float)
    edge_counts = np.zeros((m-1, ALPH, ALPH), dtype=float)

    y = init_from_T0(U, rng)

    # Each k: update evens, update odds, then treat current y as one sample
    for _ in range(S):
        for parity in (0, 1):
            for s in range(parity, m, 2):
                p = cond_dist(s, y, U, T)

                if rb:
                    node_counts[s] += p
                    if s - 1 >= 0:
                        edge_counts[s-1, y[s-1], :] += p
                    if s + 1 < m:
                        edge_counts[s, :, y[s+1]] += p

                y[s] = int(rng.choice(ALPH, p=p))

        if not rb:
            node_counts[np.arange(m), y] += 1.0
            for s in range(m-1):
                edge_counts[s, y[s], y[s+1]] += 1.0

    if rb:
        # normalize by number of times each position was updated (≈ S)
        node_counts /= S
        edge_counts /= S
        # for safety, renormalize
        node_counts = node_counts / np.sum(node_counts, axis=1, keepdims=True)
        edge_counts = edge_counts / np.sum(edge_counts, axis=(1,2), keepdims=True)
    else:
        node_counts /= S
        edge_counts /= S

    return node_counts, edge_counts

def logp_grad_word_mcmc(word, W, T, S=10, rng=None, rb=False):
    X = get_X(word)
    y = normalize_labels(word["y"] if isinstance(word, dict) and "y" in word else word.y)
    p_node, p_edge = mcmc_marginals(word, W, T, S=S, rng=rng, rb=rb)

    gW = np.zeros_like(W)
    for i, yi in enumerate(y):
        gW[:, yi] += X[i]
    gW -= X.T @ p_node

    gT = np.zeros_like(T)
    if len(y) > 1:
        for i in range(len(y) - 1):
            gT[y[i], y[i+1]] += 1.0
        gT -= np.sum(p_edge, axis=0)

    # We don’t need lp for optimization here; return 0 as placeholder
    return 0.0, gW, gT
