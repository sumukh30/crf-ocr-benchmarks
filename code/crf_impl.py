import numpy as np
import itertools

K = 26
D = 128


def logsumexp(a, axis=None):
    a = np.asarray(a, dtype=float)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def safe_matmul(A, B, name="matmul"):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    with np.errstate(all="ignore"):
        C = A @ B
    if not np.isfinite(C).all():
        raise FloatingPointError(f"{name} produced non-finite values")
    return C


def unpack_params(x, D=D, K=K):
    """
    x layout (matches PDF model layout): w1..w26 then T row-major. [file:1]
    Each wc is length 128, so first 26*128 entries are w1 block, w2 block, ..., w26 block. [file:1]
    """
    x = np.asarray(x, dtype=float).ravel()
    W = x[:D * K].reshape(K, D).T          # (D,K), columns are w_c
    T = x[D * K:].reshape(K, K)           # (K,K), row-major T11,T12,...,T26,26
    return W, T


def pack_params(W, T):
    """
    Inverse of unpack_params: w1..w26 then T row-major. [file:1]
    """
    return np.concatenate([W.T.reshape(-1), T.reshape(-1)])


def node_scores(X, W):
    # X: (m,D), W: (D,K) -> U: (m,K)
    return safe_matmul(X, W, name="U=X@W")


def forward_backward(U, T):
    """
    Compute node and edge marginals using log-space forward-backward for numerical stability.
    Returns:
      node_marg: (m,K)
      edge_marg: (m-1,K,K) where edge_marg[s,i,j] = p(y_s=i, y_{s+1}=j | X)
      logZ: scalar
    """
    U = np.asarray(U, dtype=float)
    m, K = U.shape

    log_alpha = np.empty((m, K), dtype=float)
    log_beta = np.empty((m, K), dtype=float)

    log_alpha[0] = U[0]
    for s in range(1, m):
        log_alpha[s] = U[s] + logsumexp(log_alpha[s - 1][:, None] + T, axis=0)

    logZ = float(logsumexp(log_alpha[m - 1], axis=0))

    log_beta[m - 1] = 0.0
    for s in range(m - 2, -1, -1):
        log_beta[s] = logsumexp(T + (U[s + 1] + log_beta[s + 1])[None, :], axis=1)

    node_marg = np.exp(log_alpha + log_beta - logZ)

    if m > 1:
        edge_marg = np.empty((m - 1, K, K), dtype=float)
        for s in range(m - 1):
            log_edge = (
                log_alpha[s][:, None]
                + T
                + (U[s + 1] + log_beta[s + 1])[None, :]
                - logZ
            )
            edge_marg[s] = np.exp(log_edge)
    else:
        edge_marg = np.zeros((0, K, K), dtype=float)

    return node_marg, edge_marg, logZ


def viterbi(U, T):
    """
    MAP decoding (max-sum / Viterbi) for the linear-chain CRF. [file:1]
    Returns yhat (0..25) and best objective value (sum of node+edge scores). [file:1]
    """
    U = np.asarray(U, dtype=float)
    m, K = U.shape

    dp = np.empty((m, K), dtype=float)
    bp = np.empty((m, K), dtype=int)

    dp[0] = U[0]
    bp[0] = -1

    for s in range(1, m):
        scores = dp[s - 1][:, None] + T
        bp[s] = np.argmax(scores, axis=0)
        dp[s] = U[s] + np.max(scores, axis=0)

    y = np.empty(m, dtype=int)
    y[m - 1] = int(np.argmax(dp[m - 1]))
    best = float(np.max(dp[m - 1]))

    for s in range(m - 1, 0, -1):
        y[s - 1] = bp[s, y[s]]

    return y, best

def sequence_score(U, T, y):
    """
    Compute sum_s U[s, y[s]] + sum_s T[y[s], y[s+1]] for a given label sequence y.
    """
    y = np.asarray(y, dtype=int)
    m = y.size
    score = float(np.sum(U[np.arange(m), y]))
    if m > 1:
        score += float(np.sum(T[y[:-1], y[1:]]))
    return score

def brute_force_decode(U, T):
    """
    Brute-force MAP decoding by enumerating all y in Y^m (only for small m). [file:1]
    Returns (yhat, best_score) where yhat is 0..25.
    """
    U = np.asarray(U, dtype=float)
    m, K = U.shape

    best_y = None
    best_score = -np.inf

    for y in itertools.product(range(K), repeat=m):
        s = sequence_score(U, T, y)
        if s > best_score:
            best_score = s
            best_y = y

    return np.asarray(best_y, dtype=int), float(best_score)

def logp_and_grad_word(word, W, T):
    """
    Compute log p(y|X) and gradients wrt W,T for a single word using marginals. [file:1]
    word: {"X": (m,D), "y": (m,)}
    Returns:
      logp, gW, gT
    """
    X = word["X"]
    y = word["y"]
    m = len(y)

    U = node_scores(X, W)
    node_marg, edge_marg, logZ = forward_backward(U, T)

    score = float(np.sum(U[np.arange(m), y]) + (np.sum(T[y[:-1], y[1:]]) if m > 1 else 0.0))
    logp = score - logZ

    gW = np.zeros_like(W)
    gT = np.zeros_like(T)

    # empirical node counts
    for s in range(m):
        gW[:, y[s]] += X[s]

    # expected node counts
    gW -= safe_matmul(X.T, node_marg, name="X.T@node_marg")

    # empirical/expected edge counts
    if m > 1:
        for s in range(m - 1):
            gT[y[s], y[s + 1]] += 1.0
        gT -= np.sum(edge_marg, axis=0)

    return logp, gW, gT


def objective_and_grad(x, word_list, C):
    """
    Objective from the PDF: -C * (average log-likelihood) + 0.5||W||^2 + 0.5||T||^2. [file:1]
    Returns (f, g) for scipy.optimize routines.
    """
    W, T = unpack_params(x)
    n = len(word_list)

    sum_logp = 0.0
    sum_gW = np.zeros_like(W)
    sum_gT = np.zeros_like(T)

    for w in word_list:
        lp, gW, gT = logp_and_grad_word(w, W, T)
        sum_logp += lp
        sum_gW += gW
        sum_gT += gT

    avg_logp = sum_logp / n
    avg_gW = sum_gW / n
    avg_gT = sum_gT / n

    f = -(C * avg_logp) + 0.5 * np.sum(W * W) + 0.5 * np.sum(T * T)

    gW_total = -(C * avg_gW) + W
    gT_total = -(C * avg_gT) + T
    g = pack_params(gW_total, gT_total)

    return float(f), g


def decode_words(W, T, word_list):
    preds = []
    for w in word_list:
        U = node_scores(w["X"], W)
        yhat, _ = viterbi(U, T)
        preds.append(yhat)
    return preds
