import numpy as np

ALPH = 26 # number of possible labels (letters a-z)
D = 128 # dimensionality of pixel feature vector

"""
    Attempt to import the dataset loader function.

    The loader reads CRF word data and returns a list of word objects
    containing:
        - X : feature matrix (letters × features)
        - y : label sequence

    This wrapper ensures compatibility with different environments.
    """
def try_import_loader():
    try:
        from data_io import load_crf_words
        return load_crf_words
    except Exception:
        from data_io import load_crf_words
        return load_crf_words

#Extract feature matrix X from a word object.
def get_X(word):
    if isinstance(word, dict):
        for k in ("X", "x", "pixels", "features"):
            if k in word:
                return np.asarray(word[k], dtype=np.float64)
        raise KeyError(f"Word dict missing X/x. Keys={list(word.keys())[:30]}")
    for k in ("X", "x"):
        if hasattr(word, k):
            return np.asarray(getattr(word, k))
    raise AttributeError("Word missing X/x attribute")

#Extract label sequence y from the word object.
def get_y(word):
    if isinstance(word, dict):
        for k in ("y", "Y", "label", "labels"):
            if k in word:
                return np.asarray(word[k])
        raise KeyError(f"Word dict missing y/Y. Keys={list(word.keys())[:30]}")
    for k in ("y", "Y"):
        if hasattr(word, k):
            return np.asarray(getattr(word, k))
    raise AttributeError("Word missing y/Y attribute")

"""
    Flatten CRF parameters into a single vector.

    W : emission weights (128 × 26)
    T : transition weights (26 × 26)

    Required for optimization routines such as L-BFGS.
    """
def normalize_labels(arr):
    arr = np.asarray(arr).reshape(-1)
    if arr.dtype.kind in ("U", "S", "O"):
        out = np.array([ord(str(a)) - ord("a") for a in arr], dtype=int)
        return out
    arr = arr.astype(int)
    # allow 1..26 or 0..25
    if arr.size and arr.min() >= 1 and arr.max() <= 26:
        return arr - 1
    return arr

def pack_params(W, T):
    return np.concatenate([W.reshape(-1), T.reshape(-1)])

D, K = 128, 26

"""
    Convert flattened parameter vector back into matrices.

    x → [W | T]

    Returns:
        W : emission weights
        T : transition weights
    """
def unpack_params(x):
    x = np.asarray(x, dtype=np.float64)
    W = x[:D*K].reshape(D, K).copy()
    T = x[D*K:].reshape(K, K).copy()
    return W, T

"""
    Numerically stable computation of:

        log(sum(exp(a)))

    Used extensively in CRF forward-backward inference
    to avoid floating point overflow.
    """
def logsumexp(a, axis=None):
    a = np.asarray(a)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if axis is None:
        return float(out.reshape(()))
    return out.squeeze(axis)

"""
    Compute emission scores for each label at each position.

    U[i, y] = score of assigning label y to letter i.

    Computed as:
        U = X @ W

    X : (m × 128)
    W : (128 × 26)
    U : (m × 26)
"""
def node_scores(X, W):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    if not np.all(np.isfinite(X)):
        raise FloatingPointError(f"X non-finite: min={np.nanmin(X)} max={np.nanmax(X)}")
    if not np.all(np.isfinite(W)):
        raise FloatingPointError(f"W non-finite: min={np.nanmin(W)} max={np.nanmax(W)}")

    # print("node_scores:",
    #     "X", X.shape, "X[min,max]=", float(np.min(X)), float(np.max(X)),
    #     "W", W.shape, "W[maxabs]=", float(np.max(np.abs(W))))
    return X @ W

"""
    Perform forward-backward inference for a linear-chain CRF.

    Computes:
        logZ    : log partition function
        p_node  : marginal probability of each label
        p_edge  : marginal probability of label transitions

    U : emission scores
    T : transition scores
    """
def forward_backward(U, T):
    # U: (m,26), T: (26,26)
    m = U.shape[0]
    la = np.zeros((m, ALPH), dtype=float)
    lb = np.zeros((m, ALPH), dtype=float)

    la[0] = U[0]
    for i in range(1, m):
        la[i] = U[i] + logsumexp(la[i-1][:, None] + T, axis=0)

    lb[m-1] = 0.0
    for i in range(m-2, -1, -1):
        lb[i] = logsumexp(T + U[i+1][None, :] + lb[i+1][None, :], axis=1)

    logZ = logsumexp(la[m-1], axis=0)

    logp_node = la + lb - logZ
    p_node = np.exp(logp_node)
    p_node = p_node / np.sum(p_node, axis=1, keepdims=True)

    p_edge = np.zeros((m-1, ALPH, ALPH), dtype=float)
    for i in range(m-1):
        logp_e = la[i][:, None] + T + U[i+1][None, :] + lb[i+1][None, :] - logZ
        pe = np.exp(logp_e)
        pe = pe / np.sum(pe)
        p_edge[i] = pe

    return logZ, p_node, p_edge

def logp_and_grad_word_exact(word, W, T):
    X = get_X(word)
    y = normalize_labels(get_y(word))
    U = node_scores(X, W)
    logZ, p_node, p_edge = forward_backward(U, T)

    score = float(np.sum(U[np.arange(len(y)), y]))
    if len(y) > 1:
        score += float(np.sum(T[y[:-1], y[1:]]))
    lp = score - logZ

    gW = np.zeros_like(W)
    for i, yi in enumerate(y):
        gW[:, yi] += X[i]
    gW -= X.T @ p_node  # expected node feature

    gT = np.zeros_like(T)
    if len(y) > 1:
        for i in range(len(y) - 1):
            gT[y[i], y[i+1]] += 1.0
        gT -= np.sum(p_edge, axis=0)

    return float(lp), gW, gT

def objective_and_grad_exact(x, word_list, C):
    W, T = unpack_params(x)
    n = len(word_list)

    sum_logp = 0.0
    sum_gW = np.zeros_like(W)
    sum_gT = np.zeros_like(T)

    for w in word_list:
        lp, gW, gT = logp_and_grad_word_exact(w, W, T)
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

def viterbi_decode(U, T):
    m = U.shape[0]
    dp = np.zeros((m, ALPH), dtype=float)
    bp = np.zeros((m, ALPH), dtype=int)

    dp[0] = U[0]
    for i in range(1, m):
        scores = dp[i-1][:, None] + T
        bp[i] = np.argmax(scores, axis=0)
        dp[i] = U[i] + np.max(scores, axis=0)

    y = np.zeros(m, dtype=int)
    y[m-1] = int(np.argmax(dp[m-1]))
    for i in range(m-2, -1, -1):
        y[i] = bp[i+1, y[i+1]]
    return y

def wordwise_error(words, W, T):
    bad = 0
    for w in words:
        X = get_X(w)
        y_true = normalize_labels(get_y(w))
        U = node_scores(X, W)
        y_hat = viterbi_decode(U, T)
        bad += int(not np.all(y_true == y_hat))
    return bad / len(words)
