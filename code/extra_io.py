import numpy as np

def load_column_vector_txt(path):
    # Each line is one number.
    return np.loadtxt(path, dtype=float).ravel()

def load_model_params(path, D=128, K=26):
    """
    model layout: w1..w26 then T11..T26,26 as a column vector. [file:1]
    Each wc is length D, so w-block length is D*K. [file:1]
    """
    v = load_column_vector_txt(path)
    W = v[:D*K].reshape(K, D).T     # (D,K)
    T = v[D*K:].reshape(K, K)       # (K,K), row-major
    return W, T

def load_decode_input(path, D=128, K=26, m=100):
    """
    decode_input layout: x1..xm, w1..w26, T11..T26,26 as a column vector. [file:1]
    """
    v = load_column_vector_txt(path)
    n_x = m * D
    n_w = K * D
    n_t = K * K
    expected = n_x + n_w + n_t
    if v.size != expected:
        raise ValueError(f"{path}: length {v.size}, expected {expected}")

    X = v[:n_x].reshape(m, D)
    W = v[n_x:n_x+n_w].reshape(K, D).T
    T = v[n_x+n_w:].reshape(K, K)
    return X, W, T
