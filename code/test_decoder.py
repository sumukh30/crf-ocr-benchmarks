import numpy as np
from crf_impl import node_scores, viterbi, brute_force_decode

# small test sequence
m = 3
K = 26
D = 128

# random synthetic data
X = np.random.randn(m, D)
W = np.random.randn(D, K)
T = np.random.randn(K, K)

# compute node scores
U = node_scores(X, W)

# run Viterbi
y_viterbi, score_v = viterbi(U, T)

# run brute force
y_brute, score_b = brute_force_decode(U, T)

print("Viterbi score:", score_v)
print("Brute force score:", score_b)

if abs(score_v - score_b) < 1e-6:
    print("Test passed: Viterbi matches brute force")
else:
    print("Test failed")