from extra_io import load_decode_input
from crf_impl import node_scores, viterbi


def main():
    X100, W, T = load_decode_input("../data/decode_input.txt", m=100)
    U100 = node_scores(X100, W)
    _, best_obj = viterbi(U100, T)

    print("Maximum objective value:")
    print(best_obj)


if __name__ == "__main__":
    main()
    