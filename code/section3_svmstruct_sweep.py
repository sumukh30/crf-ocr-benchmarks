#!/usr/bin/env python3
# Script to train and evaluate Structured SVM (SVM-HMM) models
# for different regularization parameters C and report
# letter-level and word-level accuracy.

import sys, csv, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # Lab_1/
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
OUTD = ROOT / "result" / "section3"
OUTD.mkdir(parents=True, exist_ok=True)

TRAIN = DATA / "train_struct.txt"
TEST  = DATA / "test_struct.txt"

# Locate the SVM-HMM binaries inside third_party_tools
def find_svmhmm_dir():
    base = ROOT / "third_party_tools"
    if (base / "svm_hmm").exists():
        return base / "svm_hmm"
    matches = sorted(base.glob("svm_hmm*"))
    if not matches:
        raise FileNotFoundError("Could not find SVMhmm folder under third_party_tools/")
    return matches[0]

SVMHMM_DIR = find_svmhmm_dir()
LEARN = SVMHMM_DIR / "svm_hmm_learn"
CLAS  = SVMHMM_DIR / "svm_hmm_classify"

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout)
    

# Read true labels and query IDs (word identifiers) from dataset
# Each query ID groups letters belonging to the same word
def read_true_labels_and_qids(path: Path):
    y, q = [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        toks = s.split()
        y.append(int(toks[0]))
        q.append(int(toks[1].split(":")[1]))
    return y, q

#read predicted labels
def read_pred_labels(path: Path):
    y = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        y.append(int(s.split()[0]))
    return y

# Compute letter-wise accuracy
# Measures fraction of correctly predicted letters
def letter_acc(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

def word_acc_by_qid(y_true, y_pred, qids):
    ok_words, total_words = 0, 0
    i, n = 0, len(y_true)
    while i < n:
        j = i
        while j < n and qids[j] == qids[i]:
            j += 1
        total_words += 1
        ok_words += int(all(y_true[k] == y_pred[k] for k in range(i, j)))
        i = j
    return ok_words / total_words

def main():
    cvals = [1, 10, 100, 1000]
    y_true, qids = read_true_labels_and_qids(TEST)

    rows = []
    for c in cvals:
        model = OUTD / f"svmstruct_c{c}.model"
        pred  = OUTD / f"svmstruct_c{c}.pred"
        run([str(LEARN), "-c", str(c), str(TRAIN), str(model)])
        run([str(CLAS),  str(TEST), str(model), str(pred)])

        y_pred = read_pred_labels(pred)
        la = letter_acc(y_true, y_pred)
        wa = word_acc_by_qid(y_true, y_pred, qids)
        rows.append({"model": "SVM-Struct", "c": c, "letter_acc": la, "word_acc": wa})
        print(f"[SVM-Struct] c={c:>4}  letter_acc={la:.4f}  word_acc={wa:.4f}")

    out_csv = OUTD / "svmstruct.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","c","letter_acc","word_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
