#!/usr/bin/env python3
import sys, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # Lab_1/
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT))

# --- LIBLINEAR high-level API import (two common layouts) ---
try:
    # Some setups expose liblinearutil as a top-level module. 
    from liblinearutil import train, predict
except ModuleNotFoundError:
    # Many installs expose it under the liblinear package. 
    from liblinear.liblinearutil import train, predict

DATA = ROOT / "data"
OUTD = ROOT / "result" / "section3"
OUTD.mkdir(parents=True, exist_ok=True)

TRAIN_STRUCT = DATA / "train_struct.txt"
TEST_STRUCT  = DATA / "test_struct.txt"

def read_struct_xy_qid(path: Path):
    # Each line: <label> qid:<wordid> <feat>:<val> ...
    y, x, qids = [], [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        toks = s.split()
        y.append(int(toks[0]))
        qids.append(int(toks[1].split(":")[1]))
        feats = {}
        for tv in toks[2:]:
            k, v = tv.split(":")
            feats[int(k)] = float(v)
        x.append(feats)
    return y, x, np.array(qids, dtype=int)

def word_acc_by_qid(y_true, y_pred, qids):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    ok, total = 0, 0
    i, n = 0, len(y_true)
    while i < n:
        j = i
        while j < n and qids[j] == qids[i]:
            j += 1
        total += 1
        ok += int(np.all(y_true[i:j] == y_pred[i:j]))
        i = j
    return ok / total

def main():
    cvals = [1, 10, 100, 1000]
    ytr, xtr, _ = read_struct_xy_qid(TRAIN_STRUCT)
    yte, xte, qids_te = read_struct_xy_qid(TEST_STRUCT)

    rows = []
    for c in cvals:
        # Train/predict API: train(y, x, options), predict(y, x, model). [web:280]
        m = train(ytr, xtr, f"-c {c} -q")
        pred_labels, pred_acc, _ = predict(yte, xte, m, "-q")  # pred_acc[0] is accuracy (%) [web:280]
        letter_acc = float(pred_acc[0]) / 100.0
        word_acc = word_acc_by_qid(yte, pred_labels, qids_te)

        rows.append({"model": "SVM-MC", "c": c, "letter_acc": letter_acc, "word_acc": word_acc})
        print(f"[SVM-MC] c={c:>4}  letter_acc={letter_acc:.4f}  word_acc={word_acc:.4f}")

    out_csv = OUTD / "svmmc.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","c","letter_acc","word_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
