#!/usr/bin/env python3
# Script is for plotting section 3 figures by ingesting respective csv files
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "result" / "section3"

def read_csv(path: Path): #reads csv files
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "model": r["model"],
                "c": float(r["c"]),
                "letter_acc": float(r["letter_acc"]),
                "word_acc": float(r["word_acc"]),
            })
    return rows

def plot_model(rows, model, metric):
    pts = sorted([r for r in rows if r["model"] == model], key=lambda r: r["c"])
    xs = [p["c"] for p in pts]
    ys = [p[metric] for p in pts]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel("-c")
    plt.ylabel(metric)
    plt.title(f"{model}: {metric} vs -c")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    out = OUTD / f"{model}_{metric}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")

def main():
    rows = []
    rows += read_csv(OUTD / "svmstruct.csv")
    rows += read_csv(OUTD / "svmmc.csv")
    rows += read_csv(OUTD / "crf.csv")

    for model in ["CRF", "SVM-Struct", "SVM-MC"]:
        plot_model(rows, model, "letter_acc")
        plot_model(rows, model, "word_acc")

if __name__ == "__main__":
    main()
