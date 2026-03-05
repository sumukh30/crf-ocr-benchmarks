import numpy as np
LETTER_TO_INT = {chr(ord('a') + i): i for i in range(26)}

def load_crf_words(path):
    by_word = {}
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if not p: 
                continue
            letter = p[1]
            word_id = int(p[3])
            pos = int(p[4])
            pix = np.array(list(map(int, p[5:])), dtype=float)
            if pix.size != 128:
                raise ValueError(f"Expected 128 pixels, got {pix.size}")
            y = LETTER_TO_INT[letter]
            by_word.setdefault(word_id, []).append((pos, pix, y))

    words = []
    for wid, items in sorted(by_word.items()):
        items.sort(key=lambda t: t[0])
        X = np.stack([it[1] for it in items], axis=0)
        y = np.array([it[2] for it in items], dtype=int)
        words.append({"X": X, "y": y, "word_id": wid})
    return words
