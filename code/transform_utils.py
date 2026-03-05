from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Transform:
    kind: str            # 'r' or 't'
    word_index_1based: int
    degrees: Optional[float] = None
    dx: Optional[float] = None
    dy: Optional[float] = None


def parse_transform_file(path: str) -> List[Transform]:
    """
    Parse datatransform.txt with lines like:
      r 317 15
      t 2149 3 3
    where the word index is 1-based (in the order of datatrain.txt). [file:1]
    """
    transforms: List[Transform] = []
    with open(path, "r") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if parts[0] == "r":
                if len(parts) != 3:
                    raise ValueError(f"{path}:{ln}: expected 'r idx deg', got: {s}")
                idx = int(parts[1])
                deg = float(parts[2])
                transforms.append(Transform(kind="r", word_index_1based=idx, degrees=deg))
            elif parts[0] == "t":
                if len(parts) != 4:
                    raise ValueError(f"{path}:{ln}: expected 't idx dx dy', got: {s}")
                idx = int(parts[1])
                dx = float(parts[2])
                dy = float(parts[3])
                transforms.append(Transform(kind="t", word_index_1based=idx, dx=dx, dy=dy))
            else:
                raise ValueError(f"{path}:{ln}: unknown transform kind '{parts[0]}'")
    return transforms


def _get_X(word):
    if hasattr(word, "X"):
        return word.X
    if isinstance(word, dict) and "X" in word:
        return word["X"]
    raise TypeError("Word must have attribute .X or key ['X'].")


def _set_X(word, Xnew):
    if hasattr(word, "X"):
        word.X = Xnew
        return
    if isinstance(word, dict):
        word["X"] = Xnew
        return
    raise TypeError("Word must be a mutable object with .X or a dict with ['X'].")


def _get_y(word):
    if hasattr(word, "y"):
        return word.y
    if isinstance(word, dict) and "y" in word:
        return word["y"]
    raise TypeError("Word must have attribute .y or key ['y'].")


def clone_word(word):
    """
    Shallow clone: keeps y, etc., but allows replacing X.
    """
    try:
        return copy.copy(word)
    except Exception:
        if isinstance(word, dict):
            return dict(word)
        raise


def _ensure_letters_rows(X: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Returns (X_letters_by_rows, transposed_flag).

    Expected downstream CRF code typically uses X shape (m, 128) = (letters, features).
    If given (128, m), we transpose and remember it.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D X, got shape {X.shape}")
    if X.shape[1] == 128:
        return X, False
    if X.shape[0] == 128:
        return X.T, True
    raise ValueError(f"X must be (m,128) or (128,m); got {X.shape}")


def _restore_X_shape(X_letters_by_rows: np.ndarray, was_transposed: bool) -> np.ndarray:
    return X_letters_by_rows.T if was_transposed else X_letters_by_rows


def _rotate_matrix(img: np.ndarray, degrees: float) -> np.ndarray:
    """
    Rotate 2D array around its center without changing size.
    Tries OpenCV first; falls back to scipy.ndimage.rotate.
    """
    try:
        import cv2  # type: ignore

        h, w = img.shape
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, degrees, 1.0)
        out = cv2.warpAffine(
            img.astype(np.float32),
            M,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return out
    except Exception:
        from scipy.ndimage import rotate  # type: ignore

        out = rotate(
            img.astype(np.float32),
            angle=degrees,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        return out.astype(np.float32)


def _translate_matrix(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Translate 2D array by (dx, dy) pixels, keeping same size.
    Tries OpenCV first; falls back to scipy.ndimage.shift.
    """
    try:
        import cv2  # type: ignore

        h, w = img.shape
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        out = cv2.warpAffine(
            img.astype(np.float32),
            M,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return out
    except Exception:
        from scipy.ndimage import shift  # type: ignore

        out = shift(
            img.astype(np.float32),
            shift=(dy, dx),  # (row_shift, col_shift)
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        return out.astype(np.float32)


def transform_letter_vector(
    x128: Union[np.ndarray, Sequence[float]],
    tr: Transform,
    shape: Tuple[int, int] = (8, 16),
    clip01: bool = True,
) -> np.ndarray:
    """
    Apply one Transform to one 128-dim letter vector.

    Important: the lab notes the 128 values are in column-major order when forming 8x16,
    so we reshape/flatten with order='F'. [file:1]
    """
    x = np.asarray(x128, dtype=np.float32).reshape(-1)
    if x.size != shape[0] * shape[1]:
        raise ValueError(f"Expected {shape[0]*shape[1]} dims, got {x.size}")

    img = x.reshape(shape, order="F")

    if tr.kind == "r":
        if tr.degrees is None:
            raise ValueError("Rotation transform missing degrees")
        img2 = _rotate_matrix(img, tr.degrees)
    elif tr.kind == "t":
        if tr.dx is None or tr.dy is None:
            raise ValueError("Translation transform missing dx/dy")
        img2 = _translate_matrix(img, tr.dx, tr.dy)
    else:
        raise ValueError(f"Unknown transform kind '{tr.kind}'")

    if clip01:
        img2 = np.clip(img2, 0.0, 1.0)

    return img2.reshape(-1, order="F").astype(np.float32)


def apply_transforms_to_words(
    words: Sequence,
    transforms: Sequence[Transform],
    num_lines: int,
    shape: Tuple[int, int] = (8, 16),
    clip01: bool = True,
) -> List:
    """
    Apply the first num_lines transforms (in file order) to a COPY of 'words'.
    Each transform applies to *all letters* of the specified word index (1-based). [file:1]
    """
    if num_lines < 0:
        raise ValueError("num_lines must be >= 0")
    num_lines = min(num_lines, len(transforms))

    out_words = [clone_word(w) for w in words]

    for tr in transforms[:num_lines]:
        idx0 = tr.word_index_1based - 1
        if idx0 < 0 or idx0 >= len(out_words):
            raise IndexError(f"Transform word index {tr.word_index_1based} out of range")

        w = out_words[idx0]
        X = _get_X(w)
        X_letters, was_T = _ensure_letters_rows(X)

        Xnew = np.asarray(X_letters, dtype=np.float32).copy()
        for i in range(Xnew.shape[0]):
            Xnew[i, :] = transform_letter_vector(Xnew[i, :], tr, shape=shape, clip01=clip01)

        _set_X(w, _restore_X_shape(Xnew, was_T))

    return out_words


def flatten_letters(words: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten word list into per-letter (X, y) for SVM-MC or computing letter-wise accuracy.
    """
    Xs = []
    ys = []
    for w in words:
        X = _get_X(w)
        y = _get_y(w)

        X_letters, _ = _ensure_letters_rows(np.asarray(X))
        y = np.asarray(y).reshape(-1)

        if X_letters.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch letters: X has {X_letters.shape[0]} rows, y has {y.shape[0]}")

        Xs.append(X_letters)
        ys.append(y)

    Xall = np.vstack(Xs).astype(np.float32)
    yall = np.concatenate(ys)

    # normalize labels to 0..25 if they are 1..26
    if yall.size > 0 and yall.min() >= 1 and yall.max() <= 26:
        yall = yall - 1

    return Xall, yall.astype(int)


def decode_accuracy_wordwise(yhat_words: Sequence[np.ndarray], words: Sequence) -> float:
    """
    Word-wise accuracy: a word is correct iff all letters are correct.
    """
    correct = 0
    for yh, w in zip(yhat_words, words):
        ytrue = np.asarray(_get_y(w)).reshape(-1)
        yh = np.asarray(yh).reshape(-1)

        if ytrue.size > 0 and ytrue.min() >= 1 and ytrue.max() <= 26:
            ytrue = ytrue - 1

        if yh.shape != ytrue.shape:
            raise ValueError("Prediction/label shape mismatch")

        correct += int(np.all(yh == ytrue))
    return correct / max(1, len(words))


def decode_accuracy_letterwise(yhat_words: Sequence[np.ndarray], words: Sequence) -> float:
    """
    Letter-wise accuracy over all letters.
    """
    yh_all = []
    yt_all = []
    for yh, w in zip(yhat_words, words):
        ytrue = np.asarray(_get_y(w)).reshape(-1)
        yh = np.asarray(yh).reshape(-1)

        if ytrue.size > 0 and ytrue.min() >= 1 and ytrue.max() <= 26:
            ytrue = ytrue - 1

        yh_all.append(yh)
        yt_all.append(ytrue)

    yh_all = np.concatenate(yh_all) if yh_all else np.array([], dtype=int)
    yt_all = np.concatenate(yt_all) if yt_all else np.array([], dtype=int)

    if yh_all.size == 0:
        return 0.0
    return float(np.mean(yh_all == yt_all))
