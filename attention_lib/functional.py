"""Basic numerically-stable numpy primitives used throughout attention_lib."""
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the given axis.

    Subtracts the max along `axis` before exponentiating so that large
    logits do not overflow float32/float64 exponentials.
    """
    x = np.asarray(x, dtype=np.float64) if x.dtype == object else np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / denom
