"""NaN-aware statistics over padded feature-history windows."""
import numpy as np


def nan_window_stats(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise NaN-aware mean/std without warnings for padded history."""
    valid = np.isfinite(block)
    count = valid.sum(axis=0)
    total = np.where(valid, block, 0.0).sum(axis=0)
    mean = np.full(block.shape[1], np.nan, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    squared = np.where(valid, (block - mean) ** 2, 0.0).sum(axis=0)
    variance = np.full(block.shape[1], np.nan, dtype=np.float64)
    np.divide(squared, count, out=variance, where=count > 0)
    return mean, np.sqrt(variance)
