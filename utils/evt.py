"""
Extreme Value Theory (EVT) for automatic threshold selection.
"""

import numpy as np
from scipy import stats
from typing import Tuple


def fit_weibull(scores: np.ndarray, tail_size: float = 0.1) -> Tuple[float, float, float]:
    """
    Fit Weibull distribution to the tail of anomaly scores.

    Args:
        scores: 1D array of anomaly scores
        tail_size: Fraction of data to use for tail fitting (e.g., 0.1 for top 10%)

    Returns:
        Tuple of (shape, location, scale) parameters
    """
    # Select tail data (highest scores)
    n_tail = max(int(len(scores) * tail_size), 10)
    tail_data = np.sort(scores)[-n_tail:]

    # Fit Weibull distribution to tail
    shape, loc, scale = stats.weibull_min.fit(tail_data, floc=0)

    return shape, loc, scale


def compute_evt_threshold(
    scores: np.ndarray,
    tail_size: float = 0.1,
    quantile: float = 0.95
) -> float:
    """
    Compute anomaly threshold using Extreme Value Theory.

    Args:
        scores: 1D array of anomaly scores from normal training data
        tail_size: Fraction of data to use for tail fitting
        quantile: Quantile for threshold (e.g., 0.95 for 95th percentile)

    Returns:
        Threshold value
    """
    # Fit Weibull to tail
    shape, loc, scale = fit_weibull(scores, tail_size)

    # Compute threshold at desired quantile
    threshold = stats.weibull_min.ppf(quantile, shape, loc=loc, scale=scale)

    return threshold


def compute_percentile_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    """
    Compute threshold as a percentile of scores.

    Args:
        scores: 1D array of anomaly scores
        percentile: Percentile value (0-100)

    Returns:
        Threshold value
    """
    return np.percentile(scores, percentile)


def compute_adaptive_threshold(
    scores: np.ndarray,
    window_size: int = 100,
    num_stds: float = 3.0
) -> np.ndarray:
    """
    Compute adaptive threshold using moving statistics.

    Args:
        scores: 1D array of anomaly scores
        window_size: Size of moving window
        num_stds: Number of standard deviations for threshold

    Returns:
        Array of adaptive thresholds
    """
    n = len(scores)
    thresholds = np.zeros(n)

    for i in range(n):
        start = max(0, i - window_size)
        window = scores[start:i+1]

        if len(window) > 1:
            mean = np.mean(window)
            std = np.std(window)
            thresholds[i] = mean + num_stds * std
        else:
            thresholds[i] = scores[i]

    return thresholds


def select_threshold(
    scores: np.ndarray,
    method: str = 'evt',
    **kwargs
) -> float:
    """
    Select threshold using specified method.

    Args:
        scores: 1D array of anomaly scores
        method: 'evt' | 'percentile' | 'fixed' | 'adaptive'
        **kwargs: Method-specific parameters

    Returns:
        Threshold value (or array for adaptive)
    """
    if method == 'evt':
        tail_size = kwargs.get('tail_size', 0.1)
        quantile = kwargs.get('quantile', 0.95)
        return compute_evt_threshold(scores, tail_size, quantile)

    elif method == 'percentile':
        percentile = kwargs.get('percentile', 95.0)
        return compute_percentile_threshold(scores, percentile)

    elif method == 'fixed':
        return kwargs.get('fixed_value', 1.5)

    elif method == 'adaptive':
        window_size = kwargs.get('window_size', 100)
        num_stds = kwargs.get('num_stds', 3.0)
        return compute_adaptive_threshold(scores, window_size, num_stds)

    else:
        raise ValueError(f"Unknown threshold method: {method}")


if __name__ == "__main__":
    # Test EVT threshold selection
    print("Testing EVT threshold selection...")

    # Generate synthetic scores (mostly normal with some anomalies)
    np.random.seed(42)
    normal_scores = np.random.beta(2, 5, size=900)  # Skewed toward low values
    anomaly_scores = np.random.beta(5, 2, size=100)  # Skewed toward high values
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    np.random.shuffle(all_scores)

    print(f"Scores shape: {all_scores.shape}")
    print(f"Scores range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"Mean: {all_scores.mean():.4f}, Std: {all_scores.std():.4f}")

    # Test EVT threshold
    evt_threshold = select_threshold(all_scores, method='evt', tail_size=0.1, quantile=0.95)
    print(f"\nEVT threshold (95th percentile): {evt_threshold:.4f}")

    # Test percentile threshold
    perc_threshold = select_threshold(all_scores, method='percentile', percentile=95)
    print(f"Percentile threshold (95th): {perc_threshold:.4f}")

    # Test fixed threshold
    fixed_threshold = select_threshold(all_scores, method='fixed', fixed_value=1.5)
    print(f"Fixed threshold: {fixed_threshold:.4f}")

    # Test adaptive threshold
    adaptive_thresholds = select_threshold(all_scores, method='adaptive', window_size=50, num_stds=2.0)
    print(f"Adaptive thresholds shape: {adaptive_thresholds.shape}")
    print(f"Adaptive thresholds range: [{adaptive_thresholds.min():.4f}, {adaptive_thresholds.max():.4f}]")

    # Compute detection rates
    for name, threshold in [('EVT', evt_threshold), ('Percentile', perc_threshold)]:
        detections = all_scores > threshold
        detection_rate = detections.sum() / len(all_scores)
        print(f"\n{name} detection rate: {detection_rate:.2%}")

    print("\nEVT tests passed!")
