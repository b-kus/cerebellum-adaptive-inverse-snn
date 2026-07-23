import numpy as np
from config.constants import GAUSS_CENTERS, GAUSS_SIGMA

def gaussian_activation(x):
    """
    Encode a scalar value using 3 overlapping Gaussian tuning curves.

    Args:
        x: scalar value (e.g., angle, velocity)

    Returns:
        array of 3 firing rates (0-1 scale)
    """
    return np.exp(-0.5 * ((x - GAUSS_CENTERS) / GAUSS_SIGMA) ** 2)